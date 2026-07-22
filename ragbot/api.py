"""
api.py — FastAPI web server with SSE streaming and optional TTS narration.

Endpoints:
  POST /chat  — multiplexed SSE stream: { type: "text"|"audio"|"error"|"done" }
  POST /clear — clears chat history for a session
  GET  /      — serves the frontend (index.html)
  GET  /audioPlayer.js — serves the audio queue player JS

Session history is persisted per-session in SQLite (memory3.db).

SSE Event Contract
------------------
Each event is a JSON object on a single line (newline-delimited JSON):
  {"type": "text",  "content": "<token>"}
  {"type": "audio", "index": <int>, "audio": "<base64-mp3>"}
  {"type": "error", "stage": "<stage>", "message": "<msg>"}
  {"type": "done"}
"""

import asyncio
import json
import os
import time

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse, Response
from pydantic import BaseModel
from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage

from ragbot.config import MEMORY_DB, TTS_AVAILABLE, TTS_MAX_SENTENCES
from ragbot.agent import run_agent
from ragbot.sentence_buffer import SentenceBuffer
from ragbot import tts_service

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _sse(obj: dict) -> str:
    """Serialise a dict as a newline-terminated JSON line."""
    return json.dumps(obj, ensure_ascii=False) + "\n"


def create_app(chunks: list, retriever) -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(title="Agentic RAG Chatbot — Port 8003")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Static file routes ─────────────────────────────────────────────────────
    @app.get("/production_tables.css")
    def serve_css():
        return FileResponse(os.path.join(_PROJECT_ROOT, "production_tables.css"))

    @app.get("/audioPlayer.js")
    def serve_audio_player():
        path = os.path.join(_PROJECT_ROOT, "audioPlayer.js")
        if os.path.exists(path):
            return FileResponse(path, media_type="application/javascript")
        return Response(status_code=404)

    @app.get("/favicon.ico")
    def serve_favicon():
        path = os.path.join(_PROJECT_ROOT, "favicon.ico")
        if os.path.exists(path):
            return FileResponse(path)
        return Response(status_code=204)

    # ── Request model ──────────────────────────────────────────────────────────
    class ChatRequest(BaseModel):
        session_id: str
        message: str
        voice_enabled: bool = False

    # ── /chat ──────────────────────────────────────────────────────────────────
    @app.post("/chat")
    async def chat(req: ChatRequest):
        async def response_generator():
            history = SQLChatMessageHistory(
                session_id=req.session_id, connection_string=MEMORY_DB,
            )

            # Effective voice flag: also gated by server-side TTS availability
            voice = req.voice_enabled and TTS_AVAILABLE

            t_start = time.perf_counter()
            buf = SentenceBuffer()
            tts_tasks: list[tuple[int, asyncio.Task]] = []  # (index, task)
            sentence_index = 0
            tts_cap_warned = False

            # ── Helper: schedule one TTS task ──────────────────────────────
            def _schedule_tts(sentence_text: str) -> asyncio.Task | None:
                nonlocal sentence_index, tts_cap_warned
                if not voice:
                    return None
                clean = tts_service.sanitize_for_tts(sentence_text)
                if not clean:
                    return None
                if sentence_index >= TTS_MAX_SENTENCES:
                    return None
                idx = sentence_index
                sentence_index += 1
                task = asyncio.create_task(tts_service.synthesize(clean))
                tts_tasks.append((idx, task))
                return task

            # ── Phase 1: Get full answer via ainvoke, stream word-by-word ──
            try:
                final_answer = await run_agent(
                    query=req.message,
                    history_messages=history.messages,
                    chunks=chunks,
                    retriever=retriever,
                )
            except Exception as e:
                print(f"[ERROR] run_agent: {e}")
                final_answer = "Sorry, I encountered an internal error. Please try again."

            print(f"[TIMER] Agent invoke: {time.perf_counter() - t_start:.3f}s")

            if not final_answer or not final_answer.strip():
                final_answer = "I'm sorry, I wasn't able to generate a response. Please try rephrasing."

            # Stream word-by-word; detect sentence boundaries for TTS in parallel
            words = final_answer.split(" ")
            for i, word in enumerate(words):
                token = word + (" " if i < len(words) - 1 else "")
                yield _sse({"type": "text", "content": token})

                if voice:
                    for sentence in buf.feed(token):
                        if sentence_index >= TTS_MAX_SENTENCES and not tts_cap_warned:
                            tts_cap_warned = True
                            yield _sse({
                                "type": "error",
                                "stage": "tts_cap",
                                "message": "Voice synthesis limit reached for this response.",
                            })
                        _schedule_tts(sentence)

                await asyncio.sleep(0)

            print(f"[STREAM] Streamed {len(words)} words | voice={voice}")


            # ── Phase 2: Flush partial sentence at end of stream ───────────
            if voice:
                remainder = buf.flush()
                if remainder:
                    _schedule_tts(remainder)

            # ── Phase 3: Drain all pending TTS tasks and emit audio events ─
            if tts_tasks:
                # Wait for all in-flight TTS tasks
                results = await asyncio.gather(
                    *[task for _, task in tts_tasks], return_exceptions=True
                )
                for (idx, _), result in sorted(zip(tts_tasks, results), key=lambda x: x[0][0]):
                    if isinstance(result, Exception) or result is None:
                        yield _sse({
                            "type": "error",
                            "stage": "tts",
                            "message": f"Voice unavailable for sentence {idx + 1}.",
                        })
                    else:
                        yield _sse({
                            "type": "audio",
                            "index": idx,
                            "audio": tts_service.audio_to_b64(result),
                        })

            # ── Phase 4: Persist history + signal done ─────────────────────

            history.add_message(HumanMessage(content=req.message))
            history.add_message(AIMessage(content=final_answer))

            print(f"[STREAM] Total: {time.perf_counter() - t_start:.3f}s | "
                  f"voice={voice} | sentences={sentence_index}")

            yield _sse({"type": "done"})

        return StreamingResponse(
            response_generator(),
            media_type="text/event-stream; charset=utf-8",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    # ── /clear ─────────────────────────────────────────────────────────────────
    @app.post("/clear")
    async def clear_chat(req: ChatRequest):
        history = SQLChatMessageHistory(
            session_id=req.session_id, connection_string=MEMORY_DB,
        )
        history.clear()
        return {"status": "ok"}

    # ── / (frontend) ───────────────────────────────────────────────────────────
    @app.get("/")
    def root():
        return FileResponse(os.path.join(_PROJECT_ROOT, "index.html"))

    return app
