"""
api.py — FastAPI web server for the Multi-Agent RAG Chatbot.
"""

import asyncio
import json
import os
import time
from pathlib import Path
from typing import Callable

from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response, StreamingResponse
from pydantic import BaseModel

from multi_agent.agents import supervisor_agent
from multi_agent.config import DOCS_DIR, SERVER_PORT
from multi_agent.memory.history import append_exchange, clear_history, get_recent_messages

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_ALLOWED_DOCUMENT_TYPES = {".pdf", ".csv", ".txt", ".md", ".json"}


# Called in: multi_agent/api.py (response_generator)
def _sse(obj: dict) -> str:
    return json.dumps(obj, ensure_ascii=False) + "\n"


# Called in: multi_agent/api.py (serve_document)
def _document_path(name: str) -> Path:
    root = Path(DOCS_DIR).resolve()
    path = (root / name).resolve()
    if root not in path.parents or path.suffix.lower() not in _ALLOWED_DOCUMENT_TYPES:
        raise HTTPException(status_code=404, detail="Document not found")
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Document not found")
    return path


# Called in: multi_agent/api.py (list_documents)
def _document_summary(path: Path) -> dict:
    stat = path.stat()
    return {
        "name": path.name,
        "type": path.suffix.lower().lstrip(".").upper(),
        "size": stat.st_size,
        "modified": stat.st_mtime,
        "url": f"/documents/{path.name}",
        "download_url": f"/documents/{path.name}?download=1",
    }


# Called in: multi_agent/main.py
def create_app(chunks: list, retriever) -> FastAPI:
    return _build_app(chunks=chunks, retriever=retriever, lifespan=None)


# Called in: multi_agent/main.py (lifespan variant)
def create_app_with_lifespan(lifespan: Callable) -> FastAPI:
    """Create app that reads chunks/retriever from app.state (set by lifespan)."""
    return _build_app(chunks=None, retriever=None, lifespan=lifespan)


def _build_app(chunks, retriever, lifespan) -> FastAPI:
    app = FastAPI(title=f"Multi-Agent RAG Chatbot — Port {SERVER_PORT}", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Route: GET /production_tables.css (FastAPI handler)
    @app.get("/production_tables.css")
    def serve_css():
        return FileResponse(os.path.join(_PROJECT_ROOT, "production_tables.css"))

    # Route: GET /audioPlayer.js (FastAPI handler)
    @app.get("/audioPlayer.js")
    def serve_audio_player():
        return FileResponse(os.path.join(_PROJECT_ROOT, "audioPlayer.js"))

    # Route: GET /favicon.ico (FastAPI handler)
    @app.get("/favicon.ico")
    def serve_favicon():
        path = os.path.join(_PROJECT_ROOT, "favicon.ico")
        if os.path.exists(path):
            return FileResponse(path)
        return Response(status_code=204)

    class ChatRequest(BaseModel):
        session_id: str
        message: str

    # Route: GET /api/documents (FastAPI handler)
    @app.get("/api/documents")
    def list_documents():
        _chunks = chunks if chunks is not None else getattr(app.state, "chunks", [])
        root = Path(DOCS_DIR)
        root.mkdir(parents=True, exist_ok=True)
        documents = [
            _document_summary(path)
            for path in sorted(root.iterdir(), key=lambda item: item.name.lower())
            if path.is_file() and path.suffix.lower() in _ALLOWED_DOCUMENT_TYPES
        ]
        return {"documents": documents, "indexed_chunks": len(_chunks)}

    # Route: GET /documents/{name:path} (FastAPI handler)
    @app.get("/documents/{name:path}")
    def serve_document(name: str, download: bool = False):
        path = _document_path(name)
        headers = {
            "Content-Disposition": f"{'attachment' if download else 'inline'}; filename*=UTF-8''{path.name}",
            "X-Content-Type-Options": "nosniff",
        }
        return FileResponse(path, headers=headers)

    @app.post("/chat")
    async def chat(
        req: ChatRequest,
        x_gemini_key: str | None = Header(default=None, alias="X-Gemini-Key"),
        x_tavily_key: str | None = Header(default=None, alias="X-Tavily-Key"),
    ):
        # Called in: multi_agent/api.py (chat - within StreamingResponse)
        async def response_generator():
            _chunks    = chunks    if chunks    is not None else getattr(app.state, "chunks",    [])
            _retriever = retriever if retriever is not None else getattr(app.state, "retriever", None)
            t_start = time.perf_counter()
            history_messages = get_recent_messages(req.session_id)
            accumulated_answer: list[str] = []
            try:
                async for token in supervisor_agent.run_streaming(
                    query=req.message,
                    history_messages=history_messages,
                    retriever=_retriever,
                    chunks=_chunks,
                    user_gemini_key=x_gemini_key,
                    user_tavily_key=x_tavily_key,
                ):
                    accumulated_answer.append(token)
                    yield _sse({"type": "text", "content": token})
                    await asyncio.sleep(0)
            except Exception as e:
                print(f"[API] Supervisor error: {e}")
                yield _sse({"type": "error", "stage": "supervisor", "message": str(e)})
            final_answer = "".join(accumulated_answer)
            if final_answer.strip():
                append_exchange(req.session_id, req.message, final_answer)
            elapsed = time.perf_counter() - t_start
            print(f"[API] Total time: {elapsed:.3f}s | tokens: {len(accumulated_answer)}")
            yield _sse({"type": "done"})

        return StreamingResponse(
            response_generator(),
            media_type="text/event-stream; charset=utf-8",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    # Route: POST /clear (FastAPI handler)
    @app.post("/clear")
    async def clear_chat(req: ChatRequest):
        clear_history(req.session_id)
        return {"status": "ok"}

    # Route: GET / (FastAPI handler)
    @app.get("/")
    def root():
        return FileResponse(os.path.join(_PROJECT_ROOT, "index.html"))

    return app
