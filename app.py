import os
import json
import asyncio
from pathlib import Path
from fastapi import FastAPI, Request, UploadFile, File, HTTPException
from fastapi.responses import FileResponse, StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import gradio as gr
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()
os.environ["USE_HUGGINGFACE_EMBEDDINGS"] = "1"

# Satisfy Hugging Face ZeroGPU startup validator
try:
    import spaces
    @spaces.GPU
    def _zerogpu_startup_check():
        pass
    _zerogpu_startup_check()
except Exception:
    pass

# Import multi_agent components
from multi_agent.retrieval.ingestion import load_and_index_documents
from multi_agent.retrieval.retriever import build_retriever
from multi_agent.agents import supervisor_agent
from multi_agent.config import DOCS_DIR

print("[HF SPACE] Preparing document index...")
chunks = load_and_index_documents()
print("[HF SPACE] Preparing hybrid retriever...")
retriever = build_retriever(chunks)
print("[HF SPACE] Multi-agent pipeline ready!")

# FastAPI app for serving custom index.html frontend
fastapi_app = FastAPI(title="Atlas Multi-Agent Workspace")

fastapi_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    session_id: str
    message: str
    voice_enabled: bool = False

@fastapi_app.get("/")
async def get_index():
    return FileResponse("index.html")

@fastapi_app.get("/production_tables.css")
async def get_css():
    return FileResponse("production_tables.css")

@fastapi_app.get("/audioPlayer.js")
async def get_js():
    return FileResponse("audioPlayer.js")

@fastapi_app.get("/api/documents")
async def get_documents():
    docs = []
    if os.path.exists(DOCS_DIR):
        for fname in sorted(os.listdir(DOCS_DIR)):
            fpath = os.path.join(DOCS_DIR, fname)
            if os.path.isfile(fpath):
                ext = os.path.splitext(fname)[1].upper().lstrip(".")
                docs.append({
                    "name": fname,
                    "type": ext or "FILE",
                    "size": os.path.getsize(fpath),
                    "url": f"/docs_multi/{fname}",
                    "download_url": f"/docs_multi/{fname}?download=1"
                })
    return JSONResponse({"documents": docs, "indexed_chunks": len(chunks)})

@fastapi_app.get("/docs_multi/{name:path}")
async def serve_doc_file(name: str, download: bool = False):
    root = Path(DOCS_DIR).resolve()
    fpath = (root / name).resolve()
    if root not in fpath.parents and root != fpath.parent:
        raise HTTPException(status_code=404, detail="File not found")
    if not fpath.is_file():
        raise HTTPException(status_code=404, detail="File not found")
    headers = {
        "Content-Disposition": f"{'attachment' if download else 'inline'}; filename*=UTF-8''{fpath.name}",
    }
    return FileResponse(fpath, headers=headers)

@fastapi_app.post("/api/upload")
async def upload_document(file: UploadFile = File(...)):
    allowed = {".pdf", ".csv", ".txt", ".md", ".json"}
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in allowed:
        raise HTTPException(status_code=400, detail=f"Unsupported file type. Allowed: {', '.join(allowed)}")
    
    os.makedirs(DOCS_DIR, exist_ok=True)
    dest_path = os.path.join(DOCS_DIR, file.filename)
    contents = await file.read()
    with open(dest_path, "wb") as f:
        f.write(contents)
    
    global chunks, retriever
    print(f"[HF SPACE] File '{file.filename}' uploaded to docs_multi/. Re-indexing...")
    chunks = load_and_index_documents()
    retriever = build_retriever(chunks)

    return JSONResponse({
        "status": "ok",
        "filename": file.filename,
        "indexed_chunks": len(chunks),
        "message": f"Successfully uploaded '{file.filename}' to docs_multi and re-indexed knowledge base."
    })

@fastapi_app.post("/chat")
async def chat_endpoint(req: ChatRequest, request: Request):
    user_gemini_key = request.headers.get("X-Gemini-Key") or None
    user_tavily_key = request.headers.get("X-Tavily-Key") or None

    async def generate_response():
        try:
            async for token in supervisor_agent.run_streaming(
                query=req.message,
                history_messages=[],
                retriever=retriever,
                chunks=chunks,
                user_gemini_key=user_gemini_key,
                user_tavily_key=user_tavily_key,
            ):
                event = {"type": "text", "content": token}
                yield json.dumps(event) + "\n"
        except Exception as e:
            event = {"type": "error", "stage": "supervisor", "message": str(e)}
            yield json.dumps(event) + "\n"

    return StreamingResponse(generate_response(), media_type="text/event-stream")

@fastapi_app.post("/clear")
async def clear_endpoint():
    return JSONResponse({"status": "ok"})

# Gradio UI fallback (available at /gradio)
async def gradio_predict(message, history):
    history_messages = []
    for msg in history:
        role = msg.get("role")
        content = msg.get("content", "")
        if role == "user":
            history_messages.append(HumanMessage(content=content))
        elif role == "assistant":
            history_messages.append(AIMessage(content=content))

    full_response = ""
    async for token in supervisor_agent.run_streaming(
        query=message,
        history_messages=history_messages,
        retriever=retriever,
        chunks=chunks,
    ):
        full_response += token
        yield full_response

gradio_demo = gr.ChatInterface(
    fn=gradio_predict,
    title="🤖 Multi-Agent RAG Chatbot",
    description="Fact-grounded multi-agent system with Context Evaluation & Fact-checking Critic.",
    type="messages",
)

# Mount Gradio under /gradio path on FastAPI app
app = gr.mount_gradio_app(fastapi_app, gradio_demo, path="/gradio")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=7860)
