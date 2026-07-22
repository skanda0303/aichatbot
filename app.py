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
    selected_doc: str | None = None
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
def list_documents():
    root = Path(DOCS_DIR)
    root.mkdir(parents=True, exist_ok=True)
    allowed = {".pdf", ".csv", ".txt", ".md", ".json"}
    docs = []
    for path in sorted(root.iterdir(), key=lambda item: item.name.lower()):
        if path.is_file() and path.suffix.lower() in allowed:
            stat = path.stat()
            ext = path.suffix.lower().strip(".")
            docs.append({
                "name": path.name,
                "size": stat.st_size,
                "type": ext.upper(),
                "url": f"/documents/{path.name}",
                "download_url": f"/documents/{path.name}?download=1"
            })
    return {"documents": docs, "indexed_chunks": len(chunks)}

@fastapi_app.delete("/api/documents/{name:path}")
def delete_document(name: str):
    root = Path(DOCS_DIR)
    target = root / name
    if not target.exists() or not target.is_file():
        raise HTTPException(status_code=404, detail="Document not found")
    try:
        target.unlink()
        global chunks, retriever
        print(f"[HF SPACE] Deleted '{name}'. Re-indexing remaining documents...")
        chunks = load_and_index_documents()
        retriever = build_retriever(chunks)
        return {"status": "ok", "message": f"Deleted '{name}' and re-indexed knowledge base."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete document: {e}")

@fastapi_app.get("/documents/{name:path}")
def serve_document(name: str, download: bool = False):
    root = Path(DOCS_DIR)
    target = root / name
    if not target.exists() or not target.is_file():
        raise HTTPException(status_code=404, detail="Document not found")
    headers = {
        "Content-Disposition": f"{'attachment' if download else 'inline'}; filename*=UTF-8''{target.name}",
        "X-Content-Type-Options": "nosniff",
    }
    return FileResponse(target, headers=headers)

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
    try:
        chunks = load_and_index_documents()
        retriever = build_retriever(chunks)
        indexed = len(chunks)
    except Exception as e:
        print(f"[HF SPACE] Re-index error (file still saved): {e}")
        indexed = 0

    return JSONResponse({
        "status": "ok",
        "filename": file.filename,
        "indexed_chunks": indexed,
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
                selected_doc=req.selected_doc,
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
