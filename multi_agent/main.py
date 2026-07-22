"""
main.py — Entry point for the Multi-Agent RAG Chatbot server.
"""

import asyncio
import os
from contextlib import asynccontextmanager
from dotenv import load_dotenv

_current_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.abspath(os.path.join(_current_dir, ".."))
load_dotenv(dotenv_path=os.path.join(_project_root, ".env"))

from multi_agent.config import SERVER_HOST, SERVER_PORT
from multi_agent.api import create_app_with_lifespan


async def initialize(app):
    from multi_agent.retrieval.ingestion import load_and_index_documents
    from multi_agent.retrieval.retriever import build_retriever

    print("[STARTUP] Preparing document index...")
    chunks = await asyncio.to_thread(load_and_index_documents)
    print("[STARTUP] Preparing hybrid retriever...")
    retriever = await asyncio.to_thread(build_retriever, chunks)
    app.state.chunks = chunks
    app.state.retriever = retriever
    app.state.ready = True
    print("[STARTUP] Document index ready")


@asynccontextmanager
async def lifespan(app):
    app.state.chunks = []
    app.state.retriever = None
    app.state.ready = False
    task = asyncio.create_task(initialize(app))
    port = int(os.environ.get("PORT", SERVER_PORT))
    is_cloud = bool(os.environ.get("PORT")) or bool(os.environ.get("SPACE_ID"))
    print(f"[STARTUP] Listening - http://0.0.0.0:{port}")
    yield
    task.cancel()
    print("[SHUTDOWN] Cleaning up...")


app = create_app_with_lifespan(lifespan)

if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", SERVER_PORT))
    is_cloud = bool(os.environ.get("PORT")) or bool(os.environ.get("SPACE_ID"))
    
    # Reload enabled for local dev, disabled on cloud hosts (Hugging Face / GCP)
    uvicorn.run(
        "multi_agent.main:app",
        host="0.0.0.0",
        port=port,
        reload=not is_cloud,
    )
