"""
main.py — Entry point for the RAG Chatbot server.

Startup sequence:
  1. Load & index documents from docs/ into Chroma vector store
  2. Build the hybrid retriever (BM25 + vector)
  3. Wire the retriever into the agent tools
  4. Create the FastAPI app and start Uvicorn

Run with: python -m ragbot.main
"""

import os
from dotenv import load_dotenv

# Load environment variables relative to package root before importing anything else
_current_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.abspath(os.path.join(_current_dir, ".."))
load_dotenv(dotenv_path=os.path.join(_project_root, ".env"))

from ragbot.config import SERVER_HOST, SERVER_PORT
from ragbot.ingestion import load_and_index_documents
from ragbot.retriever import build_retriever
from ragbot.agent import initialise_agent
from ragbot.api import create_app

# Startup sequence
chunks    = load_and_index_documents()
retriever = build_retriever(chunks)
initialise_agent(retriever, chunks)
app       = create_app(chunks, retriever)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("ragbot.main:app", host=SERVER_HOST, port=SERVER_PORT, reload=True)
