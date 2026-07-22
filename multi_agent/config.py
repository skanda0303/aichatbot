"""
config.py — Central configuration for the Multi-Agent RAG Chatbot.

All tunable constants (model names, retrieval weights, chunk sizes,
server settings) are preserved exactly from the single-agent ragbot.
API keys are loaded from a .env file in the project root via python-dotenv.
"""

import os
from dotenv import load_dotenv

# Find the absolute path to the project root and load the .env file
_current_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.abspath(os.path.join(_current_dir, ".."))
_env_path = os.path.join(_project_root, ".env")
load_dotenv(dotenv_path=_env_path)

# Paths — multi_agent uses its own separate folder, vector store, and hash file
# so documents can be indexed independently from ragbot/
DOCS_DIR  = os.path.join(_project_root, "docs_multi")          # <-- dedicated docs folder
HASH_FILE = os.path.join(_project_root, "docs_multi_hash.json") # <-- dedicated fingerprint
CHROMA_DB = os.path.join(_project_root, "chroma_db_multi")     # <-- dedicated vector store
MEMORY_DB = f"sqlite:///{os.path.join(_project_root, 'memory_multi.db')}"

# Models — identical to ragbot/config.py
EMBEDDING_MODEL   = "bge-m3"
LLM_MODEL         = "gemini-3.1-flash-lite"
CHROMA_COLLECTION = "bge_m3_multi"                              # <-- separate collection name

# API keys (loaded from .env)
GOOGLE_API_KEY      = os.getenv("GOOGLE_API_KEY", "")
TAVILY_API_KEY      = os.getenv("TAVILY_API_KEY", "")
COMPOSIO_API_KEY    = os.getenv("COMPOSIO_API_KEY", "")
# Entity ID used to scope Composio tool executions to this user's connected accounts
COMPOSIO_USER_ID    = os.getenv("COMPOSIO_USER_ID", "")

# Retrieval tuning — identical to ragbot/config.py
RETRIEVER_K          = 12
BM25_WEIGHT          = 0.3
VECTOR_WEIGHT        = 0.7
REDUNDANCY_THRESHOLD = 0.85

# Chunking — identical to ragbot/config.py
CHUNK_SIZE    = 1000
CHUNK_OVERLAP = 150

# Agent behaviour — identical to ragbot/config.py
MAX_HISTORY_MESSAGES = 6
LLM_TEMPERATURE      = 0.2

# Web search — identical to ragbot/config.py
MAX_WEB_RESULTS = 5
MAX_FETCH_CHARS = 12000

# Server (reads PORT from environment if available, e.g. 7860 on Hugging Face or 8004 locally)
SERVER_HOST = "0.0.0.0"
SERVER_PORT = int(os.getenv("PORT", "8004"))
