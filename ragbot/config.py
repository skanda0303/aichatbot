"""
config.py — Central configuration for the RAG Chatbot.

All tunable constants (model names, retrieval weights, chunk sizes,
server settings) live here. API keys are loaded from a .env file
in the project root via python-dotenv.
"""

import os
from dotenv import load_dotenv

# Find the absolute path to the project root and load the .env file
_current_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.abspath(os.path.join(_current_dir, ".."))
_env_path = os.path.join(_project_root, ".env")
load_dotenv(dotenv_path=_env_path)

# Paths
DOCS_DIR  = "docs"
HASH_FILE = "docs_hash.json"
CHROMA_DB = "chroma_db"
MEMORY_DB = "sqlite:///memory3.db"

# Models
EMBEDDING_MODEL   = "bge-m3"
LLM_MODEL         = "gemini-3.5-flash-lite"
CHROMA_COLLECTION = "bge_m3"

# API keys (loaded from .env)
GOOGLE_API_KEY      = os.getenv("GOOGLE_API_KEY", "")
TAVILY_API_KEY      = os.getenv("TAVILY_API_KEY", "")
FISH_AUDIO_API_KEY  = os.getenv("FISH_AUDIO_API_KEY", "")
FISH_AUDIO_VOICE_ID = os.getenv("FISH_AUDIO_VOICE_ID", "")  # reference_id for Fish Audio 2.1 Pro

# Retrieval tuning
RETRIEVER_K          = 12
BM25_WEIGHT          = 0.3
VECTOR_WEIGHT        = 0.7
REDUNDANCY_THRESHOLD = 0.85

# Chunking
CHUNK_SIZE    = 1000
CHUNK_OVERLAP = 150

# Agent behaviour
MAX_AGENT_ITERATIONS = 6
MAX_HISTORY_MESSAGES = 6
LLM_TEMPERATURE      = 0.2

# Web search
MAX_WEB_RESULTS = 5
MAX_FETCH_CHARS = 12000

# TTS settings
TTS_FORMAT        = os.getenv("TTS_FORMAT", "mp3")
TTS_MAX_SENTENCES = int(os.getenv("TTS_MAX_SENTENCES", "20"))
TTS_BACKEND       = os.getenv("TTS_BACKEND", "s2.1-pro-free")  # s2.1-pro-free = S2.1 Pro Free
TTS_AVAILABLE     = False  # disabled for now

if not TTS_AVAILABLE:
    import logging as _logging
    _logging.warning(
        "[TTS] FISH_AUDIO_API_KEY is not set — voice synthesis is disabled. "
        "voice_enabled requests will be silently downgraded to text-only mode."
    )

# Server
SERVER_HOST = "0.0.0.0"
SERVER_PORT = 8003
