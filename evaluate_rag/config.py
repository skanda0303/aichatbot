"""
config.py — Central configuration for the RAG Evaluator.

All constants are kept identical to ragbot/config.py.
API keys are loaded from the project-root .env file.
"""

import os
from dotenv import load_dotenv

# Locate project root (evaluate_rag/ is one level below the project root)
_current_dir  = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.abspath(os.path.join(_current_dir, ".."))
_env_path     = os.path.join(_project_root, ".env")
load_dotenv(dotenv_path=_env_path)

# Paths  (relative to project root, same as ragbot)
DOCS_DIR  = os.path.join(_project_root, "docs")
HASH_FILE = os.path.join(_project_root, "docs_hash.json")
CHROMA_DB = os.path.join(_project_root, "chroma_db")

# Shared DB file (same Chroma store as the main bot)
MEMORY_DB = "sqlite:///memory3.db"

# Models  — identical to ragbot/config.py
EMBEDDING_MODEL   = "bge-m3"
LLM_MODEL         = "gemini-3.5-flash-lite"
CHROMA_COLLECTION = "bge_m3"

# API keys
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")

# Retrieval tuning  — identical to ragbot/config.py
RETRIEVER_K          = 12
BM25_WEIGHT          = 0.3
VECTOR_WEIGHT        = 0.7
REDUNDANCY_THRESHOLD = 0.85

# Chunking  — identical to ragbot/config.py
CHUNK_SIZE    = 1000
CHUNK_OVERLAP = 150

# Agent behaviour  — identical to ragbot/config.py
MAX_HISTORY_MESSAGES = 6
LLM_TEMPERATURE      = 0.2

# Evaluation-specific: the two chunk-count values to compare per query
EVAL_K_VALUES = [3, 5]

# BEIR Dataset Configuration
BEIR_DATASET = "scifact"

