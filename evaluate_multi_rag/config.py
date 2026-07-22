"""
config.py — Configuration for the Multi-Agent RAG Evaluator.

Uses the same models, retrieval weights, and chunking as multi_agent/config.py.
Points to its own dedicated Chroma collection (bge_m3_eval_multi) so the BEIR
SciFact corpus does not overwrite the multi_agent production index.
"""

import os
from dotenv import load_dotenv

_current_dir  = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.abspath(os.path.join(_current_dir, ".."))
_env_path     = os.path.join(_project_root, ".env")
load_dotenv(dotenv_path=_env_path)

# ── Paths (own Chroma DB — does NOT touch multi_agent's chroma_db_multi) ──────
CHROMA_DB = os.path.join(_project_root, "chroma_db_eval_multi")

# ── Models (identical to multi_agent/config.py) ───────────────────────────────
EMBEDDING_MODEL   = "bge-m3"
LLM_MODEL         = "gemini-3.1-flash-lite"
CHROMA_COLLECTION = "bge_m3_eval_multi"

# ── API keys ──────────────────────────────────────────────────────────────────
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")

# ── Retrieval (identical to multi_agent/config.py) ───────────────────────────
RETRIEVER_K          = 12
BM25_WEIGHT          = 0.3
VECTOR_WEIGHT        = 0.7
REDUNDANCY_THRESHOLD = 0.85

# ── Chunking (identical to multi_agent/config.py) ────────────────────────────
CHUNK_SIZE    = 1000
CHUNK_OVERLAP = 150

# ── LLM behaviour ────────────────────────────────────────────────────────────
LLM_TEMPERATURE = 0.2

# ── Evaluation settings ───────────────────────────────────────────────────────
BEIR_DATASET  = "scifact"
EVAL_K_VALUES = [3, 5]          # number of final chunks to test per query
EVAL_SIZE     = 75              # same as evaluate_rag: 50 present + 25 absent
