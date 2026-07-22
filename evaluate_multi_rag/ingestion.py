"""
ingestion.py — Chroma vector store for the multi-agent evaluator.

Uses its own collection (bge_m3_eval_multi) in chroma_db_eval_multi/ so it
stays completely separate from both the production multi_agent index
(chroma_db_multi/) and the original evaluate_rag index (chroma_db/).
"""

import os
import json

from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

from evaluate_multi_rag.config import (
    CHROMA_DB, EMBEDDING_MODEL, CHROMA_COLLECTION,
)

embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)

vectorstore = Chroma(
    persist_directory=CHROMA_DB,
    embedding_function=embeddings,
    collection_name=CHROMA_COLLECTION,
)

_FINGERPRINT_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "eval_multi_fingerprint.json"
)


def save_fingerprint(tag: str) -> None:
    with open(_FINGERPRINT_PATH, "w") as f:
        json.dump({"tag": tag}, f)


def load_fingerprint() -> str:
    if os.path.exists(_FINGERPRINT_PATH):
        with open(_FINGERPRINT_PATH) as f:
            return json.load(f).get("tag", "")
    return ""
