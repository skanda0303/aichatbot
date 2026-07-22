"""
retrieval/ingestion.py — Document loading, fingerprinting, and Chroma vector-store indexing.

Identical logic to ragbot/ingestion.py — same embedding model (bge-m3), same Chroma
collection name, same chunk size / overlap.  Only the path resolution differs (uses
multi_agent/config.py for DOCS_DIR, HASH_FILE, CHROMA_DB).
"""

import os
import json
import hashlib
from collections import defaultdict

from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from multi_agent.config import (
    DOCS_DIR, HASH_FILE, CHROMA_DB,
    EMBEDDING_MODEL, CHROMA_COLLECTION,
    CHUNK_SIZE, CHUNK_OVERLAP,
)
from multi_agent.retrieval.table_serialization import (
    load_pdf_prose_and_tables,
)

def _init_embeddings():
    """Initialize embeddings with automatic fallback to HuggingFace BAAI/bge-m3 if Ollama is unreachable."""
    if os.getenv("USE_HUGGINGFACE_EMBEDDINGS", "").lower() in ("1", "true"):
        try:
            from langchain_huggingface import HuggingFaceEmbeddings
            print("[INFO] Loading HuggingFace BAAI/bge-m3 embeddings...")
            return HuggingFaceEmbeddings(model_name="BAAI/bge-m3", model_kwargs={"device": "cpu"})
        except Exception as e:
            print(f"[WARN] HuggingFaceEmbeddings load failed: {e}")

    try:
        emb = OllamaEmbeddings(model=EMBEDDING_MODEL)
        return emb
    except Exception as e:
        print(f"[WARN] Ollama unavailable: {e}. Falling back to HuggingFace BAAI/bge-m3 embeddings...")
        try:
            from langchain_huggingface import HuggingFaceEmbeddings
            return HuggingFaceEmbeddings(model_name="BAAI/bge-m3", model_kwargs={"device": "cpu"})
        except Exception as hf_e:
            print(f"[ERROR] HuggingFace fallback failed: {hf_e}")
            return OllamaEmbeddings(model=EMBEDDING_MODEL)


# Embedding model & vector store
embeddings = _init_embeddings()
vectorstore = Chroma(
    persist_directory=CHROMA_DB,
    embedding_function=embeddings,
    collection_name=CHROMA_COLLECTION,
)


# Called in: multi_agent/retrieval/ingestion.py (load_and_index_documents)
def get_docs_fingerprint() -> str:
    """MD5 hash over all files in DOCS_DIR and table_serialization.py logic — changes when files or parser rules change.

    Includes both PDFs/CSVs and table parsing logic so code updates trigger a re-index.
    """
    hasher = hashlib.md5()
    for fname in sorted(os.listdir(DOCS_DIR)):
        fpath = os.path.join(DOCS_DIR, fname)
        if os.path.isfile(fpath):
            with open(fpath, "rb") as f:
                hasher.update(fname.encode())
                hasher.update(f.read())
    
    # Also hash table_serialization.py so parser changes invalidate old indexes
    ts_path = os.path.join(os.path.dirname(__file__), "table_serialization.py")
    if os.path.exists(ts_path):
        with open(ts_path, "rb") as f:
            hasher.update(f.read())

    return hasher.hexdigest()


# Called in: multi_agent/retrieval/ingestion.py (load_and_index_documents)
def load_stored_fingerprint() -> str:
    if os.path.exists(HASH_FILE):
        with open(HASH_FILE) as f:
            return json.load(f).get("hash", "")
    return ""


# Called in: multi_agent/retrieval/ingestion.py (load_and_index_documents)
def save_fingerprint(h: str) -> None:
    with open(HASH_FILE, "w") as f:
        json.dump({"hash": h}, f)


# Called in: multi_agent/main.py
def load_and_index_documents() -> list[Document]:
    """Load PDFs, detect changes via fingerprint, and index into Chroma. Returns chunk list."""
    os.makedirs(DOCS_DIR, exist_ok=True)

    if not os.listdir(DOCS_DIR):
        print("[INFO] docs/ is empty — no documents ingested.")
        return []

    current_fp   = get_docs_fingerprint()
    stored_fp    = load_stored_fingerprint()
    existing_ids = vectorstore.get()["ids"]

    # Unchanged — reuse existing index
    if current_fp == stored_fp and existing_ids:
        print(f"[OK] Documents unchanged — reusing existing index ({len(existing_ids)} chunks).")
        existing_data = vectorstore.get()
        chunks: list[Document] = []
        if existing_data and "documents" in existing_data:
            for doc_text, metadata in zip(existing_data["documents"], existing_data["metadatas"]):
                chunks.append(Document(page_content=doc_text, metadata=metadata))
        return chunks

    # Changed — re-index
    print("[INFO] Document changes detected. Re-indexing...")
    raw_docs, pdf_table_docs = load_pdf_prose_and_tables(DOCS_DIR)

    grouped: dict = defaultdict(list)
    for d in raw_docs:
        grouped[d.metadata.get("source", "unknown")].append(
            (d.metadata.get("page", 0), d.page_content)
        )

    merged_docs: list[Document] = []
    for src, pages in sorted(grouped.items()):
        pages.sort(key=lambda x: x[0])
        full_text = " ".join(" ".join(c for _, c in pages).split())
        merged_docs.append(Document(page_content=full_text, metadata={"source": src, "page": "0"}))

    # 1) PDF text → chunked with sliding splitter (overlap > 0)
    chunks = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP,
    ).split_documents(merged_docs)

    # 2) Tables embedded inside PDFs → serialized row-by-row (pdfplumber)
    chunks.extend(pdf_table_docs)

    if existing_ids:
        vectorstore.delete(ids=existing_ids)

    # Batch document addition to prevent Ollama runner crashes on large batch requests
    batch_size = 32
    for i in range(0, len(chunks), batch_size):
        vectorstore.add_documents(chunks[i : i + batch_size])
    save_fingerprint(current_fp)
    print(
        f"[OK] Ingested {len(chunks)} chunks "
        f"({len(merged_docs)} PDF file(s) -> {len(chunks) - len(pdf_table_docs)} text chunks, "
        f"{len(pdf_table_docs)} PDF-table-row documents)."
    )
    return chunks
