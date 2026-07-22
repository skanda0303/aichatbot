"""
ingestion.py — Document loading, fingerprinting, and Chroma vector store indexing.

Identical logic to ragbot/ingestion.py.
Reads PDFs from docs/, computes an MD5 fingerprint to detect changes,
and either reuses the existing Chroma index (fast path) or re-indexes
all documents (merges pages per file → splits into chunks → stores in Chroma).
"""

import os
import json
import hashlib
from collections import defaultdict

from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from evaluate_rag.config import (
    DOCS_DIR, HASH_FILE, CHROMA_DB,
    EMBEDDING_MODEL, CHROMA_COLLECTION,
    CHUNK_SIZE, CHUNK_OVERLAP,
)

# Embedding model & vector store  — same Chroma DB as ragbot
embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
vectorstore = Chroma(
    persist_directory=CHROMA_DB,
    embedding_function=embeddings,
    collection_name=CHROMA_COLLECTION,
)


def get_docs_fingerprint() -> str:
    """MD5 hash over all files in DOCS_DIR — changes when files are added/removed/modified."""
    hasher = hashlib.md5()
    for fname in sorted(os.listdir(DOCS_DIR)):
        fpath = os.path.join(DOCS_DIR, fname)
        if os.path.isfile(fpath):
            with open(fpath, "rb") as f:
                hasher.update(fname.encode())
                hasher.update(f.read())
    return hasher.hexdigest()


def load_stored_fingerprint() -> str:
    if os.path.exists(HASH_FILE):
        with open(HASH_FILE) as f:
            return json.load(f).get("hash", "")
    return ""


def save_fingerprint(h: str) -> None:
    with open(HASH_FILE, "w") as f:
        json.dump({"hash": h}, f)


def load_and_index_documents() -> list[Document]:
    """Load PDFs, detect changes via fingerprint, and index into Chroma. Returns chunk list."""
    os.makedirs(DOCS_DIR, exist_ok=True)

    if not os.listdir(DOCS_DIR):
        print("[INFO] docs/ is empty — no documents ingested.")
        return []

    current_fp   = get_docs_fingerprint()
    stored_fp    = load_stored_fingerprint()
    existing_count = vectorstore._collection.count()

    # Unchanged — reuse existing index
    if current_fp == stored_fp and existing_count > 0:
        print(f"[OK] Documents unchanged — reusing existing index ({existing_count} chunks).")
        chunks: list[Document] = []
        limit = 500
        offset = 0
        while True:
            existing_data = vectorstore._collection.get(limit=limit, offset=offset)
            ids = existing_data.get("ids", [])
            if not ids:
                break
            for doc_text, metadata in zip(existing_data["documents"], existing_data["metadatas"]):
                chunks.append(Document(page_content=doc_text, metadata=metadata))
            if len(ids) < limit:
                break
            offset += limit
        return chunks

    # Changed — re-index
    print("[INFO] Document changes detected. Re-indexing...")
    raw_docs = PyPDFDirectoryLoader(DOCS_DIR).load()

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

    chunks = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP,
    ).split_documents(merged_docs)

    if existing_count > 0:
        while True:
            existing_batch = vectorstore._collection.get(limit=500)["ids"]
            if not existing_batch:
                break
            vectorstore.delete(ids=existing_batch)

    # Batch document addition to prevent Ollama runner crashes on large batch requests
    batch_size = 32
    for i in range(0, len(chunks), batch_size):
        vectorstore.add_documents(chunks[i : i + batch_size])
    save_fingerprint(current_fp)
    print(f"[OK] Ingested {len(chunks)} chunks from {len(merged_docs)} file(s).")
    return chunks
