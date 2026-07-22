"""
index_dataset.py — Indexes the entire BEIR dataset (e.g., SciFact) except for target documents
of 10 selected absent queries to test hallucination resistance.
"""

import os
import json
import time
import datasets
from langchain_core.documents import Document
from evaluate_rag.config import CHUNK_SIZE, CHUNK_OVERLAP, BEIR_DATASET
from evaluate_rag.ingestion import vectorstore, save_fingerprint

_corpus_dict = None

def get_corpus_dict():
    global _corpus_dict
    if _corpus_dict is None:
        print(f"[INFO] Loading BEIR corpus for '{BEIR_DATASET}'...")
        ds = datasets.load_dataset(f"mteb/{BEIR_DATASET}", "corpus")
        split_name = list(ds.keys())[0]  # usually 'corpus'
        corpus_ds = ds[split_name]
        _corpus_dict = {row["_id"]: row for row in corpus_ds}
        print(f"[INFO] Loaded {len(_corpus_dict)} documents in corpus.")
    return _corpus_dict

def load_paper_chunks(paper_id: str) -> list[Document]:
    corpus = get_corpus_dict()
    doc_row = corpus.get(paper_id)
    if not doc_row:
        return []

    title = doc_row.get("title", "")
    text = doc_row.get("text", "")
    full_text = f"{title}\n{text}" if title else text
    full_text = " ".join(full_text.split())

    from langchain_text_splitters import RecursiveCharacterTextSplitter
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )

    doc = Document(page_content=full_text, metadata={"source": paper_id, "title": title})
    return splitter.split_documents([doc])

def main():
    print(f"[INFO] Loading BEIR queries & qrels for '{BEIR_DATASET}'...")
    queries_ds = datasets.load_dataset(f"mteb/{BEIR_DATASET}", "queries")
    queries_split = list(queries_ds.keys())[0]
    queries = {row["_id"]: row["text"] for row in queries_ds[queries_split]}

    qrels_ds = datasets.load_dataset(f"mteb/{BEIR_DATASET}", "default")
    qrels_rows = []
    for split in qrels_ds.keys():
        qrels_rows.extend(qrels_ds[split])

    qrels = {}
    for row in qrels_rows:
        q_id = row["query-id"]
        c_id = row["corpus-id"]
        score = row["score"]
        if score >= 1:
            if q_id not in qrels or score > qrels[q_id]["score"]:
                qrels[q_id] = {"doc_id": c_id, "score": score}

    corpus = get_corpus_dict()
    all_paper_ids = sorted(list(corpus.keys()))

    # Select 65 present queries and 10 absent queries (with non-overlapping target docs)
    valid_qids = [q_id for q_id in queries.keys() if q_id in qrels]
    
    present_qids = []
    absent_qids = []
    present_docs = set()
    absent_docs = set()

    for q_id in valid_qids:
        doc_id = qrels[q_id]["doc_id"]
        if len(present_qids) < 50:
            present_qids.append(q_id)
            present_docs.add(doc_id)

    # All corpus docs will be indexed (no exclusions since absent_docs is empty)
    selected_papers = [pid for pid in all_paper_ids]
    total_selected = len(selected_papers)

    print(f"[INFO] Total papers in BEIR corpus: {len(all_paper_ids)}")
    print(f"[INFO] Total papers to index: {total_selected}")
    print(f"[INFO] Selected {len(present_qids)} queries as 'present' (can answer).")
    print(f"[INFO] Selected {len(absent_qids)} queries as 'absent' (cannot answer).")


    # Clear existing Chroma store safely in batches
    print("[INFO] Clearing existing vectorstore collection...")
    try:
        count = vectorstore._collection.count()
        if count > 0:
            print(f"[INFO] Deleting {count} existing chunks in batches...")
            while True:
                existing_batch = vectorstore._collection.get(limit=500)["ids"]
                if not existing_batch:
                    break
                vectorstore.delete(ids=existing_batch)
            print("[INFO] DB cleared.")
        else:
            print("[INFO] DB is already empty.")
    except Exception as e:
        print(f"[WARN] Error while clearing DB: {e}")

    # Index selected papers in batches of chunks
    print("\nStarting indexing process...")
    t_start = time.perf_counter()

    batch_chunks = []
    chunk_counter = 0
    BATCH_SIZE_CHUNKS = 100

    for idx, paper_id in enumerate(selected_papers, start=1):
        try:
            chunks = load_paper_chunks(paper_id)
            batch_chunks.extend(chunks)
            chunk_counter += len(chunks)
        except Exception as e:
            print(f"[WARN] Failed to load {paper_id}: {e}")

        # Batch upload
        while len(batch_chunks) >= BATCH_SIZE_CHUNKS:
            upload_batch = batch_chunks[:BATCH_SIZE_CHUNKS]
            batch_chunks = batch_chunks[BATCH_SIZE_CHUNKS:]
            
            batch_start = time.perf_counter()
            vectorstore.add_documents(upload_batch)
            print(f"[PROGRESS] Indexed batch of {len(upload_batch)} chunks. Total: {chunk_counter - len(batch_chunks)}/{chunk_counter} chunks (Papers indexed: {idx}/{total_selected}). Batch time: {time.perf_counter() - batch_start:.2f}s")

    # Upload remainder
    if batch_chunks:
        batch_start = time.perf_counter()
        vectorstore.add_documents(batch_chunks)
        print(f"[PROGRESS] Indexed final batch of {len(batch_chunks)} chunks. Total: {chunk_counter}. Batch time: {time.perf_counter() - batch_start:.2f}s")

    elapsed = time.perf_counter() - t_start
    print(f"\n[SUCCESS] Indexed {total_selected} papers ({chunk_counter} chunks) in {elapsed:.2f} seconds ({elapsed/60:.2f} minutes).")

    # Save a status file/fingerprint to denote indexing configuration
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "indexed_config.json")
    with open(config_path, "w") as f:
        json.dump({
            "eval_present_queries": present_qids,
            "eval_absent_queries": absent_qids,
            "indexed_positives": list(present_docs),
            "indexed_papers": selected_papers
        }, f)

    save_fingerprint(f"beir_{BEIR_DATASET}_full_indexing")

if __name__ == "__main__":
    main()
