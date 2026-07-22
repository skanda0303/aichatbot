"""
index_dataset.py — Indexes the full BEIR SciFact corpus into the
multi-agent evaluator's own Chroma collection (chroma_db_eval_multi/).

Selects:
  50 present queries  — target docs ARE indexed (can be retrieved)
  25 absent queries   — target docs NOT indexed (tests hallucination resistance)

Same logic as evaluate_rag/index_dataset.py, but writes to the
evaluate_multi_rag collection to avoid touching the production indexes.

Run once before running run_eval_multi.py:
  python -m evaluate_multi_rag.index_dataset
"""

import os
import json
import time
import datasets
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from evaluate_multi_rag.config import CHUNK_SIZE, CHUNK_OVERLAP, BEIR_DATASET
from evaluate_multi_rag.ingestion import vectorstore, save_fingerprint

_corpus_dict = None


def get_corpus_dict() -> dict:
    global _corpus_dict
    if _corpus_dict is None:
        print(f"[INFO] Loading BEIR corpus for '{BEIR_DATASET}'...")
        ds = datasets.load_dataset(f"mteb/{BEIR_DATASET}", "corpus")
        split_name = list(ds.keys())[0]
        _corpus_dict = {row["_id"]: row for row in ds[split_name]}
        print(f"[INFO] Loaded {len(_corpus_dict)} documents in corpus.")
    return _corpus_dict


def load_paper_chunks(paper_id: str) -> list[Document]:
    corpus  = get_corpus_dict()
    doc_row = corpus.get(paper_id)
    if not doc_row:
        return []
    title     = doc_row.get("title", "")
    text      = doc_row.get("text", "")
    full_text = f"{title}\n{text}" if title else text
    full_text = " ".join(full_text.split())
    splitter  = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    doc = Document(page_content=full_text, metadata={"source": paper_id, "title": title})
    return splitter.split_documents([doc])


def main():
    print(f"[INFO] Loading BEIR queries & qrels for '{BEIR_DATASET}'...")
    queries_ds    = datasets.load_dataset(f"mteb/{BEIR_DATASET}", "queries")
    queries_split = list(queries_ds.keys())[0]
    queries       = {row["_id"]: row["text"] for row in queries_ds[queries_split]}

    qrels_ds   = datasets.load_dataset(f"mteb/{BEIR_DATASET}", "default")
    qrels_rows = []
    for split in qrels_ds.keys():
        qrels_rows.extend(qrels_ds[split])

    qrels: dict = {}
    for row in qrels_rows:
        q_id  = row["query-id"]
        c_id  = row["corpus-id"]
        score = row["score"]
        if score >= 1:
            if q_id not in qrels or score > qrels[q_id]["score"]:
                qrels[q_id] = {"doc_id": c_id, "score": score}

    corpus         = get_corpus_dict()
    all_paper_ids  = sorted(corpus.keys())
    valid_qids     = [q for q in queries if q in qrels]

    # ── Select 50 present + 25 absent queries ─────────────────────────────────
    present_qids: list[str] = []
    absent_qids:  list[str] = []
    present_docs: set[str]  = set()
    absent_docs:  set[str]  = set()

    for q_id in valid_qids:
        doc_id = qrels[q_id]["doc_id"]
        if len(present_qids) < 50:
            present_qids.append(q_id)
            present_docs.add(doc_id)

    # Absent queries: pick queries whose target doc is NOT in present_docs
    for q_id in valid_qids:
        if len(absent_qids) >= 25:
            break
        if q_id in present_qids:
            continue
        doc_id = qrels[q_id]["doc_id"]
        if doc_id not in present_docs:
            absent_qids.append(q_id)
            absent_docs.add(doc_id)

    # ── Papers to index: all corpus except absent target docs ─────────────────
    excluded_papers = absent_docs          # keep absent target docs OUT of index
    selected_papers = [p for p in all_paper_ids if p not in excluded_papers]
    total_selected  = len(selected_papers)

    print(f"[INFO] Total papers in BEIR corpus  : {len(all_paper_ids)}")
    print(f"[INFO] Papers excluded (absent docs) : {len(excluded_papers)}")
    print(f"[INFO] Papers to index               : {total_selected}")
    print(f"[INFO] Present queries               : {len(present_qids)}")
    print(f"[INFO] Absent  queries               : {len(absent_qids)}")

    # ── Clear existing collection ─────────────────────────────────────────────
    print("[INFO] Clearing existing vectorstore collection...")
    try:
        count = vectorstore._collection.count()
        if count > 0:
            print(f"[INFO] Deleting {count} existing chunks in batches...")
            while True:
                batch_ids = vectorstore._collection.get(limit=500)["ids"]
                if not batch_ids:
                    break
                vectorstore.delete(ids=batch_ids)
            print("[INFO] DB cleared.")
        else:
            print("[INFO] DB is already empty.")
    except Exception as e:
        print(f"[WARN] Error while clearing DB: {e}")

    # ── Index papers in batches ───────────────────────────────────────────────
    print("\nStarting indexing process...")
    t_start       = time.perf_counter()
    batch_chunks: list[Document] = []
    chunk_counter = 0
    BATCH_SIZE    = 100

    for idx, paper_id in enumerate(selected_papers, start=1):
        try:
            chunks = load_paper_chunks(paper_id)
            batch_chunks.extend(chunks)
            chunk_counter += len(chunks)
        except Exception as e:
            print(f"[WARN] Failed to load {paper_id}: {e}")

        while len(batch_chunks) >= BATCH_SIZE:
            upload = batch_chunks[:BATCH_SIZE]
            batch_chunks = batch_chunks[BATCH_SIZE:]
            t0 = time.perf_counter()
            vectorstore.add_documents(upload)
            print(
                f"[PROGRESS] Papers {idx}/{total_selected} | "
                f"Chunks indexed: {chunk_counter - len(batch_chunks)} | "
                f"Batch time: {time.perf_counter() - t0:.2f}s"
            )

    if batch_chunks:
        t0 = time.perf_counter()
        vectorstore.add_documents(batch_chunks)
        print(f"[PROGRESS] Final batch: {len(batch_chunks)} chunks. Time: {time.perf_counter() - t0:.2f}s")

    elapsed = time.perf_counter() - t_start
    print(
        f"\n[SUCCESS] Indexed {total_selected} papers ({chunk_counter} chunks) "
        f"in {elapsed:.2f}s ({elapsed/60:.2f} min)."
    )

    # ── Save config for eval script ───────────────────────────────────────────
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "indexed_config_multi.json")
    with open(config_path, "w") as f:
        json.dump({
            "eval_present_queries": present_qids,
            "eval_absent_queries":  absent_qids,
            "indexed_positives":    list(present_docs),
            "indexed_papers":       selected_papers,
            "excluded_papers":      list(excluded_papers),
        }, f)
    print(f"[INFO] Saved eval config → {config_path}")

    save_fingerprint(f"beir_{BEIR_DATASET}_multi_eval_full")


if __name__ == "__main__":
    main()
