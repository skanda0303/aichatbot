"""
open_ragbench_eval.py — Evaluator script for BEIR Benchmark (replacing Open RAG Benchmark).

Loads the BEIR dataset, indexes a subset of documents into Chroma,
and runs a retrieval evaluation for both k=3 and k=5.
"""

import os
import json
import time
import datasets
from langchain_core.documents import Document
from evaluate_rag.config import CHUNK_SIZE, CHUNK_OVERLAP, BEIR_DATASET
from evaluate_rag.ingestion import vectorstore, save_fingerprint
from evaluate_rag.retriever import build_retriever
from evaluate_rag.rag_pipeline import retrieve_chunks

_cached_corpus = None

def get_corpus_dict():
    global _cached_corpus
    if _cached_corpus is None:
        print(f"[INFO] Loading BEIR corpus for '{BEIR_DATASET}'...")
        ds = datasets.load_dataset(f"mteb/{BEIR_DATASET}", "corpus")
        split_name = list(ds.keys())[0]  # usually 'corpus'
        corpus_ds = ds[split_name]
        _cached_corpus = {row["_id"]: row for row in corpus_ds}
    return _cached_corpus


def load_dataset_metadata():
    """Load queries, qrels, and corpus mapping."""
    print(f"[INFO] Loading BEIR queries for '{BEIR_DATASET}'...")
    queries_ds = datasets.load_dataset(f"mteb/{BEIR_DATASET}", "queries")
    split_name = list(queries_ds.keys())[0]
    queries_list = queries_ds[split_name]
    queries = {row["_id"]: {"query": row["text"]} for row in queries_list}

    print(f"[INFO] Loading BEIR qrels for '{BEIR_DATASET}'...")
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
    return queries, qrels, corpus


def load_and_chunk_paper(paper_id: str) -> list[Document]:
    """Load a single paper from the corpus and split it into chunks."""
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


def run_benchmark_eval(num_queries: int = 20):
    """
    Run evaluation on the first N queries from the dataset.
    Automatically finds the corresponding golden papers, indexes them,
    and runs retrieval to calculate accuracy metrics.
    """
    queries, qrels, _ = load_dataset_metadata()

    # Select evaluation subset
    eval_queries = [q_id for q_id in queries.keys() if q_id in qrels][:num_queries]
    print(f"[INFO] Selected first {len(eval_queries)} queries with qrels for evaluation.")

    # Identify the golden documents needed for these queries
    golden_doc_ids = set()
    for q_uuid in eval_queries:
        if q_uuid in qrels:
            golden_doc_ids.add(qrels[q_uuid]["doc_id"])

    print(f"[INFO] Ingesting {len(golden_doc_ids)} unique papers into Chroma...")

    # Load and index documents
    all_chunks = []
    for doc_id in golden_doc_ids:
        chunks = load_and_chunk_paper(doc_id)
        all_chunks.extend(chunks)

    if not all_chunks:
        print("[ERROR] No chunks generated. Please check dataset files.")
        return

    print(f"[INFO] Clearing existing vectorstore and indexing {len(all_chunks)} chunks...")
    try:
        count = vectorstore._collection.count()
        if count > 0:
            while True:
                existing_batch = vectorstore._collection.get(limit=500)["ids"]
                if not existing_batch:
                    break
                vectorstore.delete(ids=existing_batch)
    except Exception as e:
        print(f"[WARN] Error while clearing DB: {e}")

    vectorstore.add_documents(all_chunks)
    save_fingerprint(f"beir_eval_running_{BEIR_DATASET}")

    # Build hybrid retriever
    retriever = build_retriever(all_chunks)

    # Metrics tracking
    metrics = {
        3: {"hits": 0, "total": 0},
        5: {"hits": 0, "total": 0}
    }

    print("\n" + "="*80)
    print("RUNNING RETRIEVAL BENCHMARK")
    print("="*80)

    for q_uuid in eval_queries:
        query_text = queries[q_uuid]["query"]
        qrel_info = qrels.get(q_uuid)
        if not qrel_info:
            continue

        target_doc = qrel_info["doc_id"]
        print(f"\nQuery: {query_text!r}")
        print(f"Target Doc ID: {target_doc}")

        for k in [3, 5]:
            pipeline_result = retrieve_chunks(
                query=query_text,
                retriever=retriever,
                all_chunks=all_chunks,
                top_k=k
            )

            # Check if any retrieved chunk belongs to the target doc_id
            hit = False
            for doc in pipeline_result["retrieved_final"]:
                if doc.metadata.get("source") == target_doc:
                    hit = True
                    break

            metrics[k]["total"] += 1
            if hit:
                metrics[k]["hits"] += 1
                print(f"  [k={k}] HIT! Target document successfully retrieved.")
            else:
                print(f"  [k={k}] MISS. Target document not retrieved.")

    print("\n" + "="*80)
    print("BENCHMARK RESULTS SUMMARY")
    print("="*80)
    for k in [3, 5]:
        total = metrics[k]["total"]
        hits = metrics[k]["hits"]
        accuracy = (hits / total) * 100 if total > 0 else 0.0
        print(f"k = {k}: accuracy / recall = {accuracy:.2f}% ({hits}/{total})")
    print("="*80)


if __name__ == "__main__":
    run_benchmark_eval(num_queries=10)
