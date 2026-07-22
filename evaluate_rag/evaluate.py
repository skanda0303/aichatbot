"""
evaluate.py — RAG evaluation runner.

For each query provided, runs the full RAG retrieval pipeline twice:
  - once with top_k = 3  (as used in production)
  - once with top_k = 5

Prints a structured side-by-side comparison for each query and at the end
prints a summary table.

Usage:
    python -m evaluate_rag.evaluate

Queries are defined in the QUERIES list below. Replace or extend them with
your actual evaluation queries (e.g. from the Open RAG Benchmark queries.json).
"""

from evaluate_rag.ingestion import load_and_index_documents
from evaluate_rag.retriever import build_retriever
from evaluate_rag.rag_pipeline import retrieve_chunks
from evaluate_rag.config import EVAL_K_VALUES

# ── Define your evaluation queries here ───────────────────────────────────────
QUERIES = [
    "What is the main topic of the document?",
    "Summarise the key findings.",
    "What methodology was used?",
    # Add more queries or load from queries.json here
]
# ──────────────────────────────────────────────────────────────────────────────


def _divider(char: str = "─", width: int = 80) -> str:
    return char * width


def run_evaluation(queries: list[str]) -> list[dict]:
    """
    Run the dual-k evaluation for every query.

    Returns a list of result dicts, one per (query, k) pair, each containing:
        query, k, num_raw, num_unique, num_final, retrieval_time, has_rag_docs,
        chunks (list of page_content strings), rag_context
    """
    print(_divider("="))
    print("RAG EVALUATION — starting document ingestion & retriever setup")
    print(_divider("="))

    chunks   = load_and_index_documents()
    retriever = build_retriever(chunks)

    all_results: list[dict] = []

    for q_idx, query in enumerate(queries, start=1):
        print(f"\n{_divider()}")
        print(f"QUERY {q_idx}/{len(queries)}: {query!r}")
        print(_divider())

        for k in EVAL_K_VALUES:
            print(f"\n  [k={k}] Retrieving...")
            pipeline_result = retrieve_chunks(
                query=query,
                retriever=retriever,
                all_chunks=chunks,
                top_k=k,
            )

            record = {
                "query":          query,
                "k":              k,
                "num_raw":        len(pipeline_result["retrieved_raw"]),
                "num_unique":     len(pipeline_result["retrieved_unique"]),
                "num_final":      len(pipeline_result["retrieved_final"]),
                "retrieval_time": pipeline_result["retrieval_time"],
                "has_rag_docs":   pipeline_result["has_rag_docs"],
                "chunks":         [d.page_content for d in pipeline_result["retrieved_final"]],
                "rag_context":    pipeline_result["rag_context"],
            }
            all_results.append(record)

            # Per-query, per-k summary
            print(f"  ├─ Raw docs returned by retriever : {record['num_raw']}")
            print(f"  ├─ After exact dedup              : {record['num_unique']}")
            print(f"  ├─ After redundancy filter (k={k}) : {record['num_final']}")
            print(f"  ├─ Retrieval time                 : {record['retrieval_time']:.3f}s")
            print(f"  └─ Has RAG docs?                  : {record['has_rag_docs']}")

            if record["chunks"]:
                for i, chunk in enumerate(record["chunks"], start=1):
                    preview = chunk[:200].replace("\n", " ")
                    print(f"\n  [Chunk {i}] {preview}{'...' if len(chunk) > 200 else ''}")
            else:
                print("  (no chunks retrieved)")

    return all_results


def print_summary(results: list[dict]) -> None:
    """Print a compact summary table comparing k=3 vs k=5 for each query."""
    print(f"\n\n{_divider('=')}")
    print("SUMMARY TABLE")
    print(_divider("="))

    header = f"{'#':<4} {'Query':<45} {'k':<3} {'Raw':<5} {'Uniq':<6} {'Final':<7} {'Time(s)':<8} {'Docs?'}"
    print(header)
    print(_divider("-"))

    for i, r in enumerate(results, start=1):
        q_short = r["query"][:43] + ".." if len(r["query"]) > 45 else r["query"]
        print(
            f"{i:<4} {q_short:<45} {r['k']:<3} {r['num_raw']:<5} "
            f"{r['num_unique']:<6} {r['num_final']:<7} {r['retrieval_time']:<8.3f} {r['has_rag_docs']}"
        )

    print(_divider("="))


def main():
    results = run_evaluation(QUERIES)
    print_summary(results)


if __name__ == "__main__":
    main()
