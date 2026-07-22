"""
rag_pipeline.py — Core RAG retrieval pipeline for evaluation.

Mirrors the pre-fetch RAG logic from ragbot/agent.py (_build_inputs).
No LLM, no web tools — only document retrieval, deduplication, and
redundancy filtering, so the evaluator can inspect what chunks are
actually handed to the model for each (query, k) combination.
"""

import time

from langchain_core.documents import Document

from evaluate_rag.config import REDUNDANCY_THRESHOLD
from evaluate_rag.retriever import filter_redundant_docs


def sanitize_tool_output(text: str) -> str:
    """Normalize whitespace and cap at 25k chars to prevent context overflow."""
    text = " ".join(text.split())
    if len(text) > 25000:
        text = text[:25000] + "... [truncated]"
    return text


def retrieve_chunks(query: str, retriever, all_chunks: list[Document], top_k: int) -> dict:
    """
    Run the identical RAG pre-fetch logic from ragbot/agent.py for a given query
    and a requested number of final chunks (top_k).

    Returns a dict with:
        retrieved_raw   : list[Document]  — raw docs from retriever (before dedup)
        retrieved_unique: list[Document]  — after exact-duplicate removal
        retrieved_final : list[Document]  — after redundancy filter, capped at top_k
        rag_context     : str             — the concatenated context string
        retrieval_time  : float           — seconds spent in retriever.invoke()
        has_rag_docs    : bool
    """
    result = {
        "retrieved_raw":    [],
        "retrieved_unique": [],
        "retrieved_final":  [],
        "rag_context":      "",
        "retrieval_time":   0.0,
        "has_rag_docs":     False,
    }

    if not all_chunks:
        return result

    try:
        t_ret = time.perf_counter()
        retrieved_docs = retriever.invoke(query)
        result["retrieval_time"] = time.perf_counter() - t_ret
        result["retrieved_raw"]  = retrieved_docs

        print(f"[EVAL] Retrieval: {result['retrieval_time']:.3f}s — {len(retrieved_docs)} raw docs")

        if not retrieved_docs:
            return result

        # Step 1: exact-content deduplication (same as ragbot/agent.py)
        seen, unique = set(), []
        for doc in retrieved_docs:
            if doc.page_content not in seen:
                seen.add(doc.page_content)
                unique.append(doc)
        result["retrieved_unique"] = unique

        # Step 2: Jaccard-similarity redundancy filter (same threshold as production)
        filtered = filter_redundant_docs(unique, threshold=REDUNDANCY_THRESHOLD)

        # Step 3: Cap to the requested top_k (production uses [:3] hardcoded)
        final = filtered[:top_k]
        result["retrieved_final"] = final

        if final:
            result["has_rag_docs"] = True
            formatted_chunks = [doc.page_content.replace("●", "\n- ") for doc in final]
            result["rag_context"] = sanitize_tool_output("\n\n---\n\n".join(formatted_chunks))
            print(f"[EVAL] k={top_k}: injecting {len(final)} RAG chunks after filtering.")

    except Exception as e:
        print(f"[EVAL] Retrieval error: {e}")

    return result
