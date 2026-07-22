"""
agents/rag_agent.py — RAG Retrieval Agent.

Performs the complete hybrid retrieval pipeline (BM25 + vector + CrossEncoder
reranking + Jaccard deduplication) and returns a structured RAGResult.

Rules:
  - No answer generation
  - No web search
  - No LLM calls
  - Returns RAGResult with chunks, scores, and metadata
"""

import time

from langchain_core.documents import Document

from multi_agent.models.schemas import RAGResult
from multi_agent.retrieval.retriever import RerankedRetriever, filter_redundant_docs
from multi_agent.config import RETRIEVER_K


# Called in: multi_agent/agents/supervisor_agent.py (run_streaming, run)
import os

def run(
    query: str,
    retriever: RerankedRetriever,
    chunks: list[Document],
    selected_doc: str | None = None,
) -> RAGResult:
    """
    Execute the full retrieval pipeline and return a RAGResult.

    Pipeline:
      BM25 search
        ↓
      Vector search
        ↓
      Ensemble merge
        ↓
      CrossEncoder rerank
        ↓
      Remove duplicates (Jaccard)
        ↓
      Top-K chunks
    """
    if not chunks:
        print("[RAG AGENT] No documents available.")
        return RAGResult(
            retrieved_chunks=[],
            avg_retrieval_score=0.0,
            cross_encoder_scores=[],
            metadata=[],
        )

    t_start = time.perf_counter()

    try:
        # If user explicitly selected a document, isolate retrieval to that document's chunks
        target_chunks = []
        if selected_doc:
            target_chunks = [
                c for c in chunks 
                if selected_doc.lower() in os.path.basename(str(c.metadata.get("source", ""))).lower()
            ]

        if target_chunks:
            from multi_agent.retrieval.retriever import build_retriever
            print(f"[RAG AGENT] Filtering retrieval strictly to selected document: '{selected_doc}' ({len(target_chunks)} chunks)")
            target_retriever = build_retriever(target_chunks)
            if hasattr(target_retriever, "invoke_with_scores"):
                docs, scores = target_retriever.invoke_with_scores(query)
            else:
                docs   = target_retriever.invoke(query)
                scores = [0.0] * len(docs)
        else:
            if hasattr(retriever, "invoke_with_scores"):
                docs, scores = retriever.invoke_with_scores(query)
            else:
                docs   = retriever.invoke(query)
                scores = [0.0] * len(docs)

        elapsed = time.perf_counter() - t_start
        print(f"[RAG AGENT] Retrieved {len(docs)} docs in {elapsed:.3f}s")

        # Additional Jaccard deduplication pass
        docs = filter_redundant_docs(docs)
        docs = docs[:RETRIEVER_K]
        scores = scores[:len(docs)]

        chunks_text = []
        for doc in docs:
            src = doc.metadata.get("source", "Knowledge Base")
            src_name = os.path.basename(str(src))
            page_info = f" (Page {doc.metadata.get('page', 0) + 1})" if "page" in doc.metadata else ""
            chunks_text.append(f"[Source: {src_name}{page_info}]\n{doc.page_content}")

        metadata  = [dict(doc.metadata) for doc in docs]
        avg_score = float(sum(scores) / len(scores)) if scores else 0.0

        print(
            f"[RAG AGENT] Returning {len(chunks_text)} chunks | "
            f"avg CrossEncoder score: {avg_score:.4f}"
        )

        return RAGResult(
            retrieved_chunks=chunks_text,
            avg_retrieval_score=avg_score,
            cross_encoder_scores=scores,
            metadata=metadata,
        )

    except Exception as e:
        print(f"[RAG AGENT] Retrieval error: {e}")
        return RAGResult(
            retrieved_chunks=[],
            avg_retrieval_score=0.0,
            cross_encoder_scores=[],
            metadata=[],
        )
