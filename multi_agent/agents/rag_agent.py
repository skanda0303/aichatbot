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
def run(
    query: str,
    retriever: RerankedRetriever,
    chunks: list[Document],
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
        # Use invoke_with_scores if available (RerankedRetriever), else fall back
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

        chunks_text = [doc.page_content for doc in docs]
        metadata    = [dict(doc.metadata) for doc in docs]
        avg_score   = float(sum(scores) / len(scores)) if scores else 0.0

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
