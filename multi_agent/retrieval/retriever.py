"""
retrieval/retriever.py — Hybrid BM25 + vector retriever with CrossEncoder reranking.

Identical logic to ragbot/retriever.py — same BM25/vector weights (0.3/0.7),
same CrossEncoder model (BAAI/bge-reranker-v2-m3), same RETRIEVER_K, same
Jaccard-based redundancy filter.
"""

import os
import logging

# Disable third-party loggers from printing connection warning retry loops to console/stderr
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("urllib3").setLevel(logging.ERROR)
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["HF_HUB_VERBOSITY"] = "error"

from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.documents import Document

from multi_agent.config import RETRIEVER_K, BM25_WEIGHT, VECTOR_WEIGHT, REDUNDANCY_THRESHOLD
from multi_agent.retrieval.ingestion import vectorstore


# Called in: multi_agent/retrieval/retriever.py (filter_redundant_docs)
def _word_set(text: str) -> set:
    return set(
        w.strip(".,;:()[]●-●*").lower()
        for w in text.split()
        if len(w.strip(".,;:()[]●-●*")) > 1
    )


# Called in: multi_agent/agents/rag_agent.py (run)
def filter_redundant_docs(docs: list[Document], threshold: float = REDUNDANCY_THRESHOLD) -> list[Document]:
    """Remove near-duplicate documents using word-level Jaccard similarity."""
    unique: list[Document] = []
    for doc in docs:
        words = _word_set(doc.page_content)
        redundant = False
        for u in unique:
            u_words = _word_set(u.page_content)
            if not words or not u_words:
                continue
            if len(words & u_words) / min(len(words), len(u_words)) > threshold:
                redundant = True
                break
        if not redundant:
            unique.append(doc)
    return unique


# Safely detect Hugging Face ZeroGPU environment
try:
    import spaces
    _HAS_SPACES = True
except ImportError:
    _HAS_SPACES = False

if _HAS_SPACES:
    @spaces.GPU
    def _predict_scores_gpu(model, pairs):
        return model.predict(pairs)
else:
    def _predict_scores_gpu(model, pairs):
        return model.predict(pairs)


class RerankedRetriever:
    """
    Wraps an EnsembleRetriever and applies CrossEncoder reranking.
    Identical to ragbot.retriever.RerankedRetriever — same model and top_n.
    """

    # Called in: multi_agent/retrieval/retriever.py (build_retriever)
    def __init__(
        self,
        base_retriever,
        reranker_model_name: str = "BAAI/bge-reranker-v2-m3",
        top_n: int = RETRIEVER_K,
    ):
        self.base_retriever = base_retriever
        self.top_n = top_n
        from sentence_transformers import CrossEncoder
        print(f"[INFO] Loading BGE Reranker model '{reranker_model_name}'...")
        try:
            self.model = CrossEncoder(reranker_model_name)
            print("[INFO] BGE Reranker loaded successfully.")
        except Exception as e:
            print(f"[WARNING] Failed to load BGE Reranker model: {e}. Falling back to no reranking.")
            self.model = None

    # Called in: multi_agent/agents/rag_agent.py (run)
    def invoke(self, query: str) -> list[Document]:
        docs = self.base_retriever.invoke(query)
        if not docs:
            return []

        seen = set()
        unique_docs: list[Document] = []
        for doc in docs:
            if doc.page_content not in seen:
                seen.add(doc.page_content)
                unique_docs.append(doc)

        if self.model is None:
            return unique_docs[: self.top_n]

        pairs = [[query, doc.page_content] for doc in unique_docs]
        scores = _predict_scores_gpu(self.model, pairs)

        scored_docs = sorted(zip(unique_docs, scores), key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in scored_docs[: self.top_n]]

    # Called in: multi_agent/agents/rag_agent.py (run)
    def invoke_with_scores(self, query: str) -> tuple[list[Document], list[float]]:
        """Like invoke() but also returns the CrossEncoder scores for each chunk."""
        docs = self.base_retriever.invoke(query)
        if not docs:
            return [], []

        seen = set()
        unique_docs: list[Document] = []
        for doc in docs:
            if doc.page_content not in seen:
                seen.add(doc.page_content)
                unique_docs.append(doc)

        if self.model is None:
            top_docs = unique_docs[: self.top_n]
            return top_docs, [1.0] * len(top_docs)

        pairs = [[query, doc.page_content] for doc in unique_docs]
        scores = _predict_scores_gpu(self.model, pairs)

        scored_pairs = sorted(zip(unique_docs, scores), key=lambda x: x[1], reverse=True)
        top_docs   = [doc   for doc, _ in scored_pairs[: self.top_n]]
        top_scores = [float(score) for _, score in scored_pairs[: self.top_n]]
        return top_docs, top_scores


# Called in: multi_agent/main.py, multi_agent/agents/rag_agent.py
def build_retriever(chunks: list[Document]) -> RerankedRetriever:
    """Build a retriever. Uses BM25-only for provided chunks, ChromaDB fallback otherwise."""
    if chunks:
        bm25_retriever = BM25Retriever.from_documents(chunks)
        bm25_retriever.k = min(len(chunks), RETRIEVER_K)
        return RerankedRetriever(bm25_retriever)

    fallback_retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})
    return RerankedRetriever(fallback_retriever)
