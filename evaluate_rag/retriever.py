"""
retriever.py — Hybrid BM25 + vector retriever and redundancy filter.

Identical logic to ragbot/retriever.py.
Combines keyword-based (BM25) and semantic (vector) search at 30/70 weight.
Also provides filter_redundant_docs() to remove near-duplicate chunks
using word-level Jaccard similarity before they reach the LLM.
"""

from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.documents import Document

from evaluate_rag.config import RETRIEVER_K, BM25_WEIGHT, VECTOR_WEIGHT, REDUNDANCY_THRESHOLD
from evaluate_rag.ingestion import vectorstore


def _word_set(text: str) -> set:
    return set(
        w.strip(".,;:()[]●-●*").lower()
        for w in text.split()
        if len(w.strip(".,;:()[]●-●*")) > 1
    )


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


class RerankedRetriever:
    def __init__(self, base_retriever, reranker_model_name="BAAI/bge-reranker-v2-m3", top_n=RETRIEVER_K):
        self.base_retriever = base_retriever
        self.top_n = top_n
        from sentence_transformers import CrossEncoder
        print(f"[INFO] Loading BGE Reranker model '{reranker_model_name}'...")
        self.model = CrossEncoder(reranker_model_name)
        print("[INFO] BGE Reranker loaded successfully.")

    def invoke(self, query: str) -> list[Document]:
        docs = self.base_retriever.invoke(query)
        if not docs:
            return []
        
        seen = set()
        unique_docs = []
        for doc in docs:
            if doc.page_content not in seen:
                seen.add(doc.page_content)
                unique_docs.append(doc)
                
        pairs = [[query, doc.page_content] for doc in unique_docs]
        from sentence_transformers import CrossEncoder
        scores = self.model.predict(pairs)
        
        scored_docs = sorted(zip(unique_docs, scores), key=lambda x: x[1], reverse=True)
        reranked_docs = [doc for doc, score in scored_docs]
        return reranked_docs[:self.top_n]


def build_retriever(chunks: list[Document]):
    """Build hybrid retriever if chunks exist, otherwise fall back to vector-only."""
    if chunks:
        bm25_retriever = BM25Retriever.from_documents(chunks)
        bm25_retriever.k = RETRIEVER_K

        vector_retriever = vectorstore.as_retriever(
            search_type="similarity", search_kwargs={"k": RETRIEVER_K},
        )

        ensemble_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, vector_retriever],
            weights=[BM25_WEIGHT, VECTOR_WEIGHT],
        )
        
        retriever = RerankedRetriever(ensemble_retriever, reranker_model_name="BAAI/bge-reranker-v2-m3", top_n=RETRIEVER_K)
        print(f"[OK] Hybrid retriever initialized with BGE M3 Reranker — BM25 {BM25_WEIGHT} / Vector {VECTOR_WEIGHT}")
        return retriever

    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})
    print("[WARNING] No documents found. Defaulting to vector retriever.")
    return retriever

