"""
evaluate_rag — RAG evaluation package.

Mirrors the ingestion / retriever / RAG-pipeline logic from ragbot/
and adds a dual-k evaluator (k=3 vs k=5) for each query.

Run:
    python -m evaluate_rag.evaluate
"""
