"""
schemas.py — Pydantic data models for inter-agent communication.

These typed schemas are the contracts between agents.  Every agent
returns one of these models so that the Supervisor can pass data
between pipeline stages without ambiguity.
"""

from __future__ import annotations
from typing import Any
from pydantic import BaseModel, Field


class RAGResult(BaseModel):
    """Output from the RAG Agent."""

    retrieved_chunks: list[str] = Field(
        default_factory=list,
        description="Ordered list of retrieved text chunks (post-rerank, deduped).",
    )
    avg_retrieval_score: float = Field(
        default=0.0,
        description="Mean CrossEncoder score across returned chunks.",
    )
    cross_encoder_scores: list[float] = Field(
        default_factory=list,
        description="Per-chunk CrossEncoder scores, aligned with retrieved_chunks.",
    )
    metadata: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Source metadata for each chunk (source file, page, etc.).",
    )


class EvalResult(BaseModel):
    """Output from the Context Evaluation Agent."""

    sufficient: bool = Field(
        description="True if retrieved RAG context fully answers the question.",
    )
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="Confidence score in the sufficiency verdict (0–1).",
    )
    reason: str = Field(
        description="Human-readable explanation of the evaluation decision.",
    )


class WebResult(BaseModel):
    """Output from the Web Search Agent."""

    web_context: list[str] = Field(
        default_factory=list,
        description="Cleaned text content from fetched web pages.",
    )
    source_urls: list[str] = Field(
        default_factory=list,
        description="URLs corresponding to each entry in web_context.",
    )
    confidence: float = Field(
        default=0.0,
        description="Average Tavily relevance score across returned results.",
    )
