"""
utils/helpers.py — Shared utility functions for the multi-agent pipeline.
"""

import re


# Called in: multi_agent/agents/answer_agent.py (_build_user_message)
def sanitize_tool_output(text: str) -> str:
    """Normalize whitespace and cap at 25 k chars to prevent context overflow."""
    text = " ".join(text.split())
    if len(text) > 25000:
        text = text[:25000] + "... [truncated]"
    return text


# Unused in production
def word_set(text: str) -> set[str]:
    """Return a cleaned lower-case word set for Jaccard-based deduplication."""
    return set(
        w.strip(".,;:()[]●-●*").lower()
        for w in text.split()
        if len(w.strip(".,;:()[]●-●*")) > 1
    )


# Called in: multi_agent/agents/evaluation_agent.py (run), multi_agent/agents/answer_agent.py (_build_user_message)
def format_chunks_for_prompt(chunks: list[str], max_chunks: int = 8) -> str:
    """Join retrieved chunks with a visible separator for LLM prompts."""
    selected = chunks[:max_chunks]
    formatted = [c.replace("●", "\n- ") for c in selected]
    return "\n\n---\n\n".join(formatted)
