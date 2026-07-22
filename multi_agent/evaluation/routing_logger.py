"""
evaluation/routing_logger.py — Logs RAG/Web routing decisions.

Appends a JSON line per query to 'multi_agent_routing.log' in the project root.
Each line records: timestamp, query, evaluation verdict, routing decision,
RAG chunk count, and web page count.
"""

import json
import os
from datetime import datetime

from multi_agent.models.schemas import EvalResult

_current_dir  = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.abspath(os.path.join(_current_dir, "..", ".."))
_LOG_PATH     = os.path.join(_project_root, "multi_agent_routing.log")


# Called in: multi_agent/agents/supervisor_agent.py (run_streaming, run)
def log_routing_decision(
    query: str,
    eval_result: EvalResult,
    route: str,
    rag_chunks: int = 0,
    web_pages: int = 0,
    composio_tools: int = 0,
) -> None:
    """
    Append a routing log entry.

    Example log line:
    {
      "ts": "2025-01-01T12:00:00",
      "query": "What is RAG?",
      "eval_sufficient": true,
      "confidence": 0.93,
      "route": "rag_only",
      "rag_chunks": 8,
      "web_pages": 0,
      "composio_tools": 0
    }
    """
    entry = {
        "ts":               datetime.now().isoformat(timespec="seconds"),
        "query":            query,
        "eval_sufficient":  eval_result.sufficient,
        "confidence":       round(eval_result.confidence, 4),
        "reason":           eval_result.reason,
        "route":            route,          # "rag_only" | "rag+web"
        "rag_chunks":       rag_chunks,
        "web_pages":        web_pages,
        "composio_tools":   composio_tools,
    }
    try:
        with open(_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"[ROUTING LOGGER] Failed to write log: {e}")
