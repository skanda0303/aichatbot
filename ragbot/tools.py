"""
tools.py — LangChain agent tools.

Defines four tools the agent can call:
  - rag_search: search uploaded PDFs via hybrid retriever
  - web_search: query Tavily API + auto-fetch top result pages
  - fetch_webpage_content: fetch full text of a specific URL
  - get_datetime: return current date and time
"""

import json
import re
import urllib.request
from datetime import datetime

from bs4 import BeautifulSoup
from langchain_core.tools import tool

from ragbot.config import TAVILY_API_KEY, MAX_FETCH_CHARS

# RAG context — injected at startup via register_rag_context()
_retriever = None
_chunks: list = []


def register_rag_context(retriever, chunks: list) -> None:
    """Called once at startup to supply the retriever and chunks to this module."""
    global _retriever, _chunks
    _retriever = retriever
    _chunks    = chunks


def _fetch_url(url: str) -> str:
    """Download, clean, and truncate the text content of a webpage."""
    try:
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/120.0.0.0 Safari/537.36"
                )
            },
        )
        with urllib.request.urlopen(req, timeout=6) as response:
            html = response.read()

        soup = BeautifulSoup(html, "html.parser")
        for element in soup(["script", "style", "nav", "footer", "header", "aside"]):
            element.decompose()

        text = soup.get_text(separator=" ")
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = "\n".join(chunk for chunk in chunks if chunk)
        text = re.sub(r"\n+", "\n", text)
        text = re.sub(r" +", " ", text)

        if len(text) > MAX_FETCH_CHARS:
            text = text[:MAX_FETCH_CHARS] + "... [truncated]"
        return text
    except Exception as e:
        return f"Failed to fetch webpage: {e}"


@tool
def rag_search(query: str) -> str:
    """Search the uploaded PDF documents for information related to the query.
    Always call this first for any question that might be in the documents.
    Returns 'NO_RELEVANT_DOCS' if nothing useful is found — in that case, use web_search as fallback."""
    from ragbot.retriever import filter_redundant_docs

    if not _chunks:
        return "NO_RELEVANT_DOCS: No documents have been uploaded."
    try:
        results = _retriever.invoke(query)
    except Exception as e:
        return f"NO_RELEVANT_DOCS: Retrieval error — {e}"

    if not results:
        return "NO_RELEVANT_DOCS: No matching documents found."

    seen, unique = set(), []
    for doc in results:
        if doc.page_content not in seen:
            seen.add(doc.page_content)
            unique.append(doc)

    filtered = filter_redundant_docs(unique)
    if not filtered:
        return "NO_RELEVANT_DOCS: All retrieved documents were redundant."

    formatted = "\n\n---\n\n".join(
        doc.page_content.replace("●", "\n- ") for doc in filtered[:8]
    )
    print(f"  [RAG TOOL] Returned {len(filtered[:8])} chunks.")
    return formatted


@tool
def web_search(query: str) -> str:
    """Search the web for real-time information. Use as a fallback when rag_search returns NO_RELEVANT_DOCS."""
    try:
        print(f"  [WEB TOOL] Querying Tavily API for: '{query}'")

        VIDEO_BLACKLIST = [
            "youtube.com", "youtu.be", "vimeo.com",
            "dailymotion.com", "tiktok.com", "instagram.com",
        ]

        payload = {
            "api_key": TAVILY_API_KEY,
            "query": query,
            "search_depth": "basic",
            "include_answer": False,
            "max_results": 5,
        }

        req = urllib.request.Request(
            "https://api.tavily.com/search",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        with urllib.request.urlopen(req, timeout=10) as response:
            res_data = json.loads(response.read().decode("utf-8"))

        search_results = res_data.get("results", [])
        filtered_results = [
            r for r in search_results
            if not any(bl in r.get("url", "").lower() for bl in VIDEO_BLACKLIST)
        ]

        if not filtered_results:
            return "No web results found for this query."

        print(f"  [WEB TOOL] {len(filtered_results)} results. Auto-fetching top 4...")

        formatted_sources = []
        for i, res in enumerate(filtered_results[:4]):
            url     = res.get("url", "")
            title   = res.get("title", "")
            snippet = res.get("content", "")
            score   = res.get("score", 0.0)

            if url:
                page_content = _fetch_url(url)
                if page_content and not page_content.startswith("Failed to fetch webpage"):
                    snippet = page_content

            formatted_sources.append(
                f"=== SOURCE {i + 1} ===\n"
                f"Title: {title}\nURL: {url}\n"
                f"Relevance: {score}\nContent:\n{snippet}"
            )

        return "\n\n---\n\n".join(formatted_sources)
    except Exception as e:
        return f"Web search failed: {e}"


@tool
def fetch_webpage_content(url: str) -> str:
    """Fetch full text of a specific URL for detailed information."""
    return _fetch_url(url)


@tool
def get_datetime() -> str:
    """Returns the current date, time, and day of the week."""
    now = datetime.now()
    return f"Current date and time: {now.strftime('%A, %B %d, %Y at %H:%M:%S')}"
