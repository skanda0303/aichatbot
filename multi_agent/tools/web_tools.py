"""
tools/web_tools.py — Tavily search helper used by the Web Search Agent.

Same configuration as ragbot/tools.py:
  - search_depth: "basic"
  - include_answer: False
  - max_results: MAX_WEB_RESULTS (5)
  - video blacklist applied
  - auto-fetches top page HTML via fetch_url()
"""

import json
import urllib.request

from multi_agent.config import TAVILY_API_KEY, MAX_WEB_RESULTS
from multi_agent.tools.scraper_tools import fetch_url

# Sites that return video content — excluded from web results
_VIDEO_BLACKLIST = [
    "youtube.com", "youtu.be", "vimeo.com",
    "dailymotion.com", "tiktok.com", "instagram.com",
]


# Called in: multi_agent/agents/web_agent.py (run)
def tavily_search(query: str, user_tavily_key: str | None = None) -> list[dict]:
    """
    Call Tavily API and return a list of result dicts:
      [{"url": str, "title": str, "snippet": str, "score": float}, ...]

    The caller (web_agent) is responsible for fetching page content.
    """
    payload = {
        "api_key": user_tavily_key or TAVILY_API_KEY,
        "query": query,
        "search_depth": "advanced",
        "include_answer": False,
        "max_results": MAX_WEB_RESULTS,
    }

    req = urllib.request.Request(
        "https://api.tavily.com/search",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urllib.request.urlopen(req, timeout=10) as response:
        res_data = json.loads(response.read().decode("utf-8"))

    results = res_data.get("results", [])
    filtered = [
        r for r in results
        if not any(bl in r.get("url", "").lower() for bl in _VIDEO_BLACKLIST)
    ]
    return filtered


# Called in: multi_agent/agents/web_agent.py (run)
def fetch_and_clean_results(results: list[dict], top_n: int = 4) -> list[dict]:
    """
    For each result (up to top_n), attempt to fetch and clean the full page HTML.
    Returns enriched result dicts with "content" replaced by full page text where possible.
    """
    enriched = []
    for res in results[:top_n]:
        url     = res.get("url", "")
        title   = res.get("title", "")
        snippet = res.get("content", "")
        score   = res.get("score", 0.0)

        if url:
            page_content = fetch_url(url)
            if page_content and not page_content.startswith("Failed to fetch"):
                snippet = page_content

        enriched.append({"url": url, "title": title, "snippet": snippet, "score": float(score)})
    return enriched
