"""
tools/scraper_tools.py — Low-level HTML-fetching helper shared between web_agent and web_tools.
"""

import re
import urllib.request

from bs4 import BeautifulSoup

from multi_agent.config import MAX_FETCH_CHARS


# Called in: multi_agent/tools/web_tools.py (fetch_and_clean_results)
def fetch_url(url: str) -> str:
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
