"""
tts_service.py — Fish Audio 2.1 Pro TTS wrapper.

Exposes:
  synthesize(text)  — async; returns raw mp3/opus bytes or None on failure.
  sanitize_for_tts(text) — strips markdown before sending to TTS.

The underlying fish_audio_sdk Session.tts() call is synchronous/blocking,
so it is dispatched to a thread-pool executor to keep the async event loop
unblocked.

An optional in-memory cache avoids re-synthesising repeated identical phrases
(e.g. "code block omitted").
"""

import asyncio
import base64
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor

from fish_audio_sdk import Session, TTSRequest

from ragbot.config import (
    FISH_AUDIO_API_KEY,
    FISH_AUDIO_VOICE_ID,
    TTS_FORMAT,
    TTS_BACKEND,
    TTS_AVAILABLE,
)

logger = logging.getLogger(__name__)

# ── Thread pool for blocking SDK calls ────────────────────────────────────────
_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="tts-worker")

# ── Optional in-memory sentence cache (text → bytes) ─────────────────────────
_CACHE: dict[str, bytes] = {}
_CACHE_MAX = 128   # evict oldest entries beyond this size


# ── Sanitizer ─────────────────────────────────────────────────────────────────
def sanitize_for_tts(text: str) -> str:
    """
    Strip markdown syntax from *text* before sending to TTS.

    Rules applied (in order):
    1. Triple-backtick code fences (with or without language tag) → "code block omitted"
    2. Inline backtick spans → bare inner text
    3. Markdown links [label](url) → label only
    4. Bold/italic markers (**text**, *text*, __text__, _text_)
    5. Leading bullet / list markers at line start (-, *, 1.)
    6. Excess whitespace / newlines → single space
    """
    # 1. Fenced code blocks
    text = re.sub(r'```[\s\S]*?```', ' code block omitted ', text)

    # 2. Inline code
    text = re.sub(r'`[^`\n]+`', lambda m: m.group(0)[1:-1], text)

    # 3. Markdown links
    text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)

    # 4. Bold (**text** or __text__)
    text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
    text = re.sub(r'__([^_]+)__', r'\1', text)

    # 5. Italic (*text* or _text_)
    text = re.sub(r'\*([^*]+)\*', r'\1', text)
    text = re.sub(r'_([^_]+)_', r'\1', text)

    # 6. Leading bullet / list markers
    text = re.sub(r'(?m)^[ \t]*[-*•]\s+', '', text)
    text = re.sub(r'(?m)^\s*\d+[.)]\s+', '', text)

    # 7. Normalise whitespace
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n+', ' ', text)

    return text.strip()


# ── Core synthesiser ──────────────────────────────────────────────────────────
def _blocking_synthesize(text: str) -> bytes | None:
    """
    Blocking Fish Audio TTS call — runs inside a thread-pool worker.
    Returns concatenated audio bytes, or None on any error.
    """
    try:
        session = Session(FISH_AUDIO_API_KEY)
        request = TTSRequest(
            text=text,
            reference_id=FISH_AUDIO_VOICE_ID or None,
            format=TTS_FORMAT,
        )
        chunks: list[bytes] = []
        for chunk in session.tts(request, backend=TTS_BACKEND):
            chunks.append(chunk)
        audio_bytes = b"".join(chunks)
        logger.debug("[TTS] Synthesised %d bytes for: %.60s…", len(audio_bytes), text)
        return audio_bytes
    except Exception as exc:
        logger.warning("[TTS] Synthesis failed: %s", exc)
        return None


async def synthesize(text: str) -> bytes | None:
    """
    Async wrapper around _blocking_synthesize.

    Returns raw audio bytes (mp3 by default) or None on failure.
    Caches results in memory to avoid duplicate API calls for identical text.
    """
    if not TTS_AVAILABLE:
        return None

    # Cache lookup
    cache_key = text.strip()
    if cache_key in _CACHE:
        logger.debug("[TTS] Cache hit for: %.60s…", cache_key)
        return _CACHE[cache_key]

    t0 = time.perf_counter()
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(_executor, _blocking_synthesize, text)
    logger.info("[TTS] %.3fs for %d chars", time.perf_counter() - t0, len(text))

    if result is not None:
        # Evict oldest entries if cache is full
        if len(_CACHE) >= _CACHE_MAX:
            oldest = next(iter(_CACHE))
            del _CACHE[oldest]
        _CACHE[cache_key] = result

    return result


def audio_to_b64(audio_bytes: bytes) -> str:
    """Encode raw audio bytes to a base64 string suitable for JSON transport."""
    return base64.b64encode(audio_bytes).decode("ascii")
