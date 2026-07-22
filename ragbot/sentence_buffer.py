"""
sentence_buffer.py — Token-by-token sentence boundary detector.

Accumulates streaming LLM tokens and emits complete sentences as they
are detected. Designed to be used once per conversation turn.

Usage:
    buf = SentenceBuffer()
    for token in token_stream:
        for sentence in buf.feed(token):
            process(sentence)
    remainder = buf.flush()
    if remainder:
        process(remainder)
"""

import re

# Sentence-ending boundary: one of . ! ? followed by whitespace or newline,
# OR a standalone newline (paragraph break). We split AFTER the boundary char.
_BOUNDARY = re.compile(r'(?<=[.!?])\s+|\n{2,}')


class SentenceBuffer:
    """
    Stateful accumulator that detects sentence boundaries in a token stream.

    feed() is called once per token. It returns a (possibly empty) list of
    complete sentences extracted from the buffer. flush() drains any remaining
    partial text at end-of-stream so nothing is dropped.
    """

    def __init__(self) -> None:
        self._buf: str = ""

    # ------------------------------------------------------------------
    def feed(self, token: str) -> list[str]:
        """
        Append *token* to the internal buffer.

        Returns a list of zero or more complete sentences that have been
        delimited so far. Each returned sentence has already been stripped
        of leading/trailing whitespace and is guaranteed to be non-empty.
        """
        self._buf += token
        sentences: list[str] = []

        # Repeatedly extract everything up-to-and-including the boundary.
        while True:
            m = _BOUNDARY.search(self._buf)
            if m is None:
                break
            # The sentence is everything up to the end of the match.
            sentence = self._buf[: m.end()].strip()
            self._buf = self._buf[m.end():]
            if sentence:
                sentences.append(sentence)

        return sentences

    # ------------------------------------------------------------------
    def flush(self) -> str | None:
        """
        Return any remaining partial text and clear the buffer.

        Call this once after the token stream is exhausted. Returns None
        if nothing remains (empty or only whitespace).
        """
        remainder = self._buf.strip()
        self._buf = ""
        return remainder if remainder else None

    # ------------------------------------------------------------------
    def reset(self) -> None:
        """Discard buffer contents — call between conversation turns."""
        self._buf = ""
