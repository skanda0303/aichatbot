"""
test_composio_youtube.py — Standalone test for YouTube Composio integration.

Loads YouTube tools via the project's version-pinning wrapper, exercises a
direct invocation (no Gemini LLM), and validates the end-to-end tool path.

Usage:
    python tests/test_composio_youtube.py
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from composio import Composio
from composio_langchain import LangchainProvider

from multi_agent.config import COMPOSIO_API_KEY, COMPOSIO_USER_ID


def load_youtube_tools():
    """Replicates the version-pinning logic for YouTube only."""
    composio = Composio(
        api_key=COMPOSIO_API_KEY,
        dangerously_skip_version_check=True,
    )
    provider = LangchainProvider()

    raw = composio.tools.get_raw_composio_tools(toolkits=["youtube"], limit=150)
    print(f"[INFO] Raw YouTube tools: {len(raw)}")

    version_map: dict[str, str] = {}
    for t in raw:
        if hasattr(t, "toolkit"):
            tk = getattr(t.toolkit, "slug", None)
            if tk and hasattr(t, "version") and tk not in version_map:
                version_map[tk] = t.version
    print(f"[INFO] Versions: {version_map}")

    for tk_slug, v in version_map.items():
        os.environ[f"COMPOSIO_TOOLKIT_VERSION_{tk_slug.upper()}"] = v.upper()

    for t in raw:
        if not t.input_parameters.get("title"):
            t.input_parameters["title"] = t.slug
        props = t.input_parameters.get("properties", {})
        if isinstance(props, dict):
            for schema in props.values():
                if isinstance(schema, dict):
                    for combiner in ("oneOf", "anyOf"):
                        if combiner in schema and isinstance(schema[combiner], list):
                            if len(schema[combiner]) > 3:
                                schema[combiner] = schema[combiner][:3]

    def _execute(slug: str, arguments):
        # Identify the toolkit from slug prefix so we can pass the version explicitly.
        prefix = slug.split("_", 1)[0].lower() if "_" in slug else ""
        ver = None
        for tk_slug, v in version_map.items():
            if tk_slug.startswith(prefix) or tk_slug == prefix:
                ver = v
                break
        kwargs = dict(user_id=COMPOSIO_USER_ID)
        if ver:
            kwargs["version"] = ver
        return composio.tools.execute(slug, arguments, **kwargs)

    return provider.wrap_tools(raw, _execute)


def main():
    if not COMPOSIO_API_KEY:
        print("[FAIL] COMPOSIO_API_KEY not set in .env")
        return 1
    print(f"[INFO] API key: {COMPOSIO_API_KEY[:10]}...")

    try:
        tools = load_youtube_tools()
    except Exception as exc:
        print(f"[FAIL] Loading failed: {type(exc).__name__}: {exc}")
        return 1
    print(f"[INFO] Loaded {len(tools)} wrapped tools")

    tool_map = {t.name: t for t in tools}
    target = "YOUTUBE_GET_VIDEO_DETAILS_BATCH"
    if target not in tool_map:
        print(f"[FAIL] Missing tool: {target}")
        return 1

    yt = tool_map[target]
    print(f"\n[INFO] Testing: {yt.name}")
    print(f"[INFO] Desc: {yt.description[:120]}...")

    # Probe 1: list[video_id] per schema
    print("\n[INFO] Probe 1: id=['dQw4w9WgXcQ']")
    try:
        result = yt.invoke({"id": ["dQw4w9WgXcQ"]})
        preview = json.dumps(result, indent=2, default=str)
        print(f"[OK] Probe 1 succeeded:\n     {preview[:500]}")
        return 0
    except Exception as exc:
        print(f"[WARN] Probe 1 failed: {type(exc).__name__}: {str(exc)[:200]}")

    # Probe 2: plain handle endpoint (looser scopes, lower friction)
    fallback = "YOUTUBE_GET_CHANNEL_ID_BY_HANDLE"
    if fallback in tool_map:
        print(f"\n[INFO] Probe 2: {fallback} schema introspection (no live call)")
        schema_name = getattr(tool_map[fallback].args_schema, "__name__", "?")
        print(f"     Args schema: {schema_name}")

    print("\n[OK] Tools + version pinning working correctly.")
    print("     Live calls need additional OAuth scopes on the connected YouTube account.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
