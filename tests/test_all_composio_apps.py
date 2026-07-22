"""
test_all_composio_apps.py — Verify every connected Composio app works end-to-end.

For each connected account we run one safe, read-only action through
composio.tools.execute(), using the version-pinning + user_id pattern from
multi_agent/tools/composio_tools.py.

Usage:
    python tests/test_all_composio_apps.py
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from composio import Composio
from multi_agent.config import COMPOSIO_API_KEY, COMPOSIO_USER_ID


# A safe read-only probe action per toolkit (None = skip live call, just report)
PROBES = {
    "github": ("GITHUB_LIST_REPOSITORIES_FOR_THE_AUTHENTICATED_USER", {}),
    "youtube": ("YOUTUBE_GET_CHANNEL_ID_BY_HANDLE", {"handle": "google"}),
    "googledocs": ("GOOGLEDOCUMENTS_GET_DOCUMENT", None),  # needs doc id; report only
    "tavily": ("TAVILY_TAVILY_SEARCH", {"query": "Composio"}),
    "context7_mcp": ("CONTEXT7_MCP_RESOLVE_LIBRARY_ID", {"libraryName": "python"}),
    "hugging_face": ("HUGGINGFACE_SEARCH_MODELS", {"query": "bert"}),
}


def main():
    if not COMPOSIO_API_KEY:
        print("[FAIL] COMPOSIO_API_KEY not set")
        return 1

    composio = Composio(api_key=COMPOSIO_API_KEY, dangerously_skip_version_check=True)

    # 1. List connected accounts
    accs = composio.connected_accounts.list(user_ids=[COMPOSIO_USER_ID])
    items = accs.items if hasattr(accs, "items") else accs
    print(f"[INFO] {len(items)} connected accounts:\n")

    # 2. Build version map
    version_map = {}
    for tk in set(getattr(a.toolkit, "slug", None) for a in items):
        if not tk:
            continue
        raw = composio.tools.get_raw_composio_tools(toolkits=[tk], limit=10)
        for t in raw:
            if hasattr(t, "toolkit"):
                tk2 = getattr(t.toolkit, "slug", None)
                if tk2 and hasattr(t, "version") and tk2 not in version_map:
                    version_map[tk2] = t.version

    results = []
    for acc in items:
        slug = getattr(acc.toolkit, "slug", None)
        status = acc.status
        print(f"=== {slug}  (status={status}) ===")
        if status != "ACTIVE":
            print(f"  [SKIP] not ACTIVE")
            results.append((slug, "SKIP", "not ACTIVE"))
            continue

        probe = PROBES.get(slug)
        if not probe:
            print(f"  [INFO] no probe defined")
            results.append((slug, "INFO", "no probe"))
            continue

        action, args = probe
        if args is None:
            print(f"  [INFO] probe '{action}' requires params; reporting only")
            results.append((slug, "INFO", "needs params"))
            continue

        ver = version_map.get(slug)
        kwargs = dict(user_id=COMPOSIO_USER_ID)
        if ver:
            kwargs["version"] = ver
        try:
            resp = composio.tools.execute(action, args, **kwargs)
            # Summarize
            if isinstance(resp, dict) and "data" in resp:
                data = resp["data"]
                if isinstance(data, dict):
                    n = len(data.get("repositories", data.get("results", data.get("models", []))))
                    summary = f"{n} items" if n else f"keys={list(data.keys())[:5]}"
                else:
                    summary = str(data)[:80]
            else:
                summary = str(resp)[:80]
            print(f"  [OK] {action} -> {summary}")
            results.append((slug, "OK", action))
        except Exception as exc:
            print(f"  [FAIL] {action}: {type(exc).__name__}: {str(exc)[:200]}")
            results.append((slug, "FAIL", str(exc)[:120]))

    # Summary
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    ok = sum(1 for _, s, _ in results if s == "OK")
    for slug, status, detail in results:
        print(f"  {slug:14s} {status:5s} {detail}")
    print(f"\n{ok}/{len(results)} apps returned OK on a live probe call.")
    return 0


if __name__ == "__main__":
    sys.exit(main())