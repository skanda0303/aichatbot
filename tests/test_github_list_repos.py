"""
test_github_list_repos.py — Test GitHub integration by listing YOUR repos.

Composio's GitHub toolkit doesn't expose a wrapped "list repos" tool, but the
underlying action GITHUB_LIST_REPOSITORIES_FOR_THE_AUTHENTICATED_USER works via
tools.execute(). This exercises the same version-pinning + user_id path used by
multi_agent/tools/composio_tools.py.

Usage:
    python tests/test_github_list_repos.py
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from composio import Composio
from multi_agent.config import COMPOSIO_API_KEY, COMPOSIO_USER_ID


def main():
    if not COMPOSIO_API_KEY:
        print("[FAIL] COMPOSIO_API_KEY not set")
        return 1

    composio = Composio(
        api_key=COMPOSIO_API_KEY,
        dangerously_skip_version_check=True,
    )

    # Discover the github toolkit version (mirrors production logic)
    raw = composio.tools.get_raw_composio_tools(toolkits=["github"], limit=150)
    version_map = {}
    for t in raw:
        if hasattr(t, "toolkit"):
            tk = getattr(t.toolkit, "slug", None)
            if tk and hasattr(t, "version") and tk not in version_map:
                version_map[tk] = t.version
    print(f"[INFO] GitHub toolkit version: {version_map.get('github')}")

    ver = version_map.get("github")
    kwargs = dict(user_id=COMPOSIO_USER_ID)
    if ver:
        kwargs["version"] = ver

    print("\n[INFO] Executing GITHUB_LIST_REPOSITORIES_FOR_THE_AUTHENTICATED_USER ...")
    try:
        result = composio.tools.execute(
            "GITHUB_LIST_REPOSITORIES_FOR_THE_AUTHENTICATED_USER",
            {},
            **kwargs,
        )
    except Exception as exc:
        print(f"[FAIL] Execution error: {type(exc).__name__}: {str(exc)[:300]}")
        return 1

    data = result.get("data", result) if isinstance(result, dict) else result
    repos = data.get("repositories", []) if isinstance(data, dict) else []
    if not repos:
        print(f"[WARN] No repositories returned. Raw: {json.dumps(result, default=str)[:400]}")
        return 0

    print(f"[OK] Found {len(repos)} repositories:\n")
    for r in repos:
        print(f"  - {r.get('full_name')}  (stars={r.get('stargazers_count')})  "
              f"{'private' if r.get('private') else 'public'}  "
              f"updated {str(r.get('updated_at'))[:10]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())