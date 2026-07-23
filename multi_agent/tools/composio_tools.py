"""
tools/composio_tools.py — Composio Tool Handler.

Exports LangChain-compatible tools from the Composio toolset for the 6 connected services:
- GitHub
- Google Docs
- Tavily
- YouTube
- Context7 (MCP)
- Hugging Face
"""

import os

from composio import Composio
from composio_langchain import LangchainProvider
from multi_agent.config import COMPOSIO_API_KEY, COMPOSIO_USER_ID


TOOLKITS = [
    "GITHUB",
    "GOOGLEDOCS",
    "TAVILY",
    "YOUTUBE",
    "CONTEXT7_MCP",
    "HUGGING_FACE",
]


def _clean_schema(schema: dict) -> None:
    """Recursively strip out 'const' and 'additionalProperties' keys to prevent LangChain warnings."""
    if not isinstance(schema, dict):
        return
    schema.pop("const", None)
    schema.pop("additionalProperties", None)
    for v in schema.values():
        if isinstance(v, dict):
            _clean_schema(v)
        elif isinstance(v, list):
            for item in v:
                if isinstance(item, dict):
                    _clean_schema(item)


def get_connected_composio_apps() -> list[dict]:
    """Retrieve currently active connected apps from Composio API."""
    if not COMPOSIO_API_KEY:
        return []
    try:
        composio = Composio(api_key=COMPOSIO_API_KEY, dangerously_skip_version_check=True)
        accs = composio.connected_accounts.list()
        connected = []
        seen = set()
        for item in getattr(accs, "items", []):
            if getattr(item, "status", "").upper() == "ACTIVE":
                slug = getattr(getattr(item, "toolkit", None), "slug", "").lower()
                if slug and slug not in seen:
                    seen.add(slug)
                    connected.append({
                        "id": getattr(item, "id", ""),
                        "slug": slug,
                        "name": slug.replace("_", " ").title(),
                        "user_id": getattr(item, "user_id", ""),
                        "updated_at": getattr(item, "updated_at", ""),
                    })
        return connected
    except Exception as e:
        print(f"[COMPOSIO] Error fetching connected apps: {e}")
        return []


def get_composio_tools():
    """
    Retrieve all LangChain-compatible tools for the 6 connected Composio apps.

    Returns:
        List of LangChain tools ready for use with LangChain agents.
    """
    if not COMPOSIO_API_KEY:
        print("[COMPOSIO] Warning: COMPOSIO_API_KEY not set, returning empty tool list")
        return []

    try:
        composio = Composio(
            api_key=COMPOSIO_API_KEY,
            dangerously_skip_version_check=True,
        )
        provider = LangchainProvider()

        version_map: dict[str, str] = {}
        all_raw_tools = []
        for toolkit in TOOLKITS:
            try:
                raw_tools = composio.tools.get_raw_composio_tools(toolkits=[toolkit], limit=150)
                all_raw_tools.extend(raw_tools)
                for tool in raw_tools:
                    if hasattr(tool, "toolkit"):
                        tk_slug = getattr(tool.toolkit, "slug", None)
                        if tk_slug and hasattr(tool, "version") and tk_slug not in version_map:
                            version_map[tk_slug.lower()] = tool.version
            except Exception as tk_err:
                print(f"[COMPOSIO] Error loading toolkit '{toolkit}': {tk_err}")

        for tk_slug, version in version_map.items():
            env_var = f"COMPOSIO_TOOLKIT_VERSION_{tk_slug.upper()}"
            os.environ[env_var] = version

        for tool in all_raw_tools:
            if not tool.input_parameters.get("title"):
                tool.input_parameters["title"] = tool.slug
            _clean_schema(tool.input_parameters)
            properties = tool.input_parameters.get("properties", {})
            if isinstance(properties, dict):
                for prop_name, prop_schema in properties.items():
                    if isinstance(prop_schema, dict):
                        for combiner in ["oneOf", "anyOf"]:
                            if combiner in prop_schema and isinstance(prop_schema[combiner], list):
                                if len(prop_schema[combiner]) > 3:
                                    prop_schema[combiner] = prop_schema[combiner][:3]

        user_id = COMPOSIO_USER_ID or "pg-test-7ea14b6c-9649-420f-b5cf-fcfbdf2e9a17"

        def _execute(slug: str, arguments):
            prefix = slug.split("_", 1)[0].lower() if "_" in slug else ""
            ver = None
            for tk_slug, v in version_map.items():
                if tk_slug.startswith(prefix) or tk_slug == prefix:
                    ver = v
                    break
            kwargs = dict(user_id=user_id)
            if ver:
                kwargs["version"] = ver
            return composio.tools.execute(slug, arguments, **kwargs)

        tools = provider.wrap_tools(all_raw_tools, execute_tool=_execute)
        return tools
    except Exception as e:
        print(f"[COMPOSIO] Failed to initialize tools: {e}")
        return []