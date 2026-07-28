"""MCP tool loading via langchain-mcp-adapters.

Replaces the hand-rolled `mcp` client wiring. `MultiServerMCPClient.get_tools()`
returns self-contained LangChain `BaseTool` objects: each tool opens its own
streamable-HTTP session on invocation, so there is no long-lived session to
manage across a graph run.

The agent treats tool output as untrusted data, so the graph wraps results
itself (see `agent/graph.py`) rather than letting the prebuilt ToolNode pass
raw content straight to the model.

Tools are grouped by the service they come from so the Surface can offer one
on/off switch per service. Disabling a group removes its tools before they ever
reach the model (see `filter_tools`).
"""

import os
from collections.abc import Iterable

from dotenv import load_dotenv
from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient

load_dotenv()

_SERVER_NAME = "cmu"

# Every tool the MCP server publishes is named `<group>_<action>` (for example
# `maps_get_path`, `eats_get_location_hours`), so a group is switched off by
# dropping each tool that carries its prefix.
TOOL_GROUP_LABELS: dict[str, str] = {
    "maps": "CMUMaps",
    "courses": "CMUCourses",
    "eats": "CMUEats",
    "guide": "CMU Guide",
}


def tool_group(tool_name: str) -> str | None:
    """Return the group `tool_name` belongs to, or None when it has no group."""
    for group in TOOL_GROUP_LABELS:
        if tool_name.startswith(f"{group}_"):
            return group
    return None


def normalize_disabled_groups(disabled: Iterable[str] | None) -> set[str]:
    """Coerce caller-supplied group ids into known group keys.

    Accepts the ids the Surface sends (`maps`) as well as the labels it shows
    (`CMUMaps`), and ignores anything unrecognized, so a malformed or renamed
    entry can never silently re-enable a group the user switched off.
    """
    if not disabled:
        return set()
    groups: set[str] = set()
    for raw in disabled:
        if not isinstance(raw, str):
            continue
        candidate = raw.strip().lower().removeprefix("cmu").strip("_- ")
        if candidate in TOOL_GROUP_LABELS:
            groups.add(candidate)
    return groups


def disabled_group_labels(disabled: Iterable[str] | None) -> list[str]:
    """Display labels for the disabled groups, in a stable order."""
    groups = normalize_disabled_groups(disabled)
    return [label for group, label in TOOL_GROUP_LABELS.items() if group in groups]


def filter_tools(
    tools: list[BaseTool],
    disabled: Iterable[str] | None,
) -> list[BaseTool]:
    """Drop every tool belonging to a disabled group.

    The model never learns the dropped tools exist: they are left out of the
    prompt catalog and out of `bind_tools`, so it has no schema to call them
    with. The tools node is built from the same filtered list, so a call
    smuggled in through conversation history resolves to "not available".
    """
    groups = normalize_disabled_groups(disabled)
    if not groups:
        return list(tools)
    return [tool for tool in tools if tool_group(tool.name or "") not in groups]


def _server_url() -> str:
    # Read at call time so import order (relative to dotenv loading) and any
    # runtime env changes are always respected.
    return os.getenv("MCP_SERVER_URL", "")


def _build_client(url: str) -> MultiServerMCPClient:
    return MultiServerMCPClient(
        {
            _SERVER_NAME: {
                "url": url,
                "transport": "streamable_http",
            }
        }
    )


async def load_mcp_tools() -> list[BaseTool]:
    """Discover tools from the configured MCP server.

    Returns an empty list when no server is configured or the server is
    unreachable, so the agent degrades gracefully to tool-free answering.
    """
    url = _server_url()
    if not url:
        return []
    try:
        client = _build_client(url)
        return await client.get_tools()
    except Exception:
        # MCP unavailable: continue without tools rather than failing the turn.
        return []
