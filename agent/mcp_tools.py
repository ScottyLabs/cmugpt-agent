"""MCP tool loading via langchain-mcp-adapters.

Each `BaseTool` from `MultiServerMCPClient.get_tools()` opens its own
streamable-HTTP session on invocation, so no session outlives a graph run.
The graph wraps tool results itself (see `agent/graph.py`) because tool
output is untrusted data.

Tools are grouped by service. The Surface switches groups off per user
(`filter_tools`) and the agent narrows each request to the groups the query
needs (`select_tools_for_query`) since every bound schema costs input tokens
on every model pass.
"""

import os
import re
from collections.abc import Iterable

from dotenv import load_dotenv
from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient

from .guards import CMU_DATA_RE

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

# Query signals per tool group. When nothing matches, every group stays
# bound, so a missed keyword can never cost the model a tool it needed.
_GROUP_HINT_RES: dict[str, re.Pattern[str]] = {
    "eats": re.compile(
        r"\b("
        r"din(?:e|es|ing)|food|eat(?:s|ing|ery|eries)?|hungry|breakfast|"
        r"lunch|dinner|brunch|coffee|cafes?|snacks?|menus?|cuisine|pizza|"
        r"restaurants?|meals?|drinks?|boba|dessert"
        r")\b",
        re.IGNORECASE,
    ),
    "courses": re.compile(
        r"\b("
        r"courses?|class(?:es)?|prereq\w*|requisites?|professors?|"
        r"instructors?|units?|gen(?:\s|-)?eds?|syllabus|semester|schedules?|"
        r"lectures?|recitations?|\d{2}-\d{3}"
        r")\b",
        re.IGNORECASE,
    ),
    "maps": re.compile(
        r"\b("
        r"where|maps?|directions?|routes?|paths?|walk\w*|navigat\w*|"
        r"buildings?|located|locations?|distance|near(?:by|est)?|closest|"
        r"get\s+to|go\s+to|from"
        r")\b",
        re.IGNORECASE,
    ),
    "guide": re.compile(
        r"\b("
        r"dorms?|housing|meal\s+plans?|leave\s+of\s+absence|transfer\w*|"
        r"majors?|minors?|accommodations?|advisors?|advising|polic(?:y|ies)|"
        r"guide|handbook|orientation|registration|enroll\w*|insurance|"
        r"shuttle|bus"
        r")\b",
        re.IGNORECASE,
    ),
}

# Args blocks stay because parameter conventions live only there. The
# trailing Returns prose and the markdown boilerplate add nothing the model
# needs.
_RETURNS_PARAGRAPH_RE = re.compile(r"\n\s*Returns[^\n]*(?:\n(?!\s*Args:)[^\n]*)*")
_MARKDOWN_BOILERPLATE_RE = re.compile(r"\s*formatted as clean markdown", re.IGNORECASE)


def condense_tool_description(description: str | None) -> str:
    if not description:
        return ""
    condensed = _RETURNS_PARAGRAPH_RE.sub("", description)
    condensed = _MARKDOWN_BOILERPLATE_RE.sub("", condensed)
    return re.sub(r"\n{3,}", "\n\n", condensed).strip()


def select_tools_for_query(
    tools: list[BaseTool],
    query: str,
    history_texts: Iterable[str] | None = None,
) -> list[BaseTool]:
    """Narrow the bound toolset to the groups the query plausibly needs.

    A query with no group signal falls back to recent user turns, so
    follow-ups keep the groups the conversation was using. A campus-shaped
    query with no group signal keeps every tool. Only when nothing anywhere
    looks like campus data does the fallback shrink to the guide group, the
    catch-all for student-life questions, so greetings stop paying for all
    23 schemas. Ungrouped tools are always kept.
    """
    matched = {
        group for group, hint in _GROUP_HINT_RES.items() if hint.search(query or "")
    }
    if not matched:
        for text in history_texts or []:
            matched.update(
                group
                for group, hint in _GROUP_HINT_RES.items()
                if isinstance(text, str) and hint.search(text)
            )
    if not matched:
        texts = [query or "", *[t for t in history_texts or [] if isinstance(t, str)]]
        if any(CMU_DATA_RE.search(text) for text in texts):
            return list(tools)
        matched = {"guide"}
    return [
        tool
        for tool in tools
        if (group := tool_group(tool.name or "")) is None or group in matched
    ]


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

    The model never learns the dropped tools exist. They are left out of the
    prompt catalog and out of `bind_tools`, so it has no schema to call them
    with. The tools node is built from the same filtered list, so a call
    smuggled in through conversation history resolves to "not available".
    """
    groups = normalize_disabled_groups(disabled)
    if not groups:
        return list(tools)
    return [tool for tool in tools if tool_group(tool.name or "") not in groups]


def _server_url() -> str:
    # Read at call time so dotenv order and runtime env changes are respected.
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
        tools = await client.get_tools()
    except Exception:
        # Continue without tools rather than failing the turn.
        return []
    # Condensing here reaches the prompt catalog and the bound schemas.
    for tool in tools:
        tool.description = condense_tool_description(tool.description)
    return tools
