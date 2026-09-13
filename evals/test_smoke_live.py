"""Smoke checks against the live services.

Confirms the MCP server publishes tools and that one real turn completes
using a data tool. Run with `uv run pytest evals`. Skipped unless
OPENROUTER_API_KEY and MCP_SERVER_URL are set.
"""

import pytest

from cmugpt import run_agent
from cmugpt.mcp_tools import load_mcp_tools
from cmugpt.schema import UserInput
from cmugpt.settings import get_settings

pytestmark = pytest.mark.live


async def test_mcp_server_publishes_tools() -> None:
    tools = await load_mcp_tools()
    assert tools, f"no tools discovered at {get_settings().mcp_server_url}"
    names = [tool.name for tool in tools]
    print(f"{len(names)} tools: {', '.join(names)}")
    assert any(
        name.startswith(("maps_", "eats_", "courses_", "guide_")) for name in names
    )


async def test_one_turn_reaches_a_data_tool() -> None:
    response = await run_agent(
        UserInput(query="What dining options are open on CMU campus?")
    )
    print(response.response_text)
    assert response.response_text.strip()
    assert response.services_used, "a dining question should reach the eats tools"
