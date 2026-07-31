"""MCP tool loading via langchain-mcp-adapters.

Replaces the hand-rolled `mcp` client wiring. `MultiServerMCPClient.get_tools()`
returns self-contained LangChain `BaseTool` objects: each tool opens its own
streamable-HTTP session on invocation, so there is no long-lived session to
manage across a graph run.

The agent treats tool output as untrusted data, so the graph wraps results
itself (see `agent/graph.py`) rather than letting the prebuilt ToolNode pass
raw content straight to the model.
"""

import os

from dotenv import load_dotenv
from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient

load_dotenv()

_SERVER_NAME = "cmu"


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
