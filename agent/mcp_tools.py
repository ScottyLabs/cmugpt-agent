"""MCP tool loading via langchain-mcp-adapters.

Replaces the hand-rolled `mcp` client wiring. `MultiServerMCPClient.get_tools()`
returns self-contained LangChain `BaseTool` objects: each tool opens its own
streamable-HTTP session on invocation, so there is no long-lived session to
manage across a graph run.

The agent treats tool output as untrusted data, so the graph wraps results
itself (see `agent/graph.py`) rather than letting the prebuilt ToolNode pass
raw content straight to the model.
"""

import asyncio
import logging
import os
import time

from dotenv import load_dotenv
from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient

load_dotenv()

logger = logging.getLogger(__name__)

_SERVER_NAME = "cmu"

# Tool discovery is a network round-trip to the MCP server, so cache the result
# instead of paying it on every request. The returned tools are self-contained
# (each invocation opens its own session), which is what makes reuse safe.
# Failures are cached briefly too, so a down server isn't hammered but service
# recovers quickly.
_CACHE_TTL_SECONDS = 60.0
_FAILURE_TTL_SECONDS = 15.0

# (server url, monotonic expiry, tools)
_cache: tuple[str, float, list[BaseTool]] | None = None
_cache_lock = asyncio.Lock()


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
    """Discover tools from the configured MCP server, cached with a short TTL.

    Returns an empty list when no server is configured or the server is
    unreachable, so the agent degrades gracefully to tool-free answering.
    """
    global _cache
    url = _server_url()
    if not url:
        return []

    cached = _cache
    if cached and cached[0] == url and cached[1] > time.monotonic():
        return cached[2]

    async with _cache_lock:
        cached = _cache  # re-check: another request may have refreshed it
        if cached and cached[0] == url and cached[1] > time.monotonic():
            return cached[2]
        try:
            tools = await _build_client(url).get_tools()
            ttl = _CACHE_TTL_SECONDS
        except Exception:
            # MCP unavailable: continue without tools rather than fail the turn.
            logger.warning("MCP tool discovery failed for %s", url, exc_info=True)
            tools = []
            ttl = _FAILURE_TTL_SECONDS
        _cache = (url, time.monotonic() + ttl, tools)
        return tools
