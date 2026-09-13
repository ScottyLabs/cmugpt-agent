"""Quick live smoke test for the Bark agent via OpenRouter + MCP."""

import asyncio
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from cmugpt import run_agent
from cmugpt.mcp_tools import load_mcp_tools
from cmugpt.schema import UserInput

_REPO_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(_REPO_ROOT / ".env")

MCP_SERVER_URL = os.getenv("MCP_SERVER_URL", "")


async def test_agent():
    """Test the full agent pipeline (OpenRouter + MCP)."""
    query = "What dining options are open on CMU campus?"
    print(f"Query: {query}\n")

    user_input = UserInput(query=query)
    response = await run_agent(user_input)

    print(f"Action:     {response.action}")
    print(f"Confidence: {response.thought.confidence}")
    print(f"Reasoning:  {response.thought.reasoning}")
    print(f"\nResponse:\n{response.response_text}")


async def test_mcp():
    """Test MCP server connectivity, tool discovery, and tool calling."""
    print("=" * 50)
    print("MCP Server Test")
    print("=" * 50)

    print(f"\nConnecting to MCP server: {MCP_SERVER_URL}")
    tools = await load_mcp_tools()
    print(f"Found {len(tools)} tool(s):")
    for tool in tools:
        print(f"  - {tool.name}: {tool.description}")

    if tools:
        first_tool = tools[0]
        print(f"\nCalling tool '{first_tool.name}' with empty args...")
        result = await first_tool.ainvoke({})
        print(f"Result:\n{str(result)[:500]}")
    else:
        print("\nNo tools found - skipping tool call test.")

    print("\nMCP test complete.")


async def main():
    test = sys.argv[1] if len(sys.argv) > 1 else "all"

    if test in ("mcp", "all"):
        await test_mcp()
    if test in ("agent", "all"):
        print()
        await test_agent()


if __name__ == "__main__":
    asyncio.run(main())
