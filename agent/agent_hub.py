import json
import os
from collections.abc import Callable
from contextlib import suppress
from typing import Any

from dotenv import load_dotenv
from mcp.client.session import ClientSession
from mcp.client.streamable_http import streamable_http_client
from mcp.types import TextContent
from openai import AsyncOpenAI

from .schema import (
    ActionType,
    AgentResponse,
    Metadata,
    Thought,
    UserInput,
)

load_dotenv()

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
MCP_SERVER_URL = os.getenv("MCP_SERVER_URL", "")

client = AsyncOpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY", ""),
    base_url=OPENROUTER_BASE_URL,
)


def _parse_agent_response(raw: str) -> AgentResponse:
    """Parse model output into AgentResponse, with a safe fallback."""
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[1]
        cleaned = cleaned.rsplit("```", 1)[0].strip()
    with suppress(json.JSONDecodeError, TypeError, ValueError):
        parsed = json.loads(cleaned)
        return AgentResponse(**parsed)
    return _fallback_response(raw)


async def _run_completion_loop(
    messages: list[dict[str, Any]],
    *,
    model: str,
    openai_tools: list[dict] | None = None,
    call_tool: Callable[[str, dict], Any] | None = None,
) -> AgentResponse:
    """Run the LLM loop and optionally support tool-calling."""
    for _ in range(10):
        chat_kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
        }
        if openai_tools:
            chat_kwargs["tools"] = openai_tools

        response = await client.chat.completions.create(**chat_kwargs)
        choice = response.choices[0]

        if (
            openai_tools
            and call_tool is not None
            and choice.finish_reason == "tool_calls"
            and choice.message.tool_calls
        ):
            messages.append(choice.message.model_dump())

            for tool_call in choice.message.tool_calls:
                fn = tool_call.function
                args = json.loads(fn.arguments) if fn.arguments else {}
                result = await call_tool(fn.name, args)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": result,
                    }
                )
            continue

        return _parse_agent_response(choice.message.content or "")

    return _fallback_response(
        "Unable to complete the request within allowed steps.",
        confidence=0.3,
    )


async def _get_mcp_tools(
    session: ClientSession,
) -> list[dict]:
    """Discover tools from the MCP server, convert to OpenAI format."""
    tools_result = await session.list_tools()
    openai_tools = []
    for tool in tools_result.tools:
        schema = tool.inputSchema or {
            "type": "object",
            "properties": {},
        }
        openai_tools.append(
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description or "",
                    "parameters": schema,
                },
            }
        )
    return openai_tools


async def _call_mcp_tool(
    session: ClientSession,
    name: str,
    arguments: dict,
) -> str:
    """Call a tool on the MCP server and return the result."""
    result = await session.call_tool(name, arguments)
    parts = []
    for content in result.content:
        if isinstance(content, TextContent):
            parts.append(content.text)
        else:
            parts.append(str(content))
    return "\n".join(parts)


def _fallback_response(text: str, confidence: float = 0.8) -> AgentResponse:
    """Build an AgentResponse for non-JSON or fallback cases."""
    return AgentResponse(
        thought=Thought(
            reasoning="Direct response",
            confidence=confidence,
        ),
        action=ActionType.RESPOND,
        tool_calls=[],
        response_text=text,
        metadata=Metadata(),
    )


async def run_agent(
    user_input: UserInput,
    model: str = "openai/gpt-4o",
    message_history: list[dict[str, str]] | None = None,
) -> AgentResponse:
    """
    Runs the agent using OpenRouter for LLM and Railway MCP for tools.

    Args:
        user_input: UserInput object containing the query.
        model: The model to use via OpenRouter.
        message_history: Optional conversation history.

    Returns:
        AgentResponse with thought, action, and response_text.
    """
    system_message = {
        "role": "system",
        "content": (
            "You are the CMUGPT Agent. Assist users with "
            "CMU campus information. "
            "Use provided tools to query data when relevant. "
            "When you have a final answer, respond with valid "
            "JSON matching this schema:\n"
            "{\n"
            '  "thought": {"reasoning": "...", '
            '"confidence": 0.0-1.0},\n'
            '  "action": "query|retrieve|search|compute|respond"'
            ",\n"
            '  "tool_calls": [],\n'
            '  "response_text": "your answer",\n'
            '  "metadata": {}\n'
            "}\n"
            "Respond with ONLY the JSON object."
        ),
    }

    messages: list[dict] = [system_message]
    if message_history:
        messages.extend(message_history)
    messages.append({"role": "user", "content": user_input.query})

    if not os.getenv("OPENROUTER_API_KEY"):
        return _fallback_response(
            "OPENROUTER_API_KEY is not configured.",
            confidence=0.2,
        )

    if MCP_SERVER_URL:
        try:
            async with (
                streamable_http_client(MCP_SERVER_URL) as (
                    read_stream,
                    write_stream,
                    _,
                ),
                ClientSession(read_stream, write_stream) as session,
            ):
                await session.initialize()
                openai_tools = await _get_mcp_tools(session)
                return await _run_completion_loop(
                    messages,
                    model=model,
                    openai_tools=openai_tools,
                    call_tool=lambda name, arguments: _call_mcp_tool(
                        session, name, arguments
                    ),
                )
        except Exception:
            # If MCP is unavailable, continue without tool-calling.
            pass

    return await _run_completion_loop(messages, model=model)
