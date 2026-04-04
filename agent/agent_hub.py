import json
import os
import re
from itertools import count
from typing import Any

import httpx
from dotenv import load_dotenv
from openai import AsyncOpenAI

from .schema import ActionType, AgentResponse, Thought, ToolCall, UserInput

load_dotenv()

MCP_SERVER_URL = os.getenv("MCP_SERVER_URL", "").strip()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "").strip()
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


class MCPHttpClient:
    """Minimal MCP JSON-RPC client over HTTP transport."""

    def __init__(self, endpoint: str) -> None:
        self._endpoint = endpoint.rstrip("/")
        self._id_counter = count(1)

    async def _call(self, method: str, params: dict[str, Any] | None = None) -> Any:
        payload = {
            "jsonrpc": "2.0",
            "id": next(self._id_counter),
            "method": method,
            "params": params or {},
        }
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(self._endpoint, json=payload)
            response.raise_for_status()
            data = response.json()

        if "error" in data:
            message = data["error"].get("message", "Unknown MCP error")
            raise RuntimeError(f"MCP error on {method}: {message}")

        return data.get("result")

    async def list_tools(self) -> list[dict[str, Any]]:
        result = await self._call("tools/list")
        tools = result.get("tools", []) if isinstance(result, dict) else []
        return [tool for tool in tools if isinstance(tool, dict)]

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        return await self._call("tools/call", {"name": name, "arguments": arguments})


def _safe_openai_tool_name(raw_name: str, existing: set[str]) -> str:
    safe_name = re.sub(r"[^a-zA-Z0-9_-]", "_", raw_name)
    safe_name = safe_name.strip("_") or "mcp_tool"
    safe_name = safe_name[:64]

    candidate = safe_name
    suffix = 1
    while candidate in existing:
        suffix_str = f"_{suffix}"
        candidate = f"{safe_name[: max(1, 64 - len(suffix_str))]}{suffix_str}"
        suffix += 1
    return candidate


def _coerce_content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                text_parts.append(str(item.get("text", "")))
        return "\n".join(part for part in text_parts if part)
    return ""


def _serialize_tool_result(
    result: Any,
) -> str | int | float | bool | list[str | int | float | bool] | None:
    if result is None:
        return None
    if isinstance(result, str | int | float | bool):
        return result
    if isinstance(result, list) and all(
        isinstance(x, str | int | float | bool) for x in result
    ):
        return result
    return json.dumps(result)


async def run_agent(
    user_input: UserInput,
    mcp_servers: list[str] | None = None,
    model: str = "openai/gpt-4o",
    message_history: list[dict[str, str]] | None = None,
) -> AgentResponse:
    """
    Runs the CMUGPT agent with structured input/output using MCP API calls.

    Args:
        user_input: UserInput object containing the query and optional context.
        mcp_servers: Deprecated; kept for backward compatibility.
        model: The model name to use via OpenRouter-compatible Chat Completions API.
        message_history: Optional conversation history for multi-turn interactions.

    Returns:
        AgentResponse: Structured response with thought, action, and response_text.
    """
    _ = mcp_servers

    if not MCP_SERVER_URL:
        raise ValueError("MCP_SERVER_URL is not set. Add it to your .env file.")

    if not OPENROUTER_API_KEY:
        raise ValueError("OPENROUTER_API_KEY is not set. Add it to your .env file.")

    openai_client = AsyncOpenAI(
        api_key=OPENROUTER_API_KEY,
        base_url=OPENROUTER_BASE_URL,
    )

    mcp_client = MCPHttpClient(MCP_SERVER_URL)
    mcp_tools = await mcp_client.list_tools()

    openai_tools: list[dict[str, Any]] = []
    openai_name_to_mcp_name: dict[str, str] = {}
    used_names: set[str] = set()
    for tool in mcp_tools:
        raw_name = str(tool.get("name", ""))
        if not raw_name:
            continue
        openai_name = _safe_openai_tool_name(raw_name, used_names)
        used_names.add(openai_name)
        openai_name_to_mcp_name[openai_name] = raw_name
        openai_tools.append(
            {
                "type": "function",
                "function": {
                    "name": openai_name,
                    "description": str(tool.get("description", "MCP tool")),
                    "parameters": tool.get(
                        "inputSchema", {"type": "object", "properties": {}}
                    ),
                },
            }
        )

    messages: list[dict[str, Any]] = [
        {
            "role": "system",
            "content": (
                "You are the CMUGPT Agent. Assist users with CMU campus information. "
                "Use provided tools when needed and keep answers concise and accurate."
            ),
        }
    ]
    for msg in message_history or []:
        role = msg.get("role")
        content = msg.get("content")
        if isinstance(role, str) and isinstance(content, str):
            messages.append({"role": role, "content": content})

    if user_input.context:
        messages.append(
            {
                "role": "system",
                "content": f"Context: {json.dumps(user_input.context)}",
            }
        )
    messages.append({"role": "user", "content": user_input.query})

    tool_calls_made: list[ToolCall] = []
    final_text = ""

    for _ in range(6):
        completion = await openai_client.chat.completions.create(
            model=model,
            messages=messages,
            tools=openai_tools,
            tool_choice="auto",
        )
        response_message = completion.choices[0].message
        tool_calls = response_message.tool_calls or []

        if not tool_calls:
            final_text = _coerce_content_to_text(response_message.content).strip()
            break

        assistant_message: dict[str, Any] = {
            "role": "assistant",
            "content": response_message.content or "",
            "tool_calls": [tc.model_dump() for tc in tool_calls],
        }
        messages.append(assistant_message)

        for tool_call in tool_calls:
            openai_tool_name = tool_call.function.name
            mcp_tool_name = openai_name_to_mcp_name.get(openai_tool_name)
            if mcp_tool_name is None:
                tool_output: Any = {"error": f"Unknown tool: {openai_tool_name}"}
            else:
                raw_args = tool_call.function.arguments or "{}"
                try:
                    parsed_args = json.loads(raw_args)
                except json.JSONDecodeError:
                    parsed_args = {}
                if not isinstance(parsed_args, dict):
                    parsed_args = {}
                try:
                    tool_output = await mcp_client.call_tool(mcp_tool_name, parsed_args)
                except Exception as exc:  # noqa: BLE001
                    tool_output = {"error": str(exc)}

            tool_calls_made.append(
                ToolCall(
                    tool_name=mcp_tool_name or openai_tool_name,
                    parameters=None,
                    result=_serialize_tool_result(tool_output),
                )
            )
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(tool_output),
                }
            )

    if not final_text:
        final_text = "I could not produce a final answer from the MCP tool outputs."

    action = ActionType.RESPOND if not tool_calls_made else ActionType.RETRIEVE
    confidence = 0.7 if tool_calls_made else 0.5

    return AgentResponse(
        thought=Thought(
            reasoning=(
                "Used MCP API tools over HTTP to gather CMU data and compose "
                "a response."
            ),
            confidence=confidence,
        ),
        action=action,
        tool_calls=tool_calls_made,
        response_text=final_text,
    )
