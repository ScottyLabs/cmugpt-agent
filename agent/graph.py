"""LangGraph implementation of the CMUGPT agent.

A single compiled `StateGraph` is the one source of truth for both the
non-streaming (`/agent/respond`) and streaming (`/agent/respond/stream`) HTTP
endpoints. The model emits plain Markdown; deterministic nodes compute
`cmu_maps`, `services_used`, and `thought` into graph state.

Graph shape: ``START -> agent``; from ``agent`` either ``-> tools -> agent``
(when the model requested tool calls) or ``-> postprocess -> END`` (final
answer).

Streaming is done with LangGraph's custom stream channel: nodes emit typed
events through the injected `writer`, and the public entrypoints forward them as
``(event_name, data)`` tuples matching the existing SSE contract
(``status`` / ``map`` / ``delta`` / ``done`` / ``error``). When the graph is run
non-streaming via ``ainvoke`` the writes are simply dropped.
"""

from __future__ import annotations

import operator
import os
from collections.abc import AsyncIterator
from typing import Annotated, Any, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    AnyMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import StreamWriter
from pydantic import SecretStr

from .cmu_maps import _apply_cmu_maps_guard, query_has_map_intent
from .guards import (
    apply_tool_transparency_guard,
    compute_thought,
    should_require_tool,
)
from .mcp_tools import load_mcp_tools
from .prompts import build_system_prompt
from .schema import ActionType, AgentResponse, CmuMaps, Metadata, Thought, UserInput

load_dotenv()

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

StreamEvent = tuple[str, dict[str, Any]]


class AgentState(TypedDict):
    """Shared state threaded through the graph."""

    messages: Annotated[list[AnyMessage], add_messages]
    query: str
    tool_invocations: Annotated[list[dict[str, Any]], operator.add]
    services_used: Annotated[list[str], operator.add]
    response_text: str
    streamed: bool
    response_payload: dict[str, Any]


def _api_key() -> str:
    return os.getenv("OPENROUTER_API_KEY", "")


def _make_chat_model(model: str) -> ChatOpenAI:
    return ChatOpenAI(
        model_name=model,
        openai_api_key=SecretStr(_api_key()),
        openai_api_base=OPENROUTER_BASE_URL,
    )


def _message_text(message: AnyMessage | AIMessageChunk | None) -> str:
    if message is None:
        return ""
    content = message.content
    if isinstance(content, str):
        return content
    # Some providers return content as a list of parts.
    parts: list[str] = []
    for part in content:
        if isinstance(part, str):
            parts.append(part)
        elif isinstance(part, dict) and isinstance(part.get("text"), str):
            parts.append(part["text"])
    return "".join(parts)


def _helper_messages(query: str) -> list[dict[str, Any]]:
    """Minimal role/content list for the deterministic helpers."""
    return [{"role": "user", "content": query}]


def _fallback_response(text: str, confidence: float = 0.8) -> AgentResponse:
    return AgentResponse(
        thought=Thought(reasoning="Direct response", confidence=confidence),
        action=ActionType.RESPOND,
        tool_calls=[],
        response_text=text,
        metadata=Metadata(),
    )


def _build_agent_node(model: ChatOpenAI, tools: list[BaseTool]):
    bound = model.bind_tools(tools) if tools else model
    bound_required = model.bind_tools(tools, tool_choice="required") if tools else model

    async def agent_node(state: AgentState, writer: StreamWriter) -> dict[str, Any]:
        query = state["query"]
        force_tool = (
            bool(tools)
            and not state["services_used"]
            and should_require_tool(_helper_messages(query))
        )
        runnable = bound_required if force_tool else bound

        # Buffer (don't live-stream) these passes so postprocess can repair the
        # text before the user sees it:
        #   * forced tool passes: prose here (e.g. an "I couldn't find a route"
        #     preamble) is not the final answer.
        #   * map queries: the model sometimes falsely claims it couldn't look
        #     up locations even though we have a working map; we strip that out
        #     in postprocess, so it must not stream live.
        suppress_stream = force_tool or query_has_map_intent(query)

        gathered: AIMessageChunk | None = None
        saw_tool_call = False
        streamed_any = False
        async for chunk in runnable.astream(state["messages"]):
            if not isinstance(chunk, AIMessageChunk):
                continue
            gathered = chunk if gathered is None else gathered + chunk
            if chunk.tool_call_chunks:
                saw_tool_call = True
            text = _message_text(chunk)
            if text and not saw_tool_call and not suppress_stream:
                writer({"event": "delta", "data": {"text": text}})
                streamed_any = True

        if gathered is None:
            gathered = AIMessageChunk(content="")

        final_message = AIMessage(
            content=gathered.content,
            tool_calls=gathered.tool_calls,
        )

        if gathered.tool_calls:
            writer({"event": "status", "data": {"text": "Checking CMU tools..."}})
            return {"messages": [final_message]}

        return {
            "messages": [final_message],
            "response_text": _message_text(gathered),
            "streamed": streamed_any,
        }

    return agent_node


def _build_tools_node(tools: list[BaseTool]):
    tools_by_name = {tool.name: tool for tool in tools}

    async def tools_node(state: AgentState, writer: StreamWriter) -> dict[str, Any]:
        last = state["messages"][-1]
        tool_calls = last.tool_calls if isinstance(last, AIMessage) else []

        new_messages: list[AnyMessage] = []
        new_invocations: list[dict[str, Any]] = []
        new_services: list[str] = []

        for call in tool_calls:
            name = call["name"]
            args = call.get("args") or {}
            call_id = call.get("id") or f"call_{name}"
            tool = tools_by_name.get(name)
            if tool is None:
                result = f"Tool '{name}' is not available."
            else:
                try:
                    raw = await tool.ainvoke(args)
                    result = raw if isinstance(raw, str) else str(raw)
                except Exception as exc:  # noqa: BLE001 - surface as tool data
                    result = f"Tool '{name}' failed: {exc}"

            new_invocations.append({"name": name, "arguments": args, "result": result})
            if name not in state["services_used"] and name not in new_services:
                new_services.append(name)

            # Wrap tool output so the model treats it as untrusted DATA, not as
            # instructions. Defense against prompt-injection from MCP content.
            wrapped = (
                f'<<<TOOL_OUTPUT name="{name}" trust="untrusted-data">>>\n'
                f"{result}\n"
                "<<<END_TOOL_OUTPUT>>>"
            )
            new_messages.append(ToolMessage(content=wrapped, tool_call_id=call_id))

        writer({"event": "status", "data": {"text": "Writing answer..."}})
        return {
            "messages": new_messages,
            "tool_invocations": new_invocations,
            "services_used": new_services,
        }

    return tools_node


async def _postprocess_node(state: AgentState, writer: StreamWriter) -> dict[str, Any]:
    query = state["query"]
    msgs = _helper_messages(query)
    invocations = state["tool_invocations"]
    services = state["services_used"]

    text = (state.get("response_text") or "").strip()
    if not text:
        text = (
            "I'm sorry, I couldn't generate a response for that. "
            "Please try rephrasing your question."
        )

    parsed = AgentResponse(
        thought=Thought(reasoning="Direct response", confidence=0.5),
        action=ActionType.RESPOND,
        tool_calls=[],
        response_text=text,
        services_used=list(services),
        cmu_maps=CmuMaps(),
        metadata=Metadata(),
    )

    parsed = _apply_cmu_maps_guard(parsed, msgs, invocations)
    parsed = apply_tool_transparency_guard(parsed, msgs, services)
    parsed.thought = compute_thought(services, invocations, parsed.response_text)
    parsed.action = ActionType.RETRIEVE if services else ActionType.RESPOND

    # When the answer was buffered (forced tool pass or a map query), it hasn't
    # been streamed yet — emit the repaired text now so the user only ever sees
    # the corrected version.
    if not state.get("streamed") and parsed.response_text:
        writer({"event": "delta", "data": {"text": parsed.response_text}})

    if parsed.cmu_maps.url:
        writer({"event": "map", "data": parsed.cmu_maps.model_dump()})

    payload = parsed.model_dump()
    writer({"event": "done", "data": payload})
    return {"response_payload": payload, "response_text": parsed.response_text}


def _route_after_agent(state: AgentState) -> str:
    last = state["messages"][-1]
    if isinstance(last, AIMessage) and last.tool_calls:
        return "tools"
    return "postprocess"


def build_graph(model: ChatOpenAI, tools: list[BaseTool]):
    """Compile the agent graph for one request (model + tools captured)."""
    graph = StateGraph(AgentState)
    graph.add_node("agent", _build_agent_node(model, tools))
    graph.add_node("tools", _build_tools_node(tools))
    graph.add_node("postprocess", _postprocess_node)

    graph.add_edge(START, "agent")
    graph.add_conditional_edges(
        "agent",
        _route_after_agent,
        {"tools": "tools", "postprocess": "postprocess"},
    )
    graph.add_edge("tools", "agent")
    graph.add_edge("postprocess", END)
    return graph.compile()


def _sanitize_history(
    message_history: list[dict[str, str]] | None,
) -> list[AnyMessage]:
    """Convert caller history to LangChain messages, dropping non user/assistant.

    We own the system prompt; smuggled `system`/`tool` turns are an injection
    vector, so only `user` and `assistant` turns are carried over.
    """
    if not message_history:
        return []
    out: list[AnyMessage] = []
    for turn in message_history:
        role = turn.get("role")
        content = turn.get("content")
        if not isinstance(content, str):
            continue
        if role == "user":
            out.append(HumanMessage(content=content))
        elif role == "assistant":
            out.append(AIMessage(content=content))
    return out


def _initial_state(
    user_input: UserInput,
    message_history: list[dict[str, str]] | None,
    tools: list[BaseTool],
) -> AgentState:
    messages: list[AnyMessage] = [SystemMessage(content=build_system_prompt(tools))]
    messages.extend(_sanitize_history(message_history))
    messages.append(HumanMessage(content=user_input.query))
    return AgentState(
        messages=messages,
        query=user_input.query,
        tool_invocations=[],
        services_used=[],
        response_text="",
        streamed=False,
        response_payload={},
    )


async def run_agent(
    user_input: UserInput,
    model: str = "openai/gpt-4o",
    message_history: list[dict[str, str]] | None = None,
) -> AgentResponse:
    """Non-streaming entry point. Runs the graph and returns the full response."""
    if not _api_key():
        return _fallback_response(
            "OPENROUTER_API_KEY is not configured.",
            confidence=0.2,
        )

    tools = await load_mcp_tools()
    graph = build_graph(_make_chat_model(model), tools)
    final = await graph.ainvoke(_initial_state(user_input, message_history, tools))

    payload = final.get("response_payload")
    if isinstance(payload, dict) and payload:
        return AgentResponse(**payload)
    return _fallback_response(
        "Unable to complete the request.",
        confidence=0.3,
    )


async def stream_agent_response(
    *,
    user_input: UserInput,
    model: str,
    message_history: list[dict[str, str]] | None,
) -> AsyncIterator[StreamEvent]:
    """Streaming entry point: yields ('delta', ...) ... ('done', ...) events."""
    if not _api_key():
        fb = _fallback_response(
            "OPENROUTER_API_KEY is not configured.",
            confidence=0.2,
        )
        yield ("delta", {"text": fb.response_text})
        yield ("done", fb.model_dump())
        return

    tools = await load_mcp_tools()
    graph = build_graph(_make_chat_model(model), tools)

    async for chunk in graph.astream(
        _initial_state(user_input, message_history, tools),
        stream_mode="custom",
    ):
        if isinstance(chunk, dict) and "event" in chunk:
            yield (chunk["event"], chunk.get("data", {}))
