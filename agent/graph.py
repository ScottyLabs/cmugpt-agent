"""LangGraph implementation of the CMUGPT agent.

A single compiled `StateGraph` backs both the non-streaming (`/agent/respond`)
and streaming (`/agent/respond/stream`) HTTP endpoints. The model emits plain
Markdown while deterministic nodes compute `cmu_maps`, `services_used`, and
`thought` into graph state.

The graph runs ``START -> agent``, then either ``agent -> tools -> agent``
when the model requested tool calls or ``agent -> postprocess -> END`` for a
final answer.

Streaming uses LangGraph's custom stream channel. Nodes emit typed events
through the injected `writer` and the public entrypoints forward them as
``(event_name, data)`` tuples matching the existing SSE contract
(``status`` / ``map`` / ``delta`` / ``done`` / ``error``). A non-streaming
``ainvoke`` run simply drops the writes.
"""

from __future__ import annotations

import logging
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
from langchain_core.tools import BaseTool, ToolException
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import StreamWriter
from pydantic import SecretStr

from .cmu_maps import _apply_cmu_maps_guard, query_has_map_intent
from .guards import (
    REFUSAL_TEXT,
    StreamScrubber,
    apply_output_guard,
    apply_tool_transparency_guard,
    canned_refusal_response,
    compute_thought,
    is_flagrant_injection,
    should_require_tool,
)
from .mcp_tools import (
    filter_tools,
    load_mcp_tools,
    normalize_disabled_groups,
    select_tools_for_query,
)
from .prompts import build_system_prompt
from .schema import ActionType, AgentResponse, CmuMaps, Metadata, Thought, UserInput
from .token_limits import record_usage

load_dotenv()

_LOG = logging.getLogger("cmugpt.agent")

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

StreamEvent = tuple[str, dict[str, Any]]


# Safety-net caps, not tuning knobs. Values are generous enough that normal
# conversations never hit them and they only engage on runaway input.

# History is billed on every model pass. Sixty messages is thirty exchanges.
_HISTORY_MAX_MESSAGES = 60
_HISTORY_MAX_MESSAGE_CHARS = 12_000

# User turns scanned for tool-group narrowing. Local regex only, no tokens.
_HISTORY_HINT_TURNS = 20

# Tool results are resent on every later pass. Twelve thousand chars fits
# every current CMU tool result, including the 9k full dining list, so this
# only engages if a tool starts returning something enormous. The marker
# keeps the model from presenting a truncated list as complete.
_TOOL_RESULT_MAX_CHARS = 12_000
_TOOL_RESULT_TRUNCATION_MARKER = (
    "\n[Result truncated. More entries exist beyond this point.]"
)


class AgentState(TypedDict):
    """Shared state threaded through the graph."""

    messages: Annotated[list[AnyMessage], add_messages]
    query: str
    tool_invocations: Annotated[list[dict[str, Any]], operator.add]
    services_used: Annotated[list[str], operator.add]
    response_text: str
    streamed: bool
    response_payload: dict[str, Any]
    # Tool groups the user switched off in the Surface. Their tools are
    # already unbound. Postprocess reads this to keep the map embed off too.
    disabled_tools: list[str]
    # Owner of the daily token budget for this run.
    user_id: str
    # Completed tool rounds. Drives the unbound late passes.
    tool_rounds: Annotated[int, operator.add]
    # Sticky across passes so a later clean pass cannot clear a trip.
    leak_detected: Annotated[bool, operator.or_]


def _api_key() -> str:
    return os.getenv("OPENROUTER_API_KEY", "")


def _make_chat_model(model: str) -> ChatOpenAI:
    return ChatOpenAI(
        model=model,
        api_key=SecretStr(_api_key()),
        base_url=OPENROUTER_BASE_URL,
        # Report usage on the final stream chunk so the budget counts real
        # usage instead of estimates.
        stream_usage=True,
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


def _record_pass_usage(state: AgentState, gathered: AIMessageChunk) -> None:
    """Charge one model pass to the user's daily budget and log it.

    Falls back to a chars/4 estimate so the budget stays enforceable when
    the stream carries no usage metadata. The log line is what makes caps
    and thresholds tunable from production data instead of guesses.
    """
    usage = getattr(gathered, "usage_metadata", None) or {}
    estimated = not usage.get("total_tokens")
    if estimated:
        input_chars = sum(len(_message_text(m)) for m in state["messages"])
        total = (input_chars + len(_message_text(gathered))) // 4
    else:
        total = int(usage["total_tokens"])
    _LOG.info(
        "pass_usage user=%s round=%s input=%s output=%s total=%s estimated=%s",
        (state.get("user_id") or "anonymous")[:8],
        state.get("tool_rounds", 0),
        usage.get("input_tokens"),
        usage.get("output_tokens"),
        total,
        estimated,
    )
    try:
        record_usage(state.get("user_id"), total)
    except Exception:
        # Never break an answer in flight, but a silent failure here would
        # disable the budget, so it must be visible.
        _LOG.exception("token budget recording failed")


def _truncate_tool_result(result: str) -> str:
    if len(result) <= _TOOL_RESULT_MAX_CHARS:
        return result
    return result[:_TOOL_RESULT_MAX_CHARS] + _TOOL_RESULT_TRUNCATION_MARKER


def _build_agent_node(model: ChatOpenAI, tools: list[BaseTool], maps_enabled: bool):
    bound = model.bind_tools(tools) if tools else model
    bound_required = model.bind_tools(tools, tool_choice="required") if tools else model

    async def agent_node(state: AgentState, writer: StreamWriter) -> dict[str, Any]:
        query = state["query"]
        # Forcing a tool call while a group is switched off can pick an
        # unrelated tool and misreport sources, so only force when nothing
        # is disabled.
        force_tool = (
            bool(tools)
            and not normalize_disabled_groups(state.get("disabled_tools"))
            and not state["services_used"]
            and should_require_tool(_helper_messages(query))
        )
        runnable = bound_required if force_tool else bound

        # Buffer passes postprocess may rewrite. Forced-pass preamble is not
        # the final answer, and map queries can draw a false failure claim
        # that postprocess strips before the user sees it.
        suppress_stream = force_tool or (maps_enabled and query_has_map_intent(query))

        # Live deltas cannot be retracted, so they lag behind the scrubber's
        # holdback. Buffered passes are scanned in postprocess instead.
        scrubber: StreamScrubber | None = None
        if not suppress_stream:
            prompt_text = (
                _message_text(state["messages"][0]) if state["messages"] else ""
            )
            scrubber = StreamScrubber(prompt_text)
        withheld_notice_sent = False

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
            if text and not saw_tool_call and scrubber is not None:
                safe = scrubber.push(text)
                if safe:
                    writer({"event": "delta", "data": {"text": safe}})
                    streamed_any = True
                elif scrubber.tripped and streamed_any and not withheld_notice_sent:
                    writer(
                        {"event": "delta", "data": {"text": "\n\n[Response withheld.]"}}
                    )
                    withheld_notice_sent = True

        if gathered is None:
            gathered = AIMessageChunk(content="")

        # Flush the held tail on every exit path, otherwise the last chars
        # of a preamble before a tool call would silently vanish.
        if scrubber is not None:
            tail = scrubber.flush()
            if tail:
                writer({"event": "delta", "data": {"text": tail}})
                streamed_any = True
            if scrubber.tripped and streamed_any and not withheld_notice_sent:
                writer({"event": "delta", "data": {"text": "\n\n[Response withheld.]"}})
        leak_detected = scrubber.tripped if scrubber is not None else False

        _record_pass_usage(state, gathered)

        final_message = AIMessage(
            content=gathered.content,
            tool_calls=gathered.tool_calls,
        )

        if gathered.tool_calls:
            writer({"event": "status", "data": {"text": "Checking CMU tools..."}})
            return {"messages": [final_message], "leak_detected": leak_detected}

        return {
            "messages": [final_message],
            "response_text": _message_text(gathered),
            "streamed": streamed_any,
            "leak_detected": leak_detected,
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
            ok = True
            if tool is None:
                ok = False
                result = f"Tool '{name}' is not available."
            else:
                try:
                    raw = await tool.ainvoke(args)
                    result = raw if isinstance(raw, str) else str(raw)
                except ToolException as exc:
                    # Server-authored errors are data the model needs, like
                    # "no building with that id".
                    ok = False
                    result = f"Tool '{name}' returned an error: {exc}"
                except Exception:
                    # Transport errors embed internal URLs and hosts, so the
                    # model gets a generic string and the detail stays in
                    # server logs.
                    ok = False
                    result = f"Tool '{name}' failed."
                    _LOG.exception("tool %s failed", name)

            new_invocations.append(
                {"name": name, "arguments": args, "result": result, "ok": ok}
            )
            if name not in state["services_used"] and name not in new_services:
                new_services.append(name)

            # Wrapped so the model treats tool output as untrusted data,
            # never as instructions. The invocation above keeps the full
            # result for map inference, only the model copy is capped.
            wrapped = (
                f'<<<TOOL_OUTPUT name="{name}" trust="untrusted-data">>>\n'
                f"{_truncate_tool_result(result)}\n"
                "<<<END_TOOL_OUTPUT>>>"
            )
            new_messages.append(ToolMessage(content=wrapped, tool_call_id=call_id))

        writer({"event": "status", "data": {"text": "Writing answer..."}})
        return {
            "messages": new_messages,
            "tool_invocations": new_invocations,
            "services_used": new_services,
            "tool_rounds": 1,
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

    # CMUMaps switched off means no map embed either, not just no map tools.
    # The guard below is what attaches the map to the answer.
    if "maps" not in normalize_disabled_groups(state.get("disabled_tools")):
        parsed = _apply_cmu_maps_guard(parsed, msgs, invocations)
    parsed = apply_tool_transparency_guard(parsed, msgs, services)

    # The output guard runs after the guards above so any text they injected
    # is scanned too. A stream-time trip forces the refusal outright, and a
    # refusal must never ship with a map attached.
    prompt_text = _message_text(state["messages"][0]) if state.get("messages") else ""
    if state.get("leak_detected"):
        parsed.response_text = REFUSAL_TEXT
        parsed.cmu_maps = CmuMaps()
    else:
        cleaned, replaced = apply_output_guard(parsed.response_text or "", prompt_text)
        parsed.response_text = cleaned
        if replaced:
            parsed.cmu_maps = CmuMaps()

    parsed.thought = compute_thought(services, invocations, parsed.response_text)
    parsed.action = ActionType.RETRIEVE if services else ActionType.RESPOND

    # A buffered answer (forced tool pass or map query) has not streamed yet.
    # Emit the repaired text now so the user only sees the corrected version.
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


def build_graph(model: ChatOpenAI, tools: list[BaseTool], maps_enabled: bool = True):
    """Compile the agent graph for one request (model and tools captured).

    `tools` must already have the user's disabled groups filtered out. The
    agent node binds exactly this list, so anything missing here is uncallable.
    """
    # ty does not yet structurally match TypedDict against langgraph's
    # StateLike protocol (still failing on ty 0.0.65). AgentState is a plain
    # TypedDict, the canonical shape LangGraph expects here.
    graph = StateGraph(AgentState)  # ty: ignore[invalid-argument-type]
    graph.add_node("agent", _build_agent_node(model, tools, maps_enabled))
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


def _cap_history_text(content: str) -> str:
    if len(content) <= _HISTORY_MAX_MESSAGE_CHARS:
        return content
    # Keep the head. Answers front-load the substance follow-ups reference.
    return content[:_HISTORY_MAX_MESSAGE_CHARS] + "\n[earlier turn truncated]"


def _sanitize_history(
    message_history: list[dict[str, str]] | None,
) -> list[AnyMessage]:
    """Convert caller history to LangChain messages, sanitized.

    Smuggled system or tool turns are an injection vector, so only user and
    assistant turns carry over.
    """
    if not message_history:
        return []
    out: list[AnyMessage] = []
    for turn in message_history[-_HISTORY_MAX_MESSAGES:]:
        role = turn.get("role")
        content = turn.get("content")
        if not isinstance(content, str):
            continue
        if role == "user":
            out.append(HumanMessage(content=_cap_history_text(content)))
        elif role == "assistant":
            out.append(AIMessage(content=_cap_history_text(content)))
    return out


def _history_hint_texts(
    message_history: list[dict[str, str]] | None,
) -> list[str]:
    """User turns that feed tool-group narrowing.

    User turns only, because assistant turns repeat tool data wholesale and
    would match every group.
    """
    if not message_history:
        return []
    texts = [
        turn["content"]
        for turn in message_history
        if turn.get("role") == "user" and isinstance(turn.get("content"), str)
    ]
    return texts[-_HISTORY_HINT_TURNS:]


def _initial_state(
    user_input: UserInput,
    message_history: list[dict[str, str]] | None,
    tools: list[BaseTool],
    disabled_tools: list[str] | None = None,
) -> AgentState:
    prompt = build_system_prompt(tools, disabled_tools)
    messages: list[AnyMessage] = [SystemMessage(content=prompt)]
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
        disabled_tools=list(disabled_tools or []),
        user_id=user_input.user_id or "",
        tool_rounds=0,
        leak_detected=False,
    )


async def _prepare_run(
    model: str,
    disabled_tools: list[str] | None,
    query: str,
    message_history: list[dict[str, str]] | None,
) -> tuple[Any, list[BaseTool]]:
    """Load tools, narrow them to the query, and compile the graph.

    Narrowing runs after the disabled-group filter so a keyword match can
    never re-bind a switched-off group.
    """
    tools = filter_tools(await load_mcp_tools(), disabled_tools)
    tools = select_tools_for_query(tools, query, _history_hint_texts(message_history))
    maps_enabled = "maps" not in normalize_disabled_groups(disabled_tools)
    return build_graph(_make_chat_model(model), tools, maps_enabled), tools


async def run_agent(
    user_input: UserInput,
    model: str = "openai/gpt-5.6-luna",
    message_history: list[dict[str, str]] | None = None,
    disabled_tools: list[str] | None = None,
) -> AgentResponse:
    """Non-streaming entry point. Runs the graph and returns the full response.

    `disabled_tools` lists the groups the user switched off in the Surface.
    Those tools are never bound.
    """
    if not _api_key():
        return _fallback_response(
            "OPENROUTER_API_KEY is not configured.",
            confidence=0.2,
        )

    # Flagrant jailbreak phrasing gets the canned refusal before any tool
    # loading or model call, so the whole turn costs zero tokens.
    if is_flagrant_injection(user_input.query):
        return canned_refusal_response()

    graph, tools = await _prepare_run(
        model, disabled_tools, user_input.query, message_history
    )
    final = await graph.ainvoke(
        _initial_state(user_input, message_history, tools, disabled_tools)
    )

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
    disabled_tools: list[str] | None = None,
) -> AsyncIterator[StreamEvent]:
    """Streaming entry point. Yields ('delta', ...) through ('done', ...)."""
    if not _api_key():
        fb = _fallback_response(
            "OPENROUTER_API_KEY is not configured.",
            confidence=0.2,
        )
        yield ("delta", {"text": fb.response_text})
        yield ("done", fb.model_dump())
        return

    # Same zero-token fast path as run_agent, kept stream-shaped.
    if is_flagrant_injection(user_input.query):
        refusal = canned_refusal_response()
        yield ("delta", {"text": refusal.response_text})
        yield ("done", refusal.model_dump())
        return

    graph, tools = await _prepare_run(
        model, disabled_tools, user_input.query, message_history
    )

    async for chunk in graph.astream(
        _initial_state(user_input, message_history, tools, disabled_tools),
        stream_mode="custom",
    ):
        if isinstance(chunk, dict) and "event" in chunk:
            yield (chunk["event"], chunk.get("data", {}))
