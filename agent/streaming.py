"""SSE streaming for the CMUGPT agent.

The agent's structured output is normally a single JSON object whose
`response_text` field contains the user-facing Markdown. Some models still
produce plain Markdown after a tool pass; we stream that directly instead of
waiting for a JSON key that will never arrive.

We stream by:

1. Running the (tool-using) LLM loop with `stream=True`.
2. Emitting `status` events while tools are being called.
3. If the final answer begins as JSON, buffering raw content until we see the
    `"response_text"` marker.
4. From there, emitting JSON-unescaped characters as `delta` events as soon
    as each escape sequence resolves. If the final answer begins as Markdown,
    emitting it directly as `delta` events.
5. Once the closing string-quote is reached for JSON, we keep buffering the
    rest.
6. After the stream ends, we parse the complete buffer and emit a `done`
    event with the full AgentResponse.

If parsing fails at any point we fall back to a single delta+done with the
non-streamed result, so streaming clients always get a usable answer.
"""

from __future__ import annotations

import json
import os
from collections.abc import AsyncIterator
from contextlib import suppress
from typing import Any

from mcp.client.session import ClientSession
from mcp.client.streamable_http import streamable_http_client

from .agent_hub import (
    _apply_tool_transparency_guard,
    _build_system_prompt,
    _call_mcp_tool,
    _fallback_response,
    _get_mcp_tools,
    _parse_agent_response,
    _should_require_tool,
    _tool_metadata_message,
    client,
)
from .schema import UserInput

MCP_SERVER_URL = os.getenv("MCP_SERVER_URL", "")

StreamEvent = tuple[str, dict[str, Any]]


def _sanitize_history(
    message_history: list[dict[str, str]] | None,
) -> list[dict[str, str]]:
    if not message_history:
        return []
    return [
        t
        for t in message_history
        if t.get("role") in ("user", "assistant") and isinstance(t.get("content"), str)
    ]


def _build_messages(
    user_input: UserInput,
    openai_tools: list[dict] | None,
    history: list[dict[str, str]],
) -> list[dict[str, Any]]:
    msgs: list[dict[str, Any]] = [
        {"role": "system", "content": _build_system_prompt(openai_tools)}
    ]
    msgs.extend(history)
    msgs.append({"role": "user", "content": user_input.query})
    return msgs


class _ResponseTextStreamer:
    """Incremental parser for streamed user-facing text.

    Feeds raw content chunks in, yields decoded characters of `response_text`
    as soon as each escape sequence is complete. If the model ignores the
    schema and starts with Markdown instead of JSON, yields the raw Markdown
    directly. The full raw buffer is also retained so the caller can parse or
    wrap it after the stream ends.
    """

    _MARKER = '"response_text"'

    def __init__(self) -> None:
        self.buffer = ""
        self._mode = "unknown"  # unknown -> json | raw
        self._raw_emit_pos = 0
        self._scan_pos = 0
        self._state = "search"  # search -> in_string -> done
        self._escape_pending = False
        self._in_unicode_escape = False
        self._unicode_digits = ""

    def feed(self, chunk: str) -> str:
        """Append a chunk and return any new decoded response_text characters."""
        if not chunk:
            return ""
        self.buffer += chunk
        out_parts: list[str] = []

        if self._mode == "unknown":
            first_content_idx = 0
            while (
                first_content_idx < len(self.buffer)
                and self.buffer[first_content_idx].isspace()
            ):
                first_content_idx += 1
            if first_content_idx >= len(self.buffer):
                return ""
            self._mode = "json" if self.buffer[first_content_idx] == "{" else "raw"

        if self._mode == "raw":
            out = self.buffer[self._raw_emit_pos :]
            self._raw_emit_pos = len(self.buffer)
            return out

        if self._state == "search":
            idx = self.buffer.find(self._MARKER, self._scan_pos)
            if idx < 0:
                # Nothing to do yet; keep enough tail to handle a marker that
                # straddles a chunk boundary.
                self._scan_pos = max(0, len(self.buffer) - len(self._MARKER))
                return ""
            i = idx + len(self._MARKER)
            while i < len(self.buffer) and self.buffer[i] in " \t\r\n:":
                i += 1
            if i >= len(self.buffer):
                self._scan_pos = idx
                return ""
            if self.buffer[i] != '"':
                self._scan_pos = i + 1
                return ""
            self._state = "in_string"
            self._scan_pos = i + 1

        if self._state == "in_string":
            i = self._scan_pos
            while i < len(self.buffer):
                ch = self.buffer[i]
                if self._in_unicode_escape:
                    self._unicode_digits += ch
                    i += 1
                    if len(self._unicode_digits) == 4:
                        with suppress(ValueError):
                            out_parts.append(chr(int(self._unicode_digits, 16)))
                        self._unicode_digits = ""
                        self._in_unicode_escape = False
                    continue
                if self._escape_pending:
                    self._escape_pending = False
                    if ch == "u":
                        self._in_unicode_escape = True
                        self._unicode_digits = ""
                        i += 1
                        continue
                    mapped = {
                        '"': '"',
                        "\\": "\\",
                        "/": "/",
                        "b": "\b",
                        "f": "\f",
                        "n": "\n",
                        "r": "\r",
                        "t": "\t",
                    }.get(ch, ch)
                    out_parts.append(mapped)
                    i += 1
                    continue
                if ch == "\\":
                    self._escape_pending = True
                    i += 1
                    continue
                if ch == '"':
                    self._state = "done"
                    self._scan_pos = i + 1
                    return "".join(out_parts)
                out_parts.append(ch)
                i += 1
            self._scan_pos = i

        return "".join(out_parts)


def _final_streaming_instruction(services_used: list[str]) -> dict[str, str]:
    names = ", ".join(f"`{name}`" for name in services_used) or "none"
    return {
        "role": "system",
        "content": (
            "Final streaming instruction: the next assistant message is the "
            "final answer. Output only the strict JSON object. Put "
            '`response_text` first and begin with `{"response_text":`. Do '
            "not put `thought`, `action`, `tool_calls`, or `services_used` "
            "before `response_text`; those fields come after it. "
            f"Authoritative services_used for this turn: {names}."
        ),
    }


async def _run_streaming_loop(
    *,
    messages: list[dict[str, Any]],
    model: str,
    openai_tools: list[dict] | None,
    call_tool: Any,
) -> AsyncIterator[StreamEvent]:
    """Run the LLM loop with streaming, yielding SSE events.

    Tool-call iterations emit status only. The final synthesis pass parses
    response_text incrementally or streams plain Markdown directly.
    """
    services_used: list[str] = []
    final_content = ""

    for _ in range(10):
        chat_kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": True,
        }
        if openai_tools:
            chat_kwargs["tools"] = openai_tools
        if openai_tools and not services_used and _should_require_tool(messages):
            chat_kwargs["tool_choice"] = "required"

        stream = await client.chat.completions.create(**chat_kwargs)

        accumulated_content = ""
        # tool_calls indexed by their position in the message
        tool_call_buf: dict[int, dict[str, Any]] = {}
        text_streamer = _ResponseTextStreamer()
        finish_reason: str | None = None

        async for chunk in stream:
            if not chunk.choices:
                continue
            choice = chunk.choices[0]
            delta = choice.delta
            if choice.finish_reason:
                finish_reason = choice.finish_reason

            if delta.tool_calls:
                for tc in delta.tool_calls:
                    idx = tc.index if tc.index is not None else 0
                    slot = tool_call_buf.setdefault(
                        idx,
                        {
                            "id": "",
                            "type": "function",
                            "function": {"name": "", "arguments": ""},
                        },
                    )
                    if tc.id:
                        slot["id"] = tc.id
                    if tc.function:
                        if tc.function.name:
                            slot["function"]["name"] += tc.function.name
                        if tc.function.arguments:
                            slot["function"]["arguments"] += tc.function.arguments

            if delta.content:
                accumulated_content += delta.content
                # Only stream deltas if this iteration produced no tool calls
                # (heuristic: if we've started seeing tool_calls, stay silent).
                if not tool_call_buf:
                    new_chars = text_streamer.feed(delta.content)
                    if new_chars:
                        yield ("delta", {"text": new_chars})

        # If the model emitted tool_calls, execute them and loop.
        if tool_call_buf and call_tool is not None:
            yield ("status", {"text": "Checking CMU tools..."})
            assistant_msg: dict[str, Any] = {
                "role": "assistant",
                "content": accumulated_content or None,
                "tool_calls": [
                    {
                        "id": tc["id"] or f"call_{i}",
                        "type": "function",
                        "function": {
                            "name": tc["function"]["name"],
                            "arguments": tc["function"]["arguments"] or "{}",
                        },
                    }
                    for i, tc in sorted(tool_call_buf.items())
                ],
            }
            messages.append(assistant_msg)

            for slot in (tc for _, tc in sorted(tool_call_buf.items())):
                name = slot["function"]["name"]
                if not name:
                    continue
                if name not in services_used:
                    services_used.append(name)
                args_raw = slot["function"]["arguments"] or "{}"
                try:
                    args = json.loads(args_raw)
                except json.JSONDecodeError:
                    args = {}
                result = await call_tool(name, args)
                wrapped = (
                    f'<<<TOOL_OUTPUT name="{name}"'
                    ' trust="untrusted-data">>>\n'
                    f"{result}\n"
                    "<<<END_TOOL_OUTPUT>>>"
                )
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": slot["id"] or f"call_{name}",
                        "content": wrapped,
                    }
                )
            messages.append(_tool_metadata_message(services_used))
            messages.append(_final_streaming_instruction(services_used))
            yield ("status", {"text": "Writing answer..."})
            continue

        # Final answer (no tool calls this iteration).
        final_content = accumulated_content
        _ = finish_reason
        break
    else:
        # Loop exhausted without final answer.
        fallback = _fallback_response(
            "Unable to complete the request within allowed steps.",
            confidence=0.3,
        )
        if services_used:
            fallback.services_used = services_used
        yield ("delta", {"text": fallback.response_text})
        yield ("done", fallback.model_dump())
        return

    # Parse the final JSON for metadata + emit done.
    parsed = _parse_agent_response(final_content)
    parsed = _apply_tool_transparency_guard(parsed, messages, services_used)
    yield ("done", parsed.model_dump())


async def stream_agent_response(
    *,
    user_input: UserInput,
    model: str,
    message_history: list[dict[str, str]] | None,
) -> AsyncIterator[StreamEvent]:
    """Public entry point: yields ('delta', ...) and ('done', ...) events."""
    if not os.getenv("OPENROUTER_API_KEY"):
        fb = _fallback_response(
            "OPENROUTER_API_KEY is not configured.",
            confidence=0.2,
        )
        yield ("delta", {"text": fb.response_text})
        yield ("done", fb.model_dump())
        return

    history = _sanitize_history(message_history)

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
                messages = _build_messages(user_input, openai_tools, history)
                async for ev in _run_streaming_loop(
                    messages=messages,
                    model=model,
                    openai_tools=openai_tools,
                    call_tool=lambda name, arguments: _call_mcp_tool(
                        session, name, arguments
                    ),
                ):
                    yield ev
                return
        except Exception:
            # Fall through to non-tool streaming.
            pass

    messages = _build_messages(user_input, None, history)
    async for ev in _run_streaming_loop(
        messages=messages,
        model=model,
        openai_tools=None,
        call_tool=None,
    ):
        yield ev
