"""Chat endpoints: the complete reply, the streamed reply, and chat titles."""

import json
import logging
from collections.abc import AsyncIterator, Mapping
from http import HTTPStatus
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import ValidationError

from cmugpt import UserInput, run_agent, stream_agent_response
from cmugpt.memory import is_valid_user_id
from cmugpt.moderation import (
    ALLOW,
    blocked_input_response,
    moderate_text,
    redacted_output_response,
)
from cmugpt.title import generate_chat_title
from cmugpt.token_limits import DailyTokenLimitExceeded, ensure_within_daily_limit

from ..deps import reject_oversized_body, require_shared_secret

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/agent", dependencies=[Depends(require_shared_secret)])

# Upper bounds on request input. Query and history text is sent to the
# model, where longer input costs more tokens, and user_id becomes each
# user's storage namespace, so none of these values may arrive unbounded.
_MAX_QUERY_CHARS = 8_000
_MAX_USER_ID_CHARS = 128
_MAX_HISTORY_MESSAGES = 40
_MAX_HISTORY_ITEMS = 200
_MAX_HISTORY_MESSAGE_CHARS = 8_000


def _normalize_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize incoming payloads into the shape expected by UserInput."""
    # Support wrappers like {"data": {...}} while keeping a strict final schema.
    candidate: Any = payload.get("data", payload)
    if not isinstance(candidate, Mapping):
        raise ValueError("Payload must be a JSON object.")

    query = candidate.get("query") or candidate.get("message") or candidate.get("input")
    if not isinstance(query, str) or not query.strip():
        raise ValueError("A non-empty 'query' field is required.")
    if len(query.strip()) > _MAX_QUERY_CHARS:
        raise ValueError(f"'query' must be at most {_MAX_QUERY_CHARS} characters.")

    context = candidate.get("context")
    if context is not None and not isinstance(context, Mapping):
        raise ValueError("'context' must be a JSON object if provided.")

    user_id = candidate.get("user_id")
    if user_id is not None and not isinstance(user_id, str):
        raise ValueError("'user_id' must be a string if provided.")
    # user_id becomes the key that separates one user's stored memory from
    # another's, and the database matches it as a pattern rather than
    # literally. Characters with special meaning there must be excluded, so
    # is_valid_user_id enforces a strict allowlist.
    if user_id is not None and not is_valid_user_id(user_id):
        raise ValueError(
            "'user_id' must match [A-Za-z0-9@:+=~-] and be at most "
            f"{_MAX_USER_ID_CHARS} characters."
        )

    normalized: dict[str, Any] = {"query": query.strip()}
    if context is not None:
        normalized["context"] = dict(context)
    if user_id is not None:
        normalized["user_id"] = user_id
    return normalized


def _parse_disabled_tools_value(raw: Any) -> list[str]:
    """Tool groups the Surface reports the user disabled.

    Unknown group ids pass through here and the agent drops them, so a
    Surface that gains a new toggle before the agent knows it keeps working.
    """
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST,
            detail="'disabled_tools' must be a list of strings if provided.",
        )
    items = [item for item in raw if isinstance(item, str)]
    if len(items) != len(raw):
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST,
            detail="'disabled_tools' must be a list of strings if provided.",
        )
    return items


def _parse_request(
    payload: Any,
) -> tuple[UserInput, str | None, list[dict[str, str]] | None, list[str]]:
    """Validate the body and return (user_input, model, history, disabled_tools)."""
    if not isinstance(payload, Mapping):
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST,
            detail="Request body must be a JSON object.",
        )

    # Optional fields may appear inside the {"data": {...}} wrapper or at
    # the top level beside it. Callers use both shapes.
    wrapper: Any = payload.get("data", payload)
    if not isinstance(wrapper, Mapping):
        wrapper = payload

    def _optional(field: str) -> Any:
        value = wrapper.get(field)
        return value if value is not None else payload.get(field)

    try:
        normalized_input = _normalize_payload(payload)
        user_input = UserInput(**normalized_input)
    except (ValueError, ValidationError) as exc:
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST,
            detail=str(exc),
        ) from exc

    raw_model = _optional("model")
    model = raw_model if isinstance(raw_model, str) else None

    message_history = _optional("message_history")
    if message_history is not None and not isinstance(message_history, list):
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST,
            detail="'message_history' must be a list if provided.",
        )
    if isinstance(message_history, list) and len(message_history) > _MAX_HISTORY_ITEMS:
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST,
            detail=f"'message_history' must have at most {_MAX_HISTORY_ITEMS} items.",
        )
    if isinstance(message_history, list):
        # Accept user/assistant/system at the boundary. The agent strips
        # `system` defensively. Surface clients keep `system` rows in their
        # DB schema, so rejecting them here would break production.
        valid_history = all(
            isinstance(item, Mapping)
            and item.get("role") in ("user", "assistant", "system")
            and isinstance(item.get("content"), str)
            for item in message_history
        )
        if not valid_history:
            raise HTTPException(
                status_code=HTTPStatus.BAD_REQUEST,
                detail=(
                    "'message_history' items must be objects with "
                    "'role' in {'user','assistant','system'} and a string "
                    "'content' field."
                ),
            )
        # History is sent to the model as-is, so both the number of turns
        # and the size of each message are capped to keep one request from
        # carrying an unbounded token cost. Oversized history is trimmed, the
        # way a context window truncates, so the request still succeeds.
        message_history = [
            {
                "role": str(item["role"]),
                "content": str(item["content"])[:_MAX_HISTORY_MESSAGE_CHARS],
            }
            for item in message_history[-_MAX_HISTORY_MESSAGES:]
        ]

    return (
        user_input,
        model,
        message_history,
        _parse_disabled_tools_value(_optional("disabled_tools")),
    )


def _enforce_daily_token_limit(user_input: UserInput) -> None:
    """Reject with 429 once the user's daily budget is exhausted.

    Checked before any model call, because the streaming endpoint has
    already returned 200 by the time its generator runs and therefore
    cannot signal this condition itself.
    """
    try:
        ensure_within_daily_limit(user_input.user_id)
    except DailyTokenLimitExceeded as exc:
        raise HTTPException(
            status_code=HTTPStatus.TOO_MANY_REQUESTS,
            detail=str(exc),
        ) from exc


@router.post("/respond")
async def agent_respond(request: Request) -> JSONResponse:
    reject_oversized_body(request)
    try:
        payload = await request.json()
    except Exception as exc:
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST,
            detail="Request body must be valid JSON object.",
        ) from exc

    user_input, model, message_history, disabled_tools = _parse_request(payload)
    _enforce_daily_token_limit(user_input)

    verdict = await moderate_text(user_input.query)
    if verdict.action != ALLOW:
        logger.warning(
            "moderation: blocked input (%s: %s)",
            verdict.action,
            ", ".join(verdict.categories),
        )
        return JSONResponse(
            content=blocked_input_response(verdict.action).model_dump(),
            status_code=HTTPStatus.OK,
        )

    try:
        agent_response = await run_agent(
            user_input=user_input,
            model=model or "openai/gpt-5.6-luna",
            message_history=message_history,
            disabled_tools=disabled_tools,
        )
    except Exception as exc:
        # Log the real error server-side. Exception text can leak internal
        # URLs/config, so clients get a generic message.
        logger.exception("agent execution failed")
        raise HTTPException(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
            detail="Agent execution failed.",
        ) from exc

    out_verdict = await moderate_text(agent_response.response_text)
    if out_verdict.action != ALLOW:
        logger.warning(
            "moderation: redacted reply (%s: %s)",
            out_verdict.action,
            ", ".join(out_verdict.categories),
        )
        agent_response = redacted_output_response(out_verdict.action)

    return JSONResponse(
        content=agent_response.model_dump(),
        status_code=HTTPStatus.OK,
    )


@router.post("/title")
async def agent_title(request: Request) -> JSONResponse:
    """Generate a short chat title from the chat's first user message.

    Returns ``{"title": null}`` when generation fails, so the caller keeps
    its placeholder title and never sees an error.
    """
    try:
        payload = await request.json()
    except Exception as exc:
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST,
            detail="Request body must be valid JSON object.",
        ) from exc

    query = payload.get("query") if isinstance(payload, Mapping) else None
    if not isinstance(query, str) or not query.strip():
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST,
            detail="Field 'query' must be a non-empty string.",
        )

    title = await generate_chat_title(query)
    return JSONResponse(content={"title": title}, status_code=HTTPStatus.OK)


def _sse(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


@router.post("/respond/stream")
async def agent_respond_stream(request: Request) -> StreamingResponse:
    """Server-Sent Events endpoint.

    Emits:
        event: status data: {"text": "<short progress label>"}
        event: map    data: <CMU Maps payload JSON>
        event: memory data: {"op": "add"|"remove", "text": "<confirmation>"}
        event: delta  data: {"text": "<chunk of response_text>"}
        event: done   data: <full AgentResponse JSON>
        event: error  data: {"error": "...", "detail": "..."}
    """
    reject_oversized_body(request)
    try:
        payload = await request.json()
    except Exception as exc:
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST,
            detail="Request body must be valid JSON object.",
        ) from exc

    user_input, model, message_history, disabled_tools = _parse_request(payload)
    _enforce_daily_token_limit(user_input)

    async def event_stream() -> AsyncIterator[bytes]:
        try:
            verdict = await moderate_text(user_input.query)
            if verdict.action != ALLOW:
                logger.warning(
                    "moderation: blocked input (%s: %s)",
                    verdict.action,
                    ", ".join(verdict.categories),
                )
                blocked = blocked_input_response(verdict.action)
                yield _sse("delta", {"text": blocked.response_text}).encode("utf-8")
                yield _sse("done", blocked.model_dump()).encode("utf-8")
                return

            # The done payload is held back until the finished reply passes an
            # output check: deltas have already streamed, but done is what the
            # Surface persists and re-renders, so redacting it retroactively
            # scrubs the reply everywhere that outlives the stream.
            final_payload: dict[str, Any] | None = None
            async for event_name, data in stream_agent_response(
                user_input=user_input,
                model=model or "openai/gpt-5.6-luna",
                message_history=message_history,
                disabled_tools=disabled_tools,
            ):
                if event_name == "done":
                    final_payload = data
                    continue
                yield _sse(event_name, data).encode("utf-8")

            if final_payload is not None:
                out_verdict = await moderate_text(
                    str(final_payload.get("response_text", ""))
                )
                if out_verdict.action != ALLOW:
                    logger.warning(
                        "moderation: redacted reply (%s: %s)",
                        out_verdict.action,
                        ", ".join(out_verdict.categories),
                    )
                    final_payload = redacted_output_response(
                        out_verdict.action
                    ).model_dump()
                yield _sse("done", final_payload).encode("utf-8")
        except Exception:
            # Same policy as the non-streaming endpoint: log the real error,
            # send the client a generic one.
            logger.exception("agent stream failed")
            err = "Agent execution failed."
            yield _sse("error", {"error": err, "detail": err}).encode("utf-8")

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            # Tell nginx-style proxies to pass events through unbuffered.
            "X-Accel-Buffering": "no",
        },
    )
