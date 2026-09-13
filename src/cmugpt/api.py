import json
import logging
import os
import secrets
import time
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from http import HTTPStatus
from typing import Annotated, Any, Literal

import uvicorn
from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import ValidationError

from cmugpt import UserInput, run_agent, stream_agent_response
from cmugpt.graph import drain_background_tasks
from cmugpt.memory import (
    clear_memory,
    close_store,
    delete_memory_item,
    ensure_store,
    is_valid_user_id,
    list_memory_items,
    setup_store,
    store_is_ready,
    store_status,
)
from cmugpt.moderation import (
    ALLOW,
    blocked_input_response,
    moderate_text,
    redacted_output_response,
)
from cmugpt.title import generate_chat_title
from cmugpt.token_limits import DailyTokenLimitExceeded, ensure_within_daily_limit

logger = logging.getLogger(__name__)

# Request bodies are rejected by header before parsing. Uvicorn imposes
# no default body limit, so oversized input is both a cost and an abuse
# vector.
_MAX_BODY_BYTES = 256 * 1024

# Upper bounds on request input. Query and history text is sent to the
# model, where longer input costs more tokens, and user_id becomes each
# user's storage namespace, so none of these values may arrive unbounded.
_MAX_QUERY_CHARS = 8_000
_MAX_USER_ID_CHARS = 128
_MAX_HISTORY_MESSAGES = 40
_MAX_HISTORY_ITEMS = 200
_MAX_HISTORY_MESSAGE_CHARS = 8_000
_PRODUCTION_ENV_NAMES = ("AGENT_ENV", "APP_ENV", "ENVIRONMENT", "SECRETSPEC_PROFILE")
_PRODUCTION_ENV_VALUES = {"prod", "production"}


def _is_production() -> bool:
    return any(
        os.getenv(name, "").strip().lower() in _PRODUCTION_ENV_VALUES
        for name in _PRODUCTION_ENV_NAMES
    )


def _validate_runtime_configuration() -> None:
    """Refuse to start a production deployment without a database or a shared
    secret. Starting anyway would run without durable memory or auth."""
    if not _is_production():
        return
    missing = [
        name
        for name in ("DATABASE_URL", "AGENT_SHARED_SECRET")
        if not os.getenv(name, "").strip()
    ]
    if missing:
        raise RuntimeError(
            "Production configuration is missing required environment "
            f"variable(s): {', '.join(missing)}. Refusing to start with "
            "non-durable or unauthenticated user memory."
        )
    secret = os.environ["AGENT_SHARED_SECRET"]
    if secret != secret.strip():
        raise RuntimeError(
            "AGENT_SHARED_SECRET cannot have leading or trailing whitespace."
        )
    if len(secret) < 32:
        raise RuntimeError(
            "AGENT_SHARED_SECRET must be at least 32 characters in production."
        )


@asynccontextmanager
async def _lifespan(_: FastAPI) -> AsyncIterator[None]:
    """Open the memory store when the app starts. On shutdown, wait for any
    background memory writes to finish, then close the store."""
    _validate_runtime_configuration()
    if not os.getenv("AGENT_SHARED_SECRET"):
        logger.warning(
            "AGENT_SHARED_SECRET is not set: /agent/respond* and /memory/* "
            "are UNAUTHENTICATED. This is only acceptable in local dev."
        )
    await setup_store()
    try:
        yield
    finally:
        await drain_background_tasks()
        await close_store()


app = FastAPI(lifespan=_lifespan)

# CORS only governs browser JS calling this API from another origin. It does
# nothing against direct (curl/script/server) requests, which is why
# AGENT_SHARED_SECRET below is the actual access boundary. This just stops a
# malicious page from riding a visitor's browser to hit the API client-side.
_allowed_origins = [
    origin.strip()
    for origin in os.getenv("ALLOWED_ORIGINS", "https://cmugpt.com").split(",")
    if origin.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_methods=["GET", "POST"],
    allow_headers=["Authorization", "Content-Type"],
)

# Optional shared-secret authentication. When AGENT_SHARED_SECRET is set,
# /agent/respond* requires a matching bearer token. auto_error=False
# preserves this module's own error envelope.
_bearer_scheme = HTTPBearer(auto_error=False)


def _require_shared_secret(
    creds: HTTPAuthorizationCredentials | None = Depends(_bearer_scheme),  # noqa: B008
) -> None:
    expected = os.getenv("AGENT_SHARED_SECRET")
    if not expected:
        return  # Auth is disabled, which only local development should do.
    token_ok = (
        creds is not None
        and creds.scheme.lower() == "bearer"
        # Compared in constant time. An ordinary `!=` stops at the first
        # wrong character, and that timing difference can reveal the secret
        # one prefix at a time.
        and secrets.compare_digest(
            creds.credentials.encode("utf-8"), expected.encode("utf-8")
        )
    )
    if not token_ok:
        raise HTTPException(
            status_code=HTTPStatus.UNAUTHORIZED,
            detail="Invalid or missing bearer token.",
        )


@app.exception_handler(HTTPException)
async def _http_exception_handler(_: Request, exc: HTTPException) -> JSONResponse:
    """Emit both `error` and `detail` so older and newer clients both work."""
    detail = exc.detail if isinstance(exc.detail, str) else str(exc.detail)
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": detail, "detail": detail},
    )


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


_READY_TTL_SECONDS = 5.0
_ready_cache: tuple[float, bool] | None = None


def _reject_oversized_body(request: Request) -> None:
    """Reject oversized bodies by header before request.json() parses them."""
    length = request.headers.get("content-length")
    if length is not None and length.isdigit() and int(length) > _MAX_BODY_BYTES:
        raise HTTPException(
            status_code=HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
            detail=f"Request body must be at most {_MAX_BODY_BYTES} bytes.",
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


@app.get("/api/health")
async def health() -> JSONResponse:
    # The memory block reports which backend this deployment is using and
    # whether it is answering queries. The readiness probe runs a real store
    # query, so the result is cached briefly to keep this unauthenticated
    # endpoint from generating database load.
    global _ready_cache
    now = time.monotonic()
    if _ready_cache is not None and now - _ready_cache[0] < _READY_TTL_SECONDS:
        ready = _ready_cache[1]
    else:
        ready = await store_is_ready()
        _ready_cache = (now, ready)
    memory = {**store_status(), "ready": ready}
    return JSONResponse(
        content={"status": "ok" if ready else "degraded", "memory": memory},
        status_code=HTTPStatus.OK if ready else HTTPStatus.SERVICE_UNAVAILABLE,
    )


@app.post("/agent/respond", dependencies=[Depends(_require_shared_secret)])
async def agent_respond(request: Request) -> JSONResponse:
    _reject_oversized_body(request)
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


@app.post("/agent/title", dependencies=[Depends(_require_shared_secret)])
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


@app.post(
    "/agent/respond/stream",
    dependencies=[Depends(_require_shared_secret)],
)
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
    _reject_oversized_body(request)
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


def _require_valid_user_id(user_id: str) -> None:
    """Reject path-param user ids that are unsafe as a memory namespace key."""
    if not is_valid_user_id(user_id):
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST,
            detail="Invalid 'user_id'.",
        )


@app.get("/memory/{user_id}", dependencies=[Depends(_require_shared_secret)])
async def get_memory(
    user_id: str,
    q: Annotated[str | None, Query(max_length=200)] = None,
    kind: Literal["learned", "remembered"] | None = None,
    limit: Annotated[int, Query(ge=1, le=200)] = 200,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> JSONResponse:
    """Search a user's learned and explicitly remembered facts."""
    _require_valid_user_id(user_id)
    store = await ensure_store()
    items, total = await list_memory_items(
        store,
        user_id,
        query=q,
        memory_type=kind,
        limit=limit,
        offset=offset,
    )
    return JSONResponse(
        content={
            "user_id": user_id,
            "items": items,
            "total": total,
            "limit": limit,
            "offset": offset,
        },
        status_code=HTTPStatus.OK,
    )


@app.delete(
    "/memory/{user_id}/items/{kind}/{item_id}",
    dependencies=[Depends(_require_shared_secret)],
)
async def delete_typed_memory_item(
    user_id: str,
    kind: Literal["learned", "remembered"],
    item_id: str,
) -> JSONResponse:
    """Delete one learned or explicitly remembered fact."""
    _require_valid_user_id(user_id)
    store = await ensure_store()
    deleted = await delete_memory_item(store, user_id, kind, item_id)
    if not deleted:
        raise HTTPException(
            status_code=HTTPStatus.NOT_FOUND,
            detail="Memory item not found.",
        )
    return JSONResponse(
        content={"status": "deleted", "id": item_id, "type": kind},
        status_code=HTTPStatus.OK,
    )


@app.delete("/memory/{user_id}", dependencies=[Depends(_require_shared_secret)])
async def clear_user_memory(user_id: str) -> JSONResponse:
    """Delete all user memory, including any legacy raw-chat snippets."""
    _require_valid_user_id(user_id)
    store = await ensure_store()
    removed = await clear_memory(store, user_id)
    return JSONResponse(
        content={"status": "cleared", "removed": removed},
        status_code=HTTPStatus.OK,
    )


def main() -> None:
    # Uvicorn configures only its own loggers. Configure the root logger so
    # the application's cmugpt.* loggers emit too.
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    port = int(os.environ.get("PORT", "5000"))
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
