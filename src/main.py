import json
import logging
import os
import secrets
import sys
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from http import HTTPStatus
from pathlib import Path
from typing import Annotated, Any, Literal

import uvicorn
from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import ValidationError

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent import UserInput, run_agent, stream_agent_response
from agent.graph import drain_background_tasks
from agent.memory import (
    clear_memory,
    close_store,
    delete_fact,
    ensure_store,
    is_valid_user_id,
    list_facts,
    list_memory_items,
    setup_store,
    store_is_ready,
    store_status,
)
from agent.memory import (
    delete_memory_item as delete_memory_record,
)

logger = logging.getLogger(__name__)

# Input caps: these payloads flow straight into LLM calls (token cost) and,
# for user_id, into per-user database namespaces - so none of them may be
# caller-controlled without bounds.
_MAX_QUERY_CHARS = 8_000
_MAX_USER_ID_CHARS = 128
_MAX_HISTORY_MESSAGES = 40
_MAX_HISTORY_MESSAGE_CHARS = 8_000
_PRODUCTION_ENV_NAMES = ("AGENT_ENV", "APP_ENV", "ENVIRONMENT", "SECRETSPEC_PROFILE")
_PRODUCTION_ENV_VALUES = {"prod", "production"}


def _is_production() -> bool:
    return any(
        os.getenv(name, "").strip().lower() in _PRODUCTION_ENV_VALUES
        for name in _PRODUCTION_ENV_NAMES
    )


def _validate_runtime_configuration() -> None:
    """Fail closed when a production deployment lacks durability or auth."""
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
    """Initialize the memory store on startup; tear it down on shutdown.

    With ``DATABASE_URL`` set this opens the Postgres connection pool (and runs
    pgvector setup) once; otherwise it builds the in-memory fallback store.
    """
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

# Optional shared-secret auth. When AGENT_SHARED_SECRET is set, every request
# to /agent/respond* must send `Authorization: Bearer <secret>`. When unset,
# auth is skipped (local dev). The HTTPBearer scheme has auto_error=False so
# we can return our own structured error envelope.
_bearer_scheme = HTTPBearer(auto_error=False)


def _require_shared_secret(
    creds: HTTPAuthorizationCredentials | None = Depends(_bearer_scheme),  # noqa: B008
) -> None:
    expected = os.getenv("AGENT_SHARED_SECRET")
    if not expected:
        return  # auth disabled (dev only; the lifespan logs a warning)
    token_ok = (
        creds is not None
        and creds.scheme.lower() == "bearer"
        # Constant-time comparison: `!=` short-circuits on the first differing
        # byte, which leaks secret prefixes through response timing.
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
    """Emit both `error` and `detail` so legacy + modern clients both work."""
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
    if len(query) > _MAX_QUERY_CHARS:
        raise ValueError(f"'query' must be at most {_MAX_QUERY_CHARS} characters.")

    context = candidate.get("context")
    if context is not None and not isinstance(context, Mapping):
        raise ValueError("'context' must be a JSON object if provided.")

    user_id = candidate.get("user_id")
    if user_id is not None and not isinstance(user_id, str):
        raise ValueError("'user_id' must be a string if provided.")
    # user_id becomes a per-user database namespace matched with an unescaped
    # SQL LIKE, so it must exclude wildcard/separator characters - not merely be
    # printable. is_valid_user_id enforces the safe allowlist.
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


def _parse_request(
    payload: Any,
) -> tuple[UserInput, str | None, list[dict[str, str]] | None]:
    """Validate the request body and return (user_input, model, history)."""
    if not isinstance(payload, Mapping):
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST,
            detail="Request body must be a JSON object.",
        )

    try:
        normalized_input = _normalize_payload(payload)
        user_input = UserInput(**normalized_input)
    except (ValueError, ValidationError) as exc:
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST,
            detail=str(exc),
        ) from exc

    raw_model = payload.get("model")
    model = raw_model if isinstance(raw_model, str) else None

    message_history = payload.get("message_history")
    if message_history is not None and not isinstance(message_history, list):
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST,
            detail="'message_history' must be a list if provided.",
        )
    if isinstance(message_history, list):
        # Accept user/assistant/system at the boundary; the agent strips
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
        # History flows verbatim into the LLM call: cap turns and per-message
        # size so a single request can't carry an unbounded token bill. Trimming
        # (rather than rejecting) mirrors normal context-window truncation.
        message_history = [
            {
                "role": str(item["role"]),
                "content": str(item["content"])[:_MAX_HISTORY_MESSAGE_CHARS],
            }
            for item in message_history[-_MAX_HISTORY_MESSAGES:]
        ]

    return user_input, model, message_history


@app.get("/health")
async def health() -> JSONResponse:
    # `memory.backend` is the one-request prod check: "postgres" means this
    # deploy got a database + DATABASE_URL; "in-memory" means it fell back.
    ready = await store_is_ready()
    memory = {**store_status(), "ready": ready}
    return JSONResponse(
        content={"status": "ok" if ready else "degraded", "memory": memory},
        status_code=HTTPStatus.OK if ready else HTTPStatus.SERVICE_UNAVAILABLE,
    )


@app.post("/agent/respond", dependencies=[Depends(_require_shared_secret)])
async def agent_respond(request: Request) -> JSONResponse:
    try:
        payload = await request.json()
    except Exception as exc:
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST,
            detail="Request body must be valid JSON object.",
        ) from exc

    user_input, model, message_history = _parse_request(payload)

    try:
        agent_response = await run_agent(
            user_input=user_input,
            model=model or "openai/gpt-4o",
            message_history=message_history,
        )
    except Exception as exc:
        # Log the real error server-side; exception text can leak internal
        # URLs/config, so clients get a generic message.
        logger.exception("agent execution failed")
        raise HTTPException(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
            detail="Agent execution failed.",
        ) from exc

    return JSONResponse(
        content=agent_response.model_dump(),
        status_code=HTTPStatus.OK,
    )


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
    try:
        payload = await request.json()
    except Exception as exc:
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST,
            detail="Request body must be valid JSON object.",
        ) from exc

    user_input, model, message_history = _parse_request(payload)

    async def event_stream() -> AsyncIterator[bytes]:
        try:
            async for event_name, data in stream_agent_response(
                user_input=user_input,
                model=model or "openai/gpt-4o",
                message_history=message_history,
            ):
                yield _sse(event_name, data).encode("utf-8")
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
            "X-Accel-Buffering": "no",  # disable nginx/proxy buffering
        },
    )


def _require_valid_user_id(user_id: str) -> None:
    """Reject path-param user ids that aren't safe as a memory namespace key."""
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
    # Keep the original facts field for older Surface clients while exposing a
    # unified item list for the searchable memory manager.
    facts = await list_facts(store, user_id)
    return JSONResponse(
        content={
            "user_id": user_id,
            "facts": facts,
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
    deleted = await delete_memory_record(store, user_id, kind, item_id)
    if not deleted:
        raise HTTPException(
            status_code=HTTPStatus.NOT_FOUND,
            detail="Memory item not found.",
        )
    return JSONResponse(
        content={"status": "deleted", "id": item_id, "type": kind},
        status_code=HTTPStatus.OK,
    )


@app.delete(
    "/memory/{user_id}/{fact_id}",
    dependencies=[Depends(_require_shared_secret)],
)
async def delete_memory_item(user_id: str, fact_id: str) -> JSONResponse:
    """Delete a single remembered fact for a user."""
    _require_valid_user_id(user_id)
    store = await ensure_store()
    await delete_fact(store, user_id, fact_id)
    return JSONResponse(
        content={"status": "deleted", "id": fact_id},
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
    # Uvicorn only configures its own loggers; configure the root logger so
    # application logs (agent.*, src.*) actually emit.
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    port = int(os.environ.get("PORT", "5000"))
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
