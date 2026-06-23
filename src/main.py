import json
import os
import sys
from collections.abc import AsyncIterator, Mapping
from http import HTTPStatus
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import ValidationError

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent import UserInput, run_agent, stream_agent_response

app = FastAPI()

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
        return  # auth disabled
    if (
        creds is None
        or creds.scheme.lower() != "bearer"
        or creds.credentials != expected
    ):
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

    context = candidate.get("context")
    if context is not None and not isinstance(context, Mapping):
        raise ValueError("'context' must be a JSON object if provided.")

    user_id = candidate.get("user_id")
    if user_id is not None and not isinstance(user_id, str):
        raise ValueError("'user_id' must be a string if provided.")

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

    return user_input, model, message_history


@app.get("/health")
async def health() -> JSONResponse:
    return JSONResponse(content={"status": "ok"}, status_code=HTTPStatus.OK)


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
        raise HTTPException(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
            detail=f"Agent execution failed: {exc}",
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
        except Exception as exc:
            err = f"Agent execution failed: {exc}"
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


def main() -> None:
    port = int(os.environ.get("PORT", "5000"))
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
