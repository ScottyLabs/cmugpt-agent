import os
import sys
from collections.abc import Mapping
from http import HTTPStatus
from pathlib import Path
from typing import Any
import uvicorn

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from pydantic import ValidationError

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent import UserInput, run_agent

app = FastAPI()


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


async def _run_agent_async(
    user_input: UserInput,
    *,
    model: str | None,
    message_history: list[dict[str, str]] | None,
) -> Any:
    """Run the async agent logic from an async FastAPI route."""
    return await run_agent(
        user_input=user_input,
        model=model or "openai/gpt-4o",
        message_history=message_history,
    )


@app.get("/health")
async def health() -> JSONResponse:
    return JSONResponse(content={"status": "ok"}, status_code=HTTPStatus.OK)


@app.post("/agent/respond")
async def agent_respond(request: Request):
    try:
        payload = await request.json()
    except Exception:
        raise HTTPException(status_code=HTTPStatus.BAD_REQUEST, detail="Request body must be valid JSON object.")

    if not isinstance(payload, Mapping):
        raise HTTPException(status_code=HTTPStatus.BAD_REQUEST, detail="Request body must be a JSON object.")

    try:
        normalized_input = _normalize_payload(payload)
        user_input = UserInput(**normalized_input)
    except (ValueError, ValidationError) as exc:
        raise HTTPException(status_code=HTTPStatus.BAD_REQUEST, detail=str(exc))

    model = payload.get("model")
    message_history = payload.get("message_history")
    if message_history is not None and not isinstance(message_history, list):
        raise HTTPException(status_code=HTTPStatus.BAD_REQUEST, detail="'message_history' must be a list if provided.")
    if isinstance(message_history, list):
        valid_history = all(
            isinstance(item, Mapping)
            and isinstance(item.get("role"), str)
            and isinstance(item.get("content"), str)
            for item in message_history
        )
        if not valid_history:
            raise HTTPException(
                status_code=HTTPStatus.BAD_REQUEST,
                detail=(
                    "'message_history' items must be objects with string 'role' and 'content' fields."
                ),
            )

    try:
        agent_response = await _run_agent_async(
            user_input,
            model=model if isinstance(model, str) else None,
            message_history=message_history,
        )
    except Exception as exc:
        raise HTTPException(status_code=HTTPStatus.INTERNAL_SERVER_ERROR, detail=f"Agent execution failed: {exc}")

    return JSONResponse(content=agent_response.model_dump(), status_code=HTTPStatus.OK)


def main() -> None:
    port = int(os.environ.get("PORT", "5001"))
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
