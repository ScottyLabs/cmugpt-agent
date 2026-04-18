import asyncio
import os
import sys
from collections.abc import Mapping
from http import HTTPStatus
from pathlib import Path
from typing import Any

from flask import Flask, jsonify, request
from pydantic import ValidationError

# Running as "python src/main.py" sets sys.path to src/, so add project root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent import UserInput, run_agent

app = Flask(__name__)


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


def _run_agent_sync(
    user_input: UserInput,
    *,
    model: str | None,
    message_history: list[dict[str, str]] | None,
) -> Any:
    """Run async agent logic from a sync Flask route."""
    try:
        return asyncio.run(
            run_agent(
                user_input=user_input,
                model=model or "openai/gpt-4o",
                message_history=message_history,
            )
        )
    except RuntimeError:
        # Handles environments where an event loop is already running.
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(
                run_agent(
                    user_input=user_input,
                    model=model or "openai/gpt-4o",
                    message_history=message_history,
                )
            )
        finally:
            loop.close()


@app.get("/health")
def health() -> tuple[dict[str, str], int]:
    return {"status": "ok"}, HTTPStatus.OK


@app.post("/agent/respond")
def agent_respond():
    payload = request.get_json(silent=True)
    if not isinstance(payload, Mapping):
        return (
            jsonify({"error": "Request body must be valid JSON object."}),
            HTTPStatus.BAD_REQUEST,
        )

    try:
        normalized_input = _normalize_payload(payload)
        user_input = UserInput(**normalized_input)
    except (ValueError, ValidationError) as exc:
        return jsonify({"error": str(exc)}), HTTPStatus.BAD_REQUEST

    model = payload.get("model")
    message_history = payload.get("message_history")
    if message_history is not None and not isinstance(message_history, list):
        return (
            jsonify({"error": "'message_history' must be a list if provided."}),
            HTTPStatus.BAD_REQUEST,
        )
    if isinstance(message_history, list):
        valid_history = all(
            isinstance(item, Mapping)
            and isinstance(item.get("role"), str)
            and isinstance(item.get("content"), str)
            for item in message_history
        )
        if not valid_history:
            return (
                jsonify(
                    {
                        "error": (
                            "'message_history' items must be objects with "
                            "string 'role' and 'content' fields."
                        )
                    }
                ),
                HTTPStatus.BAD_REQUEST,
            )

    try:
        agent_response = _run_agent_sync(
            user_input,
            model=model if isinstance(model, str) else None,
            message_history=message_history,
        )
    except Exception as exc:
        return jsonify(
            {"error": f"Agent execution failed: {exc}"}
        ), HTTPStatus.INTERNAL_SERVER_ERROR

    return jsonify(agent_response.model_dump()), HTTPStatus.OK


def main() -> None:
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "5000")))


if __name__ == "__main__":
    main()
