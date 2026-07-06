"""CI smoke tests for the CMUGPT agent HTTP surface.

These tests avoid live OpenRouter and MCP calls so the default CI pipeline can
run without secrets. The existing live E2E scripts can still be run manually
when those services are configured.
"""

import os
from collections.abc import Iterator
from contextlib import contextmanager
from http import HTTPStatus
from typing import Any
from unittest.mock import patch

from fastapi.testclient import TestClient

from agent.schema import ActionType, AgentResponse, Thought, UserInput
from src import main as app_module


@contextmanager
def temporary_env(name: str, value: str | None) -> Iterator[None]:
    original = os.environ.get(name)
    if value is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = value
    try:
        yield
    finally:
        if original is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = original


def assert_equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise AssertionError(f"{label}: expected {expected!r}, got {actual!r}")


def assert_true(condition: bool, label: str) -> None:
    if not condition:
        raise AssertionError(label)


async def fake_run_agent(
    user_input: UserInput,
    *,
    model: str,
    message_history: list[dict[str, str]] | None = None,
) -> AgentResponse:
    history_count = len(message_history or [])
    return AgentResponse(
        thought=Thought(reasoning=f"smoke test via {model}", confidence=0.91),
        action=ActionType.RESPOND,
        response_text=(
            f"**Echo:** {user_input.query} "
            f"(user={user_input.user_id or 'anonymous'}, history={history_count})"
        ),
        services_used=[],
    )


def test_health(client: TestClient) -> None:
    response = client.get("/health")
    assert_equal(response.status_code, HTTPStatus.OK, "health status")
    payload = response.json()
    assert_equal(payload["status"], "ok", "health status field")
    assert_true("backend" in payload["memory"], "health reports memory backend")


def test_agent_respond_accepts_supported_payload_shapes(
    client: TestClient,
) -> None:
    response = client.post(
        "/agent/respond",
        json={
            "data": {
                "message": "Where is Gates?",
                "context": {"previous_location": "CUC"},
                "user_id": "ci-user",
            },
            "model": "openai/gpt-4o-mini",
            "message_history": [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "hi"},
            ],
        },
    )
    payload = response.json()

    assert_equal(response.status_code, HTTPStatus.OK, "agent response status")
    assert_equal(payload["action"], "respond", "agent action")
    assert_equal(payload["thought"]["confidence"], 0.91, "agent confidence")
    assert_true("Where is Gates?" in payload["response_text"], "query echo")
    assert_true("history=2" in payload["response_text"], "history forwarding")
    assert_equal(payload["services_used"], [], "services used")


def test_agent_respond_rejects_invalid_payload(client: TestClient) -> None:
    response = client.post("/agent/respond", json={"query": ""})
    payload = response.json()

    assert_equal(response.status_code, HTTPStatus.BAD_REQUEST, "bad request status")
    assert_true("query" in payload["detail"].lower(), "bad request detail")
    assert_equal(payload["error"], payload["detail"], "legacy error envelope")


def test_agent_respond_enforces_input_caps(client: TestClient) -> None:
    oversized_query = client.post("/agent/respond", json={"query": "x" * 8001})
    assert_equal(
        oversized_query.status_code,
        HTTPStatus.BAD_REQUEST,
        "oversized query rejected",
    )

    oversized_user = client.post(
        "/agent/respond",
        json={"query": "Hi", "user_id": "u" * 129},
    )
    assert_equal(
        oversized_user.status_code,
        HTTPStatus.BAD_REQUEST,
        "oversized user_id rejected",
    )

    long_history = [{"role": "user", "content": f"msg {i}"} for i in range(45)]
    trimmed = client.post(
        "/agent/respond",
        json={"query": "Hi", "message_history": long_history},
    )
    assert_equal(trimmed.status_code, HTTPStatus.OK, "long history accepted")
    assert_true(
        "history=40" in trimmed.json()["response_text"],
        "history trimmed to the cap",
    )


def test_memory_endpoints_reject_wildcard_user_id(client: TestClient) -> None:
    # A LIKE-wildcard user_id must be rejected at the boundary, not reach the
    # store (where it would match every user's namespace).
    body_wildcard = client.post(
        "/agent/respond",
        json={"query": "Hi", "user_id": "%"},
    )
    assert_equal(
        body_wildcard.status_code,
        HTTPStatus.BAD_REQUEST,
        "wildcard user_id in body rejected",
    )

    path_wildcard = client.get("/memory/%25")  # %25 decodes to '%'
    assert_equal(
        path_wildcard.status_code,
        HTTPStatus.BAD_REQUEST,
        "wildcard user_id in path rejected",
    )


def test_agent_respond_enforces_shared_secret(client: TestClient) -> None:
    with temporary_env("AGENT_SHARED_SECRET", "ci-secret"):
        missing_auth = client.post("/agent/respond", json={"query": "Hi"})
        wrong_auth = client.post(
            "/agent/respond",
            headers={"Authorization": "Bearer nope"},
            json={"query": "Hi"},
        )
        authorized = client.post(
            "/agent/respond",
            headers={"Authorization": "Bearer ci-secret"},
            json={"query": "Hi"},
        )

    assert_equal(
        missing_auth.status_code,
        HTTPStatus.UNAUTHORIZED,
        "missing auth status",
    )
    assert_equal(wrong_auth.status_code, HTTPStatus.UNAUTHORIZED, "wrong auth status")
    assert_equal(authorized.status_code, HTTPStatus.OK, "authorized status")


def run() -> None:
    with patch.object(app_module, "run_agent", fake_run_agent):
        client = TestClient(app_module.app)
        test_health(client)
        test_agent_respond_accepts_supported_payload_shapes(client)
        test_agent_respond_rejects_invalid_payload(client)
        test_agent_respond_enforces_input_caps(client)
        test_memory_endpoints_reject_wildcard_user_id(client)
        test_agent_respond_enforces_shared_secret(client)


if __name__ == "__main__":
    run()
    print("Agent smoke tests passed.")
