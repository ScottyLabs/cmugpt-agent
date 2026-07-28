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
from langchain_core.tools import BaseTool

from agent.mcp_tools import disabled_group_labels, filter_tools, tool_group
from agent.prompts import build_system_prompt
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
    disabled_tools: list[str] | None = None,
) -> AgentResponse:
    history_count = len(message_history or [])
    off = ",".join(disabled_tools or [])
    return AgentResponse(
        thought=Thought(reasoning=f"smoke test via {model}", confidence=0.91),
        action=ActionType.RESPOND,
        response_text=(
            f"**Echo:** {user_input.query} "
            f"(user={user_input.user_id or 'anonymous'}, history={history_count}, "
            f"off=[{off}])"
        ),
        services_used=[],
    )


def test_health(client: TestClient) -> None:
    response = client.get("/health")
    assert_equal(response.status_code, HTTPStatus.OK, "health status")
    assert_equal(response.json(), {"status": "ok"}, "health payload")


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


class StubTool(BaseTool):
    """Minimal BaseTool stand-in: only `name` matters for group filtering."""

    name: str
    description: str = "stub"

    def _run(self, *_args: Any, **_kwargs: Any) -> str:
        return "stub"


STUB_TOOLS = [
    StubTool(name=name)
    for name in (
        "maps_search_buildings",
        "maps_get_path",
        "eats_get_locations_open_now",
        "courses_search_courses_by_query",
        "guide_search_guide",
        "ungrouped_tool",
    )
]


def tool_names(tools: list[BaseTool]) -> list[str]:
    return [tool.name for tool in tools]


def test_tool_group_mapping() -> None:
    assert_equal(tool_group("maps_get_path"), "maps", "maps group")
    assert_equal(tool_group("eats_get_location_hours"), "eats", "eats group")
    assert_equal(tool_group("courses_fetch_course_by_id"), "courses", "courses group")
    assert_equal(tool_group("guide_search_guide"), "guide", "guide group")
    assert_equal(tool_group("mapsomething"), None, "prefix needs underscore")
    assert_equal(tool_group("ungrouped_tool"), None, "unknown group")


def test_filter_tools_drops_only_disabled_groups() -> None:
    assert_equal(
        tool_names(filter_tools(STUB_TOOLS, None)),
        tool_names(STUB_TOOLS),
        "nothing disabled keeps every tool",
    )
    assert_equal(
        tool_names(filter_tools(STUB_TOOLS, [])),
        tool_names(STUB_TOOLS),
        "empty list keeps every tool",
    )
    assert_equal(
        tool_names(filter_tools(STUB_TOOLS, ["maps"])),
        [
            "eats_get_locations_open_now",
            "courses_search_courses_by_query",
            "guide_search_guide",
            "ungrouped_tool",
        ],
        "maps off drops both maps tools",
    )
    assert_equal(
        tool_names(filter_tools(STUB_TOOLS, ["eats", "courses"])),
        [
            "maps_search_buildings",
            "maps_get_path",
            "guide_search_guide",
            "ungrouped_tool",
        ],
        "two groups off",
    )
    assert_equal(
        tool_names(filter_tools(STUB_TOOLS, ["maps", "eats", "courses", "guide"])),
        ["ungrouped_tool"],
        "every known group off",
    )


def test_filter_tools_accepts_labels_and_ignores_junk() -> None:
    assert_equal(
        tool_names(filter_tools(STUB_TOOLS, ["CMUMaps"])),
        tool_names(filter_tools(STUB_TOOLS, ["maps"])),
        "UI label behaves like the id",
    )
    assert_equal(
        tool_names(filter_tools(STUB_TOOLS, ["", "nope", "maps_get_path"])),
        tool_names(STUB_TOOLS),
        "unknown ids never disable anything",
    )
    assert_equal(
        disabled_group_labels(["eats", "maps"]),
        ["CMUMaps", "CMUEats"],
        "labels come back in a stable order",
    )


def test_prompt_hides_disabled_tools() -> None:
    enabled_prompt = build_system_prompt(STUB_TOOLS)
    maps_off = filter_tools(STUB_TOOLS, ["maps"])
    disabled_prompt = build_system_prompt(maps_off, ["maps"])

    assert_true("maps_get_path" in enabled_prompt, "catalog lists maps when on")
    assert_true("maps_get_path" not in disabled_prompt, "catalog hides maps when off")
    assert_true(
        "switched CMUMaps OFF" in disabled_prompt,
        "directions section covers the switched-off map",
    )
    assert_true(
        "**CMUMaps**" in disabled_prompt,
        "prompt names the switched-off tool so the model can explain itself",
    )
    assert_true(
        "eats_get_locations_open_now" in disabled_prompt,
        "unrelated groups stay available",
    )


def test_agent_respond_forwards_disabled_tools(client: TestClient) -> None:
    response = client.post(
        "/agent/respond",
        json={"query": "Where is Gates?", "disabled_tools": ["maps", "eats"]},
    )
    payload = response.json()

    assert_equal(response.status_code, HTTPStatus.OK, "disabled tools status")
    assert_true(
        "off=[maps,eats]" in payload["response_text"], "disabled tools forwarded"
    )


def test_agent_respond_rejects_malformed_disabled_tools(client: TestClient) -> None:
    response = client.post(
        "/agent/respond",
        json={"query": "Where is Gates?", "disabled_tools": "maps"},
    )
    payload = response.json()

    assert_equal(
        response.status_code,
        HTTPStatus.BAD_REQUEST,
        "malformed disabled_tools status",
    )
    assert_true(
        "disabled_tools" in payload["detail"], "malformed disabled_tools detail"
    )


def run() -> None:
    # Importing the app loads .env, so a developer with AGENT_SHARED_SECRET set
    # locally would otherwise get 401s on the unauthenticated cases below.
    with temporary_env("AGENT_SHARED_SECRET", None):
        run_tests()


def run_tests() -> None:
    test_tool_group_mapping()
    test_filter_tools_drops_only_disabled_groups()
    test_filter_tools_accepts_labels_and_ignores_junk()
    test_prompt_hides_disabled_tools()
    with patch.object(app_module, "run_agent", fake_run_agent):
        client = TestClient(app_module.app)
        test_health(client)
        test_agent_respond_accepts_supported_payload_shapes(client)
        test_agent_respond_rejects_invalid_payload(client)
        test_agent_respond_enforces_shared_secret(client)
        test_agent_respond_forwards_disabled_tools(client)
        test_agent_respond_rejects_malformed_disabled_tools(client)


if __name__ == "__main__":
    run()
    print("Agent smoke tests passed.")
