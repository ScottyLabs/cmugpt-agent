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
from langchain_core.messages import SystemMessage
from langchain_core.tools import BaseTool

from agent import graph as graph_module
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
    response = client.get("/api/health")
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


def test_production_requires_database_and_shared_secret() -> None:
    with (
        temporary_env("AGENT_ENV", "production"),
        temporary_env("DATABASE_URL", None),
        temporary_env("AGENT_SHARED_SECRET", None),
    ):
        try:
            app_module._validate_runtime_configuration()
        except RuntimeError as exc:
            message = str(exc)
        else:
            raise AssertionError(
                "production starts without durable authenticated memory"
            )
    assert_true("DATABASE_URL" in message, "missing database is reported")
    assert_true("AGENT_SHARED_SECRET" in message, "missing shared secret is reported")

    with (
        temporary_env("AGENT_ENV", "production"),
        temporary_env("DATABASE_URL", "postgresql://example.invalid/cmugpt"),
        temporary_env("AGENT_SHARED_SECRET", "too-short"),
    ):
        try:
            app_module._validate_runtime_configuration()
        except RuntimeError as exc:
            assert_true("at least 32" in str(exc), "short secret is rejected")
        else:
            raise AssertionError("production accepts a weak shared secret")

    with (
        temporary_env("AGENT_ENV", "production"),
        temporary_env("DATABASE_URL", "postgresql://example.invalid/cmugpt"),
        temporary_env("AGENT_SHARED_SECRET", "s" * 32),
    ):
        app_module._validate_runtime_configuration()


def test_latency_planner_keeps_generic_turns_tool_free() -> None:
    assert_true(
        not graph_module._needs_data_tools("Reply with exactly one short sentence."),
        "generic turn skips MCP tools",
    )
    assert_true(
        not graph_module._needs_memory_tools("Reply with exactly one short sentence."),
        "generic turn skips memory tools",
    )
    assert_true(
        not graph_module._needs_memory_recall("Reply with exactly one short sentence."),
        "generic turn skips memory recall",
    )


def test_latency_planner_preserves_memory_tools() -> None:
    assert_true(
        graph_module._needs_memory_tools("Remember that I am vegetarian."),
        "explicit remember keeps memory tools",
    )
    assert_true(
        graph_module._needs_memory_recall("Where should I eat on campus?"),
        "personalized recommendation recalls memory",
    )
    assert_true(
        graph_module._needs_memory_recall("What animal do I like?"),
        "personal fact question recalls memory",
    )
    assert_true(
        graph_module._needs_memory_recall("What did I tell you earlier?"),
        "question about an earlier user statement recalls memory",
    )


def test_data_tool_gate_scans_recent_history() -> None:
    history = [
        {"role": "user", "content": "Where is Gates?"},
        {"role": "assistant", "content": "Gates is on Forbes."},
    ]
    assert_true(
        graph_module._needs_data_tools("what about Wean Hall?", history),
        "follow-up turn keeps data tools via history",
    )
    assert_true(
        not graph_module._needs_data_tools("what about Wean Hall?", None),
        "same text without history still skips tools",
    )
    small_talk = [{"role": "user", "content": "hello"}]
    assert_true(
        not graph_module._needs_data_tools("thanks!", small_talk),
        "non-data threads still skip tools",
    )


def test_force_latch_counts_memory_tool_rounds() -> None:
    from langchain_core.messages import (
        AIMessage,
        AnyMessage,
        HumanMessage,
        ToolMessage,
    )

    before: list[AnyMessage] = [
        HumanMessage(content="Remember that I love the Underground.")
    ]
    assert_true(
        not graph_module._had_tool_round(before),
        "no tool round before the first pass",
    )
    after: list[AnyMessage] = [
        *before,
        AIMessage(content="", tool_calls=[]),
        ToolMessage(content="Saved to memory: ...", tool_call_id="call_remember"),
    ]
    assert_true(
        graph_module._had_tool_round(after),
        "memory-only round releases the force latch",
    )


def test_every_llm_turn_uses_the_canonical_system_prompt() -> None:
    state = graph_module._initial_state(
        UserInput(query="Explain recursion briefly.", user_id="ci-user"),
        None,
        [],
    )
    first = state["messages"][0]
    assert_true(isinstance(first, SystemMessage), "first message is system policy")
    assert_equal(
        first.content,
        build_system_prompt([]),
        "no-tool turns use the canonical prompt",
    )
    assert_true(
        "Immutable rules (highest priority)" in str(first.content),
        "canonical security rules are present",
    )


class StubTool(BaseTool):
    """Minimal BaseTool stand-in: only `name` matters for group filtering."""

    name: str
    description: str = "stub"

    def _run(self, *_args: Any, **_kwargs: Any) -> str:
        return "stub"


STUB_TOOLS: list[BaseTool] = [
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
    # Keep smoke tests independent of a developer's local .env. The dedicated
    # shared-secret test below sets and verifies authentication explicitly.
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
        test_agent_respond_enforces_input_caps(client)
        test_memory_endpoints_reject_wildcard_user_id(client)
        test_agent_respond_enforces_shared_secret(client)
        test_production_requires_database_and_shared_secret()
        test_latency_planner_keeps_generic_turns_tool_free()
        test_latency_planner_preserves_memory_tools()
        test_data_tool_gate_scans_recent_history()
        test_force_latch_counts_memory_tool_rounds()
        test_every_llm_turn_uses_the_canonical_system_prompt()
        test_agent_respond_forwards_disabled_tools(client)
        test_agent_respond_rejects_malformed_disabled_tools(client)


if __name__ == "__main__":
    run()
    print("Agent smoke tests passed.")
