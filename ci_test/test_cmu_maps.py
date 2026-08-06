"""Unit tests for deterministic CMU Maps inference in agent/cmu_maps.py.

Pure logic tests with no LLM, MCP, or network. Each case pairs a realistic
query with the map it must produce or the required abstention. Abstentions
matter as much as hits, since a map on every building mention is worse than
waiting for location intent.

Run directly with uv run python ci_test/test_cmu_maps.py.
"""

from typing import Any

from agent.cmu_maps import _infer_cmu_maps, query_has_map_intent


def _messages(query: str) -> list[dict[str, Any]]:
    return [{"role": "user", "content": query}]


def _summary(query: str, tool_invocations: list[dict[str, Any]] | None = None) -> str:
    maps = _infer_cmu_maps(_messages(query), tool_invocations or [])
    if not maps.url:
        return "none"
    if maps.mode == "directions":
        return f"{maps.src}->{maps.dest}"
    return f"@{maps.target}"


def assert_case(query: str, expected: str, **kwargs: Any) -> None:
    actual = _summary(query, **kwargs)
    if actual != expected:
        raise AssertionError(f"{query!r}: expected {expected}, got {actual}")


DIRECTIONS_CASES = [
    ("How do I get from Mudge House to Stever House?", "MUD->STE"),
    ("how do i get from mudge to stever", "MUD->STE"),
    ("directions from gates to the uc", "GHC->CUC"),
    ("How do I get to Tepper from Morewood Gardens?", "MOR->TEP"),
    ("route between Porter Hall and Hamburg Hall", "PH->HBH"),
    ("walk from GHC to WEH", "GHC->WEH"),
    # Origin stated as the speaker's position rather than with from.
    ("I'm at Gates, how do I get to Baker?", "GHC->BH"),
    # A distance question still yields the route between its endpoints.
    ("how far is Gates from Donner?", "DON->GHC"),
]

LOCATION_CASES = [
    ("Where is Gates?", "@GHC"),
    ("where's baker hall", "@BH"),
    # Apostrophe free typo that a bare word boundary match never catches.
    ("wheres hunt library", "@HL"),
    ("Take me to Scaife Hall", "@SH"),
    ("can you show me where Posner Hall is", "@POS"),
    ("Where is Warner Hall? I need to pay my bill", "@WH"),
    # Both words are shared with MMA, so only a curated alias resolves this.
    ("Where is Margaret Morrison?", "@MM"),
    ("maggie mo", "@MM"),
    # A named interior maps to the building that contains it.
    ("Where is Sorrells Library?", "@WEH"),
    # Room data sits behind auth, the building is what the map can show.
    ("Where is GHC 4301?", "@GHC"),
    # A message that is only a location needs no intent verb.
    ("newell simon hall", "@NSH"),
    ("Wean 5310", "@WEH"),
]

# Queries that must not produce a map: no location intent, off campus
# targets, or a building named only in passing.
ABSTAIN_CASES = [
    "Is La Prima open?",
    "What time does the UC gym close?",
    "how do i get to giant eagle from campus",
    "show me my grades",
    "how far along is course registration?",
    "what classes are in Gates this semester",
    "thanks!",
    "my andrew id is jc965",
]


def test_directions() -> None:
    for query, expected in DIRECTIONS_CASES:
        assert_case(query, expected)


def test_locations() -> None:
    for query, expected in LOCATION_CASES:
        assert_case(query, expected)


def test_abstentions() -> None:
    for query in ABSTAIN_CASES:
        assert_case(query, "none")


def test_tool_invocations_win_over_query_text() -> None:
    # A routing tool call carries resolved IDs and outranks query text parsing.
    calls = [
        {
            "name": "maps_get_path",
            "arguments": {"start_id": "TEP", "end_id": "MM"},
        }
    ]
    assert_case(
        "How do I get from the business school to Maggie Mo?",
        "TEP->MM",
        tool_invocations=calls,
    )


def _show_map_call(destination: Any, origin: Any = None) -> dict[str, Any]:
    args: dict[str, Any] = {"destination": destination}
    if origin is not None:
        args["origin"] = origin
    return {"name": "maps_show_map", "arguments": args, "result": "Map attached."}


def test_show_map_decides_location() -> None:
    # The model's call wins even when no pattern can see intent in the query.
    assert_case(
        "any coffee around maggie mo?",
        "@MM",
        tool_invocations=[_show_map_call("MM")],
    )


def test_show_map_decides_directions() -> None:
    # Phrasing the regexes cannot parse, resolved by the model from context.
    assert_case(
        "im at mudge rn, whats the fastest way to my 9am in wean",
        "MUD->WEH",
        tool_invocations=[_show_map_call("WEH", origin="MUD")],
    )


def test_show_map_same_origin_and_destination_is_location() -> None:
    assert_case(
        "where is gates",
        "@GHC",
        tool_invocations=[_show_map_call("GHC", origin="GHC")],
    )


def test_show_map_invalid_codes_fall_back() -> None:
    # A hallucinated destination degrades to regex inference of the query.
    assert_case(
        "Where is Gates?",
        "@GHC",
        tool_invocations=[_show_map_call("ZZZ")],
    )
    # A hallucinated origin degrades to a location map, not a broken route.
    assert_case(
        "Where is Gates?",
        "@GHC",
        tool_invocations=[_show_map_call("GHC", origin="ZZZ")],
    )
    # Junk argument shapes are ignored entirely.
    assert_case(
        "Where is Gates?",
        "@GHC",
        tool_invocations=[_show_map_call(None), _show_map_call(123)],
    )


def test_show_map_last_call_wins() -> None:
    # The model corrected itself mid turn, the final decision stands.
    assert_case(
        "show my route",
        "MUD->STE",
        tool_invocations=[
            _show_map_call("MM"),
            _show_map_call("STE", origin="MUD"),
        ],
    )


def test_show_map_outranks_routing_tool_ids() -> None:
    # Mid reasoning data lookups may involve other buildings. The explicit
    # presentation call is what the user should see.
    calls = [
        {"name": "maps_get_path", "arguments": {"start_id": "TEP", "end_id": "MM"}},
        _show_map_call("GHC", origin="MUD"),
    ]
    assert_case("walk me over", "MUD->GHC", tool_invocations=calls)


def test_show_map_tool_schema_matches_catalog() -> None:
    from agent.buildings import LOCATION_ID_TO_LABEL
    from agent.map_tool import BuildingCode, build_show_map_tool

    # StrEnum members are their own values. str() avoids ty tripping on the
    # runtime built enum's attributes.
    codes = {str(member) for member in BuildingCode}
    if codes != set(LOCATION_ID_TO_LABEL):
        raise AssertionError("BuildingCode enum diverged from the catalog")
    tool = build_show_map_tool()
    if tool.name != "maps_show_map":
        raise AssertionError(f"unexpected tool name {tool.name!r}")
    result = tool.invoke({"destination": "STE", "origin": "MUD"})
    if "Mudge House (MUD)" not in result or "Stever House (STE)" not in result:
        raise AssertionError(f"unexpected tool result {result!r}")


def test_map_intent_matches_inference() -> None:
    # query_has_map_intent gates answer buffering. It must say yes wherever
    # inference maps and no wherever inference abstains.
    for query, _ in DIRECTIONS_CASES + LOCATION_CASES:
        if not query_has_map_intent(query):
            raise AssertionError(f"{query!r}: inference maps, intent says no")
    for query in ABSTAIN_CASES:
        if query_has_map_intent(query):
            raise AssertionError(f"{query!r}: intent says yes, inference abstains")


def run() -> None:
    test_directions()
    test_locations()
    test_abstentions()
    test_tool_invocations_win_over_query_text()
    test_show_map_decides_location()
    test_show_map_decides_directions()
    test_show_map_same_origin_and_destination_is_location()
    test_show_map_invalid_codes_fall_back()
    test_show_map_last_call_wins()
    test_show_map_outranks_routing_tool_ids()
    test_show_map_tool_schema_matches_catalog()
    test_map_intent_matches_inference()


if __name__ == "__main__":
    run()
    print("CMU Maps inference tests passed.")
