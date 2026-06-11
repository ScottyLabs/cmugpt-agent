"""Deterministic CMU Maps inference.

Extracts CMU Maps building/room IDs and builds embeddable map URLs from the
user's query and any tool invocations made during the turn. This is pure,
framework-agnostic logic: it never calls the LLM, so the agent's map links stay
deterministic and unit-testable regardless of the model used.
"""

import re
from typing import Any
from urllib.parse import quote

from .guards import latest_user_text
from .schema import AgentResponse, CmuMaps

CMU_MAPS_BASE_URL = "https://maps.scottylabs.org"

CMU_MAPS_QUERY_RE = re.compile(
    r"\b("
    r"where|located|location|directions?|route|path|map|walk|walking|"
    r"get\s+to|go\s+to|from|between"
    r")\b",
    re.IGNORECASE,
)

KNOWN_CMU_LOCATIONS: list[tuple[str, str, str]] = [
    ("margaret morrison carnegie hall", "MM", "Margaret Morrison Carnegie Hall"),
    ("margaret morrison", "MM", "Margaret Morrison Carnegie Hall"),
    ("mudge house", "MUD", "Mudge House"),
    ("mudge", "MUD", "Mudge House"),
    ("tepper school of business", "TEP", "Tepper School of Business"),
    ("tepper building", "TEP", "Tepper School of Business"),
    ("tepper", "TEP", "Tepper School of Business"),
    ("tsb", "TEP", "Tepper School of Business"),
    ("gates and hillman centers", "GHC", "Gates & Hillman Centers"),
    ("gates & hillman centers", "GHC", "Gates & Hillman Centers"),
    ("gates hillman centers", "GHC", "Gates & Hillman Centers"),
    ("gates hillman center", "GHC", "Gates & Hillman Centers"),
    ("gates hillman", "GHC", "Gates & Hillman Centers"),
    ("hillman centers", "GHC", "Gates & Hillman Centers"),
    ("hillman center", "GHC", "Gates & Hillman Centers"),
    ("gates", "GHC", "Gates & Hillman Centers"),
    ("ghc", "GHC", "Gates & Hillman Centers"),
    ("cohon university center", "CUC", "Cohon University Center"),
    ("cohon center", "CUC", "Cohon University Center"),
    ("cohon", "CUC", "Cohon University Center"),
    ("university center", "CUC", "Cohon University Center"),
    ("cuc", "CUC", "Cohon University Center"),
    ("uc", "CUC", "Cohon University Center"),
    ("wean hall", "WEH", "Wean Hall"),
    ("wean", "WEH", "Wean Hall"),
    ("doherty hall", "DH", "Doherty Hall"),
    ("doherty", "DH", "Doherty Hall"),
    ("porter hall", "PH", "Porter Hall"),
    ("porter", "PH", "Porter Hall"),
    ("baker hall", "BH", "Baker Hall"),
    ("baker", "BH", "Baker Hall"),
    ("hunt library", "HL", "Hunt Library"),
    ("hunt", "HL", "Hunt Library"),
    ("scott hall", "SC", "Scott Hall"),
    ("hamerschlag hall", "HH", "Hamerschlag Hall"),
    ("hamerschlag", "HH", "Hamerschlag Hall"),
    ("newell-simon hall", "NSH", "Newell-Simon Hall"),
    ("newell simon", "NSH", "Newell-Simon Hall"),
    ("resnik house", "RES", "Resnik House"),
    ("resnik", "RES", "Resnik House"),
]

LOCATION_ID_TO_LABEL: dict[str, str] = {}
for _, _loc_id, _label in KNOWN_CMU_LOCATIONS:
    LOCATION_ID_TO_LABEL.setdefault(_loc_id, _label)
LOCATION_ID_RE = re.compile(r"\b[A-Z][A-Z0-9]{1,4}\b")
PAREN_LOCATION_RE = re.compile(
    r"(?P<label>[A-Z][A-Za-z0-9 '&.-]{1,80})\s*\((?P<id>[A-Z0-9]{2,5})\)"
)

MAP_FAILURE_CLAIM_RE = re.compile(
    r"\b("
    r"wasn['’]?t\s+able|was\s+not\s+able|couldn['’]?t|could\s+not|"
    r"unable|failed|didn['’]?t\s+find|did\s+not\s+find"
    r")\b.{0,240}\b("
    r"location|building|map|directions?|path|route|tool|tools|retrieve"
    r")\b",
    re.IGNORECASE | re.DOTALL,
)


def _clean_location_phrase(text: str) -> str:
    cleaned = re.sub(r"[?!.,;:]+$", "", text.strip())
    cleaned = re.sub(
        r"\b(on|at|in)\s+(?:the\s+)?(?:cmu|carnegie mellon)\s+campus\b",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    return re.sub(r"\s+", " ", cleaned).strip(" \"'")


def _location_from_id(loc_id: str | None) -> tuple[str, str | None] | None:
    if not isinstance(loc_id, str):
        return None
    normalized = loc_id.strip().upper()
    if not normalized:
        return None
    match = LOCATION_ID_RE.fullmatch(normalized)
    if not match:
        return None
    return normalized, LOCATION_ID_TO_LABEL.get(normalized)


def _location_from_text(text: str | None) -> tuple[str, str | None] | None:
    if not isinstance(text, str) or not text.strip():
        return None
    cleaned = _clean_location_phrase(text)
    lowered = cleaned.lower()

    for alias, loc_id, label in KNOWN_CMU_LOCATIONS:
        if re.search(rf"\b{re.escape(alias)}\b", lowered):
            return loc_id, label

    # Bare uppercase tokens are only treated as building IDs when they are known
    # CMU Maps IDs. Without this guard, arbitrary capitalized words in free text
    # (e.g. "DAN", "SIO") would be mistaken for locations and could hijack the
    # answer with a bogus map/directions override.
    explicit_id = LOCATION_ID_RE.search(cleaned)
    if explicit_id:
        loc = _location_from_id(explicit_id.group(0))
        if loc and loc[0] in LOCATION_ID_TO_LABEL:
            return loc
    return None


def _location_from_tool_result(result: str | None) -> tuple[str, str | None] | None:
    if not isinstance(result, str):
        return None
    for match in PAREN_LOCATION_RE.finditer(result):
        loc = _location_from_id(match.group("id"))
        if loc:
            return loc[0], match.group("label").strip()
    return _location_from_text(result)


def _direction_locations_from_query(
    query: str,
) -> tuple[tuple[str, str | None], tuple[str, str | None]] | None:
    patterns = [
        re.compile(
            r"\bfrom\s+(?P<src>.+?)\s+to\s+(?P<dest>.+?)(?:[?!.,;:]|$)",
            re.IGNORECASE,
        ),
        re.compile(
            r"\bto\s+(?P<dest>.+?)\s+from\s+(?P<src>.+?)(?:[?!.,;:]|$)",
            re.IGNORECASE,
        ),
        re.compile(
            r"\bbetween\s+(?P<src>.+?)\s+and\s+(?P<dest>.+?)(?:[?!.,;:]|$)",
            re.IGNORECASE,
        ),
        re.compile(
            r"^\s*(?P<dest>[A-Za-z0-9 '&.-]+?)\s+from\s+"
            r"(?P<src>[A-Za-z0-9 '&.-]+?)(?:[?!.,;:]|$)",
            re.IGNORECASE,
        ),
    ]
    for pattern in patterns:
        match = pattern.search(query)
        if not match:
            continue
        src = _location_from_text(match.group("src"))
        dest = _location_from_text(match.group("dest"))
        if src and dest:
            return src, dest
    return None


def _target_location_from_query(query: str) -> tuple[str, str | None] | None:
    patterns = [
        re.compile(
            r"\bwhere\s+is\s+(?P<target>.+?)(?:[?!.,;:]|$)",
            re.IGNORECASE,
        ),
        re.compile(
            r"\bwhere(?:'s|s)\s+(?P<target>.+?)(?:[?!.,;:]|$)",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(?:show|find|locate)\s+(?:me\s+)?(?P<target>.+?)(?:[?!.,;:]|$)",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(?:directions?|route|path|walk|walking|get\s+to|go\s+to)\s+"
            r"(?:to\s+)?(?P<target>.+?)(?:[?!.,;:]|$)",
            re.IGNORECASE,
        ),
    ]
    for pattern in patterns:
        match = pattern.search(query)
        if not match:
            continue
        target = _location_from_text(match.group("target"))
        if target:
            return target
    return _location_from_text(query)


def _location_url(target: str) -> str:
    encoded = quote(target, safe="")
    return f"{CMU_MAPS_BASE_URL}/{encoded}?dst={encoded}"


def _directions_url(src: str, dest: str) -> str:
    encoded_target = quote(dest, safe="")
    encoded_src = quote(src, safe="")
    encoded_dest = quote(dest, safe="")
    return f"{CMU_MAPS_BASE_URL}/{encoded_target}?src={encoded_src}&dst={encoded_dest}"


def _maps_payload_for_location(
    target: tuple[str, str | None],
) -> CmuMaps:
    loc_id, label = target
    return CmuMaps(
        url=_location_url(loc_id),
        mode="location",
        target=loc_id,
        target_label=label,
        src=None,
        src_label=None,
        dest=loc_id,
        dest_label=label,
    )


def _maps_payload_for_directions(
    src: tuple[str, str | None],
    dest: tuple[str, str | None],
) -> CmuMaps:
    src_id, src_label = src
    dest_id, dest_label = dest
    return CmuMaps(
        url=_directions_url(src_id, dest_id),
        mode="directions",
        target=dest_id,
        target_label=dest_label,
        src=src_id,
        src_label=src_label,
        dest=dest_id,
        dest_label=dest_label,
    )


def _is_valid_cmu_maps_url(url: str | None) -> bool:
    return isinstance(url, str) and url.startswith(f"{CMU_MAPS_BASE_URL}/")


def _tool_arguments(call: dict[str, Any]) -> dict[str, Any]:
    arguments = call.get("arguments")
    return arguments if isinstance(arguments, dict) else {}


def _infer_cmu_maps(
    messages: list[dict[str, Any]],
    tool_invocations: list[dict[str, Any]],
) -> CmuMaps:
    query = latest_user_text(messages)
    if not query or not CMU_MAPS_QUERY_RE.search(query):
        return CmuMaps()

    for call in tool_invocations:
        name = call.get("name")
        args = _tool_arguments(call)
        if name in {"maps_get_path", "maps_distance_between"}:
            src = _location_from_id(args.get("start_id"))
            dest = _location_from_id(args.get("end_id"))
            if src and dest:
                return _maps_payload_for_directions(src, dest)

    query_direction = _direction_locations_from_query(query)
    if query_direction:
        src, dest = query_direction
        return _maps_payload_for_directions(src, dest)

    for call in tool_invocations:
        name = call.get("name")
        if name not in {"maps_search_buildings", "maps_list_possible_locations"}:
            continue
        args = _tool_arguments(call)
        target = _location_from_tool_result(call.get("result")) or _location_from_text(
            args.get("query")
        )
        if target:
            return _maps_payload_for_location(target)

    target = _target_location_from_query(query)
    if target:
        return _maps_payload_for_location(target)
    return CmuMaps()


def _cmu_maps_success_text(cmu_maps: CmuMaps) -> str:
    """Minimal, route-specific pointer to the map.

    This is a fallback only: the primary directions/answer come from the model,
    which describes the specific route (and uses the routing tool when it is
    available). This text is used when the model wrongly claims it could not
    find a place even though we have a map to show. It references the actual
    requested locations rather than any hardcoded route.
    """
    if cmu_maps.mode == "directions":
        src = cmu_maps.src_label or cmu_maps.src or "your starting point"
        dest = cmu_maps.dest_label or cmu_maps.dest or "your destination"
        src_id = f" ({cmu_maps.src})" if cmu_maps.src else ""
        dest_id = f" ({cmu_maps.dest})" if cmu_maps.dest else ""
        return (
            f"Here's the suggested route from **{src}{src_id}** to "
            f"**{dest}{dest_id}** on the Carnegie Mellon University campus. "
            "Follow the highlighted path on the map below."
        )

    target = cmu_maps.target_label or cmu_maps.target or "that location"
    return f"Here's **{target}** on CMU Maps."


def _apply_cmu_maps_guard(
    parsed: AgentResponse,
    messages: list[dict[str, Any]],
    tool_invocations: list[dict[str, Any]],
) -> AgentResponse:
    inferred = _infer_cmu_maps(messages, tool_invocations)
    if inferred.url:
        parsed.cmu_maps = inferred
        # Keep the model's route-specific prose. Only override when it wrongly
        # claims it couldn't find the place while we do have a map to show.
        if MAP_FAILURE_CLAIM_RE.search(parsed.response_text or ""):
            parsed.response_text = _cmu_maps_success_text(inferred)
    elif not _is_valid_cmu_maps_url(parsed.cmu_maps.url):
        parsed.cmu_maps = CmuMaps()
    return parsed
