"""Deterministic CMU Maps validation and fallback inference.

The model's maps_show_map call (agent/map_tool.py) is the primary source of
the map attached to an answer, since the model reads phrasing and history no
pattern list can. This module is the deterministic layer around that decision.
It validates the model's codes against the catalog, builds the URL, and falls
back to regex inference over the latest query when the tool was not called.
Pure framework free logic with no LLM calls, unit testable under any model.
"""

import re
from typing import Any
from urllib.parse import quote

from .buildings import (
    KNOWN_CMU_LOCATIONS,
    LOCATION_ID_TO_LABEL,
    normalize,
)
from .guards import latest_user_text
from .schema import AgentResponse, CmuMaps

CMU_MAPS_BASE_URL = "https://maps.scottylabs.org"

# Local presentation tool the model calls to attach a map. The maps_ prefix
# puts it under the existing CMUMaps group toggle.
SHOW_MAP_TOOL_NAME = "maps_show_map"

CMU_MAPS_QUERY_RE = re.compile(
    r"\b("
    # The wheres alternative covers the apostrophe free typo, which a bare
    # word boundary match on where cannot reach.
    r"where'?s?|located|location|directions?|route|path|map|walk|walking|"
    r"get\s+to|go\s+to|take\s+me|show\s+me|navigate|how\s+far|from|between"
    r")\b",
    re.IGNORECASE,
)

# KNOWN_CMU_LOCATIONS maps aliases to codes and names, LOCATION_ID_TO_LABEL
# maps codes to names. Both derive from buildings.json in agent/buildings.py.
LOCATION_ID_RE = re.compile(r"\b[A-Z][A-Z0-9]{1,4}\b")
PAREN_LOCATION_RE = re.compile(
    r"(?P<label>[A-Z][A-Za-z0-9 '&.-]{1,80})\s*\((?P<id>[A-Z0-9]{2,5})\)"
)

MAP_FAILURE_CLAIM_RE = re.compile(
    r"\b("
    r"wasn['\u2019]?t\s+able|was\s+not\s+able|couldn['\u2019]?t|could\s+not|"
    r"cannot|can['\u2019]?t|unable|failed|fail|error|issue|problem|trouble|"
    r"didn['\u2019]?t\s+find|did\s+not\s+find"
    r")\b.{0,200}?\b("
    r"look(?:ing|ed)?\s*up|retriev\w*|find(?:ing)?|locat(?:e|ing|ions?)|"
    r"data|directions?|route|path|building"
    r")\b",
    re.IGNORECASE | re.DOTALL,
)

# Sentences that send the user to an external map or site for directions.
# The repair step drops them because an embedded map is already attached.
EXTERNAL_MAP_REDIRECT_RE = re.compile(
    r"\b(official\s+cmu\s+website|cmu\.edu|google\s+maps?)\b",
    re.IGNORECASE,
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
    lowered = normalize(cleaned)

    for alias, loc_id, label in KNOWN_CMU_LOCATIONS:
        if re.search(rf"\b{re.escape(alias)}\b", lowered):
            return loc_id, label

    # Bare uppercase tokens count as building IDs only when the catalog knows
    # them, otherwise arbitrary capitalized words in free text would hijack
    # the answer with a bogus map.
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
        # Covers origins stated as the speaker's position, as in I'm at Gates,
        # how do I get to Baker. The source capture stops at punctuation so it
        # does not swallow the question.
        re.compile(
            r"\b(?:i'?m|i\s+am|we'?re|we\s+are)\s+(?:at|in|near|by)\s+"
            r"(?P<src>[^,.;?!]+?)[,.;]?\s.*?"
            r"\b(?:get|go|walk|head|come)\s+(?:over\s+)?to\s+"
            r"(?P<dest>.+?)(?:[?!.,;:]|$)",
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


# Trailing room number as in Wean 5310. Indoor data sits behind auth, so the
# room only selects which building to show.
_ROOM_SUFFIX_RE = re.compile(r"\s+[A-Za-z]?\d{2,4}[A-Za-z]?$")
_FILLER_RE = re.compile(
    r"^(?:please|hey|hi|ok(?:ay)?)[\s,]+|[\s,]+(?:please|thanks|thank\s+you)$",
    re.IGNORECASE,
)


def _bare_location_query(query: str) -> tuple[str, str | None] | None:
    """Resolve a message that is only a location, such as newell simon hall.

    The intent gate needs a verb, so a bare building name never reaches the
    target patterns, yet typing only a building is a request to see it. To
    stay safe without the verb, the whole cleaned message must equal one alias
    exactly. A location inside a longer sentence still needs explicit intent.
    """
    cleaned = _FILLER_RE.sub("", _clean_location_phrase(query))
    cleaned = _ROOM_SUFFIX_RE.sub("", cleaned).strip()
    if not cleaned or len(cleaned) < 2:
        return None
    lowered = normalize(cleaned)
    for alias, loc_id, label in KNOWN_CMU_LOCATIONS:
        if lowered == alias:
            return loc_id, label
    return _location_from_id(cleaned)


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


def _catalog_location(loc_id: Any) -> tuple[str, str | None] | None:
    """A location only when the id is a code the catalog actually contains."""
    loc = _location_from_id(loc_id) if isinstance(loc_id, str) else None
    return loc if loc and loc[0] in LOCATION_ID_TO_LABEL else None


def _maps_from_show_map(call: dict[str, Any]) -> CmuMaps | None:
    """The model's map decision when this call is a valid maps_show_map.

    The tool schema constrains codes, but these args are raw model output
    recorded before validation, so membership is checked again. A hallucinated
    code degrades to fallback inference instead of a URL the map rejects.
    """
    if call.get("name") != SHOW_MAP_TOOL_NAME:
        return None
    args = _tool_arguments(call)
    dest = _catalog_location(args.get("destination"))
    if not dest:
        return None
    src = _catalog_location(args.get("origin"))
    if src and src[0] != dest[0]:
        return _maps_payload_for_directions(src, dest)
    return _maps_payload_for_location(dest)


def _infer_cmu_maps(
    messages: list[dict[str, Any]],
    tool_invocations: list[dict[str, Any]],
) -> CmuMaps:
    # The model's explicit decision outranks everything below, including the
    # intent gate, since it judges phrasing and history no pattern can. The
    # latest call wins because the model may correct itself.
    for call in reversed(tool_invocations):
        decided = _maps_from_show_map(call)
        if decided:
            return decided

    query = latest_user_text(messages)
    if not query:
        return CmuMaps()
    if not CMU_MAPS_QUERY_RE.search(query):
        bare = _bare_location_query(query)
        return _maps_payload_for_location(bare) if bare else CmuMaps()

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


def query_has_map_intent(query: str) -> bool:
    """True when the query alone can place a building or route on the map.

    Decides whether the answer is buffered, so a false failure claim never
    streams ahead of a map already known to be buildable.
    """
    if not query:
        return False
    if not CMU_MAPS_QUERY_RE.search(query):
        return _bare_location_query(query) is not None
    if _direction_locations_from_query(query):
        return True
    return _target_location_from_query(query) is not None


def _cmu_maps_success_text(cmu_maps: CmuMaps) -> str:
    """Minimal route specific pointer to the map.

    Fallback only, used when the model wrongly claims a place could not be
    found although a map exists. Names the requested locations rather than any
    hardcoded route.
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


def _strip_false_map_failure(text: str) -> str:
    """Drop lines that falsely claim a failed lookup or push an external map.

    Operates line by line so real directions survive. Only the offending prose
    lines are removed.
    """
    kept: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped and (
            MAP_FAILURE_CLAIM_RE.search(stripped)
            or EXTERNAL_MAP_REDIRECT_RE.search(stripped)
        ):
            continue
        kept.append(line)
    cleaned = "\n".join(kept)
    return re.sub(r"\n{3,}", "\n\n", cleaned).strip()


_LOW_VALUE_REMAINDER_RE = re.compile(
    r"^\s*(?:let me know|feel free|hope (?:this|that) helps|"
    r"is there anything else|happy to help|anything else)",
    re.IGNORECASE,
)


def _is_low_value_remainder(text: str) -> bool:
    """True when what survives scrubbing is empty or just a closing pleasantry."""
    return len(text.strip()) < 15 or bool(_LOW_VALUE_REMAINDER_RE.match(text))


def _repair_false_map_failure(text: str, cmu_maps: CmuMaps) -> str:
    """Swap a false map failure claim for the correct map pointer.

    Keeps useful directions the model produced and falls back to the pointer
    alone when nothing of value survives scrubbing.
    """
    success = _cmu_maps_success_text(cmu_maps)
    scrubbed = _strip_false_map_failure(text)
    if scrubbed and not _is_low_value_remainder(scrubbed):
        return f"{success}\n\n{scrubbed}"
    return success


def _apply_cmu_maps_guard(
    parsed: AgentResponse,
    messages: list[dict[str, Any]],
    tool_invocations: list[dict[str, Any]],
) -> AgentResponse:
    inferred = _infer_cmu_maps(messages, tool_invocations)
    if inferred.url:
        parsed.cmu_maps = inferred
        # The validated map is authoritative. If the text still claims the
        # lookup failed, repair it so the user never sees the contradiction.
        if MAP_FAILURE_CLAIM_RE.search(parsed.response_text or ""):
            parsed.response_text = _repair_false_map_failure(
                parsed.response_text or "", inferred
            )
    elif not _is_valid_cmu_maps_url(parsed.cmu_maps.url):
        parsed.cmu_maps = CmuMaps()
    return parsed
