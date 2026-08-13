"""Building catalog loaded from the committed buildings.json.

buildings.json is a flat code-to-name map and nothing else. Aliases, the
reverse index, and single-word matches are derived at load time, which keeps
the data file small and editable while still resolving "scott", "scott
hall", and "donner". The maps app addresses buildings by short code, so
resolution always returns both code and name.
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

# The repository root is searched first, matching the deployed uvicorn
# working directory, then the package directory for wheel builds that ship
# buildings.json as package data.
_BUILDINGS_CANDIDATES = (
    Path(__file__).resolve().parents[1] / "buildings.json",
    Path(__file__).resolve().parent / "buildings.json",
)


def _buildings_path() -> Path | None:
    return next((p for p in _BUILDINGS_CANDIDATES if p.is_file()), None)


# Trailing words generic enough that the distinctive part alone should also
# resolve, so "scott hall" additionally yields "scott".
_GENERIC_SUFFIX = {
    "hall",
    "house",
    "center",
    "centers",
    "building",
    "buildings",
    "complex",
    "tower",
    "towers",
    "apartments",
    "apartment",
    "gardens",
    "garden",
    "wing",
    "annex",
    "residence",
    "institute",
    "library",
    "quad",
}
# Words too generic to stand alone as an alias.
_STOPWORDS = _GENERIC_SUFFIX | {
    "the",
    "of",
    "and",
    "university",
    "college",
    "school",
    "north",
    "south",
    "east",
    "west",
    "a",
    "at",
    "for",
    "s",
}
# Curated nicknames and disambiguations, each taking precedence over
# generated mappings. "hamerschlag" alone denotes the academic hall, not the
# dorm. "margaret morrison" is listed because both of its words are shared
# with MMA, so generation drops them and no two-word alias would otherwise
# exist.
_EXTRA_ALIASES: dict[str, str] = {
    "uc": "CUC",
    "university center": "CUC",
    "tsb": "TEP",
    "hamerschlag": "HH",
    "margaret morrison": "MM",
    "maggie mo": "MM",
    # Well-known interiors that students name directly, each mapping to the
    # containing building, the closest thing the map can display.
    "sorrells": "WEH",
}

# Minimal fallback used when buildings.json is absent or corrupt.
_FALLBACK_CODE_TO_NAME = {
    "GHC": "Gates & Hillman Centers",
    "CUC": "Cohon University Center",
    "WEH": "Wean Hall",
    "DH": "Doherty Hall",
    "BH": "Baker Hall",
    "PH": "Porter Hall",
    "NSH": "Newell-Simon Hall",
    "SC": "Scott Hall",
    "HH": "Hamerschlag Hall",
    "TEP": "Tepper Building",
    "HL": "Hunt Library",
    "MM": "Margaret Morrison Carnegie Hall",
    "MUD": "Mudge House",
    "DON": "Donner House",
}


def normalize(text: str) -> str:
    """Lowercase, fold ampersand to "and", and reduce punctuation to spaces.

    Applied to both alias generation and query matching so the two always
    operate in the same character space.
    """
    lowered = text.lower().replace("&", " and ")
    lowered = re.sub(r"[^a-z0-9 ]", " ", lowered)
    return re.sub(r"\s+", " ", lowered).strip()


def _build_index(
    code_to_name: dict[str, str],
) -> tuple[list[tuple[str, str, str]], dict[str, str]]:
    alias_to_codes: dict[str, set[str]] = {}
    single_word_to_codes: dict[str, set[str]] = {}

    def add(alias: str, code: str) -> None:
        if alias:
            alias_to_codes.setdefault(alias, set()).add(code)

    for code, name in code_to_name.items():
        normalized = normalize(name)
        add(normalized, code)  # full name
        if len(code) >= 3:
            # Two-character codes collide with ordinary words, so only longer
            # codes register as their own alias.
            add(code.lower(), code)
        words = normalized.split()
        stripped = words[:]
        while len(stripped) > 1 and stripped[-1] in _GENERIC_SUFFIX:
            stripped = stripped[:-1]
        if stripped != words:
            add(" ".join(stripped), code)  # name minus its generic suffix
        for token in words:
            if token not in _STOPWORDS and len(token) >= 3:
                single_word_to_codes.setdefault(token, set()).add(code)

    # Single-word aliases register only when unambiguous across the catalog.
    for word, codes in single_word_to_codes.items():
        if len(codes) == 1:
            add(word, next(iter(codes)))

    # Curated aliases override anything generated, including ambiguous matches.
    for alias, code in _EXTRA_ALIASES.items():
        if code in code_to_name:
            alias_to_codes[normalize(alias)] = {code}

    known = [
        (alias, next(iter(codes)), code_to_name[next(iter(codes))])
        for alias, codes in alias_to_codes.items()
        if len(codes) == 1
    ]
    # Longest aliases first, so "scott hall" matches before "scott".
    known.sort(key=lambda item: -len(item[0]))
    return known, dict(code_to_name)


@lru_cache(maxsize=1)
def load_location_index() -> tuple[tuple[tuple[str, str, str], ...], dict[str, str]]:
    """Read buildings.json once and derive the alias index and label map."""
    path = _buildings_path()
    try:
        raw = json.loads(path.read_text(encoding="utf-8")) if path else None
    except (OSError, json.JSONDecodeError):
        raw = None
    catalog = raw if isinstance(raw, dict) and raw else _FALLBACK_CODE_TO_NAME
    code_to_name = {
        str(code): str(name)
        for code, name in catalog.items()
        if isinstance(code, str) and isinstance(name, str) and name.strip()
    }
    known, id_to_label = _build_index(code_to_name)
    return tuple(known), id_to_label


_known, LOCATION_ID_TO_LABEL = load_location_index()
KNOWN_CMU_LOCATIONS: list[tuple[str, str, str]] = list(_known)

# Curated nicknames exported for the system prompt. The model resolves what
# users say, so it must be informed of slang the formal names do not carry.
CURATED_NICKNAMES: list[tuple[str, str]] = sorted(
    (alias, code) for alias, code in _EXTRA_ALIASES.items()
)
