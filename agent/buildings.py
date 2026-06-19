"""Building catalog loaded from the committed ``buildings.json``.

``buildings.json`` is a flat ``{code: name}`` map (e.g. ``"MOR": "Morewood
Gardens"``) and nothing else. Everything users actually type against —
aliases, the reverse index, fuzzy single-word matches — is derived here at load
time, so the data file stays tiny and human-editable while a user can still say
"scott", "scott hall", or "donner" and get the right CMU Maps code.

The maps app needs the short code in its URL (``maps.scottylabs.org/<code>``),
so resolution always returns ``(code, name)``.
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

# Look in the repo root first (how `uvicorn src.main:app` runs in deploy) and
# then alongside this package, so a wheel build that ships buildings.json as
# package data still finds it.
_BUILDINGS_CANDIDATES = (
    Path(__file__).resolve().parents[1] / "buildings.json",
    Path(__file__).resolve().parent / "buildings.json",
)


def _buildings_path() -> Path | None:
    return next((p for p in _BUILDINGS_CANDIDATES if p.is_file()), None)


# Trailing words generic enough that "<distinctive> <generic>" should also
# resolve on just "<distinctive>" (so "scott hall" also yields "scott").
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
# Curated nicknames / disambiguations. Each wins over any generated mapping,
# including ambiguous ones (e.g. "hamerschlag" alone -> the academic Hall, not
# the dorm House).
_EXTRA_ALIASES = {
    "uc": "CUC",
    "university center": "CUC",
    "tsb": "TEP",
    "hamerschlag": "HH",
}

# Minimal fallback so the agent still runs if buildings.json is absent/corrupt.
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
    """Lowercase, fold ``&`` -> ``and``, reduce punctuation/hyphens to spaces.

    Used for BOTH alias generation and query matching so the two always live in
    the same character space (``Newell-Simon`` and ``newell simon`` both match).
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
            add(code.lower(), code)  # the code itself (skip 2-char: too noisy)
        words = normalized.split()
        stripped = words[:]
        while len(stripped) > 1 and stripped[-1] in _GENERIC_SUFFIX:
            stripped = stripped[:-1]
        if stripped != words:
            add(" ".join(stripped), code)  # name minus its generic suffix
        for token in words:
            if token not in _STOPWORDS and len(token) >= 3:
                single_word_to_codes.setdefault(token, set()).add(code)

    # Register single-word aliases only when unambiguous across the catalog.
    for word, codes in single_word_to_codes.items():
        if len(codes) == 1:
            add(word, next(iter(codes)))

    # Curated aliases override anything generated (including ambiguous matches).
    for alias, code in _EXTRA_ALIASES.items():
        if code in code_to_name:
            alias_to_codes[normalize(alias)] = {code}

    known = [
        (alias, next(iter(codes)), code_to_name[next(iter(codes))])
        for alias, codes in alias_to_codes.items()
        if len(codes) == 1
    ]
    # Longest aliases first so "scott hall" wins over "scott".
    known.sort(key=lambda item: -len(item[0]))
    return known, dict(code_to_name)


@lru_cache(maxsize=1)
def load_location_index() -> tuple[tuple[tuple[str, str, str], ...], dict[str, str]]:
    """Read buildings.json (cached) and derive the alias index + code->name."""
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
