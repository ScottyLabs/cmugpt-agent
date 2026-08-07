"""Deterministic, framework-agnostic guards and metadata computation.

These helpers enforce what the system prompt can only request. The prompt
asks the model to behave, while the guards here make leaks and false
disclosures impossible regardless of what the model generates. None of them
call an LLM, so they add zero input tokens and stay easy to unit-test.

Covered here. Tool-transparency repair, deterministic `thought` computation,
secret and prompt-leak scrubbing of outgoing text, a streaming holdback
scrubber, and a zero-token fast path for flagrant injection attempts.
"""

import os
import re
from typing import Any
from urllib.parse import urlparse

from .schema import ActionType, AgentResponse, Thought

# Strings the prompt explicitly tells the model to echo. The leak detector
# must never treat these as prompt leakage, or crisis answers and polite
# refusals would be replaced by refusals. prompts.py interpolates these same
# constants into the prompt so the allowlist cannot drift from it.
IDENTITY_PHRASE = "CMUGPT, an assistant for CMU campus information"
CRISIS_RESOURCES_LINE = (
    "CMU CaPS (412-268-2922), the 988 Suicide & Crisis Lifeline, or CMU "
    "Police (412-268-2323)"
)
GENERAL_INFO_QUALIFIER = "based on general info - please verify"
REFUSAL_TEXT = (
    "I can't help with that, but I'd be glad to help you find a building, "
    "dining option, or course on campus."
)
ECHO_SAFE_SNIPPETS = (
    IDENTITY_PHRASE,
    CRISIS_RESOURCES_LINE,
    GENERAL_INFO_QUALIFIER,
    REFUSAL_TEXT,
)

TOOL_TRANSPARENCY_RE = re.compile(
    r"\b(mcp|mcps|tool|tools|external service|external services|look(?:ed)? up)\b",
    re.IGNORECASE,
)

CMU_DATA_RE = re.compile(
    r"\b("
    r"cmu|carnegie mellon|campus|dining|food|eat|eateries|restaurant|"
    r"cafe|cafes|coffee|menu|open|hours|building|where|location|"
    r"course|class|professor|faculty|event|events|transit|shuttle|"
    r"parking|library|libraries"
    r")\b",
    re.IGNORECASE,
)

NEGATIVE_TOOL_CLAIM_PATTERNS = [
    re.compile(
        r"\bI\s+(?:have\s+not|haven't|did\s+not|didn't)\s+"
        r"(?:use|used)\s+any\s+(?:MCPs?\s+or\s+)?tools?\s*"
        r"(?:yet|so\s+far)?[^.!\n]*(?:[.!]\s*)?",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:no|none)\s+(?:MCPs?\s+or\s+)?tools?\s+"
        r"(?:were\s+)?used[^.!\n]*(?:[.!]\s*)?",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:the\s+)?information\s+(?:I\s+provided\s+)?"
        r"(?:is|was)\s+based\s+on\s+general\s+knowledge[^.!\n]*(?:[.!]\s*)?",
        re.IGNORECASE,
    ),
]

# Heuristic markers that suggest the assistant declined or redirected. Used only
# to calibrate confidence; correctness of refusals is enforced elsewhere.
REFUSAL_MARKERS = (
    "can't help",
    "cannot help",
    "can't assist",
    "cannot assist",
    "i can't",
    "i cannot",
    "i won't",
    "i'm not able",
    "i am not able",
    "unable to",
    "not able to help",
)


def latest_user_text(messages: list[dict[str, Any]]) -> str:
    """Return the most recent user message text from a role/content list."""
    for message in reversed(messages):
        if message.get("role") == "user" and isinstance(message.get("content"), str):
            return message["content"]
    return ""


def asks_about_tools(text: str) -> bool:
    return bool(TOOL_TRANSPARENCY_RE.search(text))


def should_require_tool(messages: list[dict[str, Any]]) -> bool:
    """Require a tool for CMU data lookups when tools are available."""
    query = latest_user_text(messages)
    if not query:
        return False
    return bool(CMU_DATA_RE.search(query))


def tool_metadata_message(services_used: list[str]) -> dict[str, str]:
    """System message reminding the model which tools were actually used."""
    names = ", ".join(f"`{name}`" for name in services_used)
    return {
        "role": "system",
        "content": (
            "Tool-use metadata for this turn: MCP/tools used: "
            f"{names}. If the user asks about tool or MCP usage, say that "
            "tools were used and name these user-safe tools. Do not claim "
            "that no tools were used."
        ),
    }


def strip_negative_tool_claims(text: str) -> str:
    cleaned = text
    for pattern in NEGATIVE_TOOL_CLAIM_PATTERNS:
        cleaned = pattern.sub("", cleaned)
    return re.sub(r"\n{3,}", "\n\n", cleaned).strip()


def apply_tool_transparency_guard(
    parsed: AgentResponse,
    messages: list[dict[str, Any]],
    services_used: list[str],
) -> AgentResponse:
    """Keep user-facing tool disclosure consistent with authoritative metadata."""
    if not services_used:
        return parsed

    parsed.services_used = services_used
    query = latest_user_text(messages)
    if not asks_about_tools(query):
        return parsed

    names = ", ".join(f"`{name}`" for name in services_used)
    disclosure = f"I did use MCP-connected tools for this turn: {names}."
    text = parsed.response_text or ""
    stripped = strip_negative_tool_claims(text)
    lower = stripped.lower()
    names_mentioned = any(name.lower() in lower for name in services_used)
    tool_mentioned = "tool" in lower or "mcp" in lower

    if not stripped:
        parsed.response_text = disclosure
    elif not tool_mentioned or not names_mentioned or stripped != text.strip():
        parsed.response_text = f"{disclosure}\n\n{stripped}"
    else:
        parsed.response_text = stripped
    return parsed


_REDACTION = "[redacted]"

# Env vars whose live values must never reach the user. The model does not
# see most of them, but tool errors and misconfigured servers can carry them.
_SECRET_ENV_NAMES = (
    "OPENROUTER_API_KEY",
    "AGENT_SHARED_SECRET",
    "MCP_SERVER_URL",
    "DATABASE_URL",
)

# Values that overlap these public URLs are never redacted, otherwise every
# legitimate map link would be mangled.
_PUBLIC_URL_PREFIXES = ("https://maps.scottylabs.org",)

# Unset or tiny values are skipped because replacing them would corrupt
# ordinary text.
_MIN_SECRET_CHARS = 8

_OPENROUTER_KEY_RE = re.compile(r"sk-or(?:-v1)?-[A-Za-z0-9]{20,}")

# Only high-entropy bearer tokens are redacted so code-help answers with
# placeholder tokens survive.
_BEARER_TOKEN_RE = re.compile(r"(?i)\bbearer\s+([A-Za-z0-9._~+/=-]{30,})")

# A leak is an 80-char normalized window of the answer found verbatim in the
# prompt. The step trades a slightly higher effective threshold for fewer
# scans.
_LEAK_WINDOW = 80
_LEAK_SCAN_STEP = 8


def _normalize_leak_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower())


def _secret_values() -> list[str]:
    values: list[str] = []
    for name in _SECRET_ENV_NAMES:
        value = (os.getenv(name) or "").strip()
        if len(value) < _MIN_SECRET_CHARS:
            continue
        if any(value in public for public in _PUBLIC_URL_PREFIXES):
            continue
        values.append(value)
        # Transport errors often carry just the host, so redact the netloc of
        # URL-shaped secrets too.
        if "://" in value:
            netloc = urlparse(value).netloc
            if len(netloc) >= _MIN_SECRET_CHARS and not any(
                netloc in public for public in _PUBLIC_URL_PREFIXES
            ):
                values.append(netloc)
    return values


def build_leak_corpus(system_prompt: str) -> str:
    """Normalized prompt text with the echo-safe snippets cut out.

    Snippets are replaced with a NUL byte so a scan window can never match
    across the seam left by a removal.
    """
    corpus = _normalize_leak_text(system_prompt)
    for snippet in ECHO_SAFE_SNIPPETS:
        corpus = corpus.replace(_normalize_leak_text(snippet), "\x00")
    return corpus


def _matches_leak_corpus(text: str, corpus: str) -> bool:
    if not corpus:
        return False
    normalized = _normalize_leak_text(text)
    if len(normalized) < _LEAK_WINDOW:
        return False
    for start in range(0, len(normalized) - _LEAK_WINDOW + 1, _LEAK_SCAN_STEP):
        if normalized[start : start + _LEAK_WINDOW] in corpus:
            return True
    return False


def _redact_bearer_match(match: re.Match[str]) -> str:
    token = match.group(1)
    has_lower = any(c.islower() for c in token)
    has_digit = any(c.isdigit() for c in token)
    # Real tokens mix lowercase and digits. Placeholders like YOUR_TOKEN_HERE
    # or your_token_goes_right_here do not.
    if has_lower and has_digit:
        return match.group(0).replace(token, _REDACTION)
    return match.group(0)


def redact_secrets(text: str) -> str:
    cleaned = text
    for value in _secret_values():
        cleaned = cleaned.replace(value, _REDACTION)
    cleaned = _OPENROUTER_KEY_RE.sub(_REDACTION, cleaned)
    return _BEARER_TOKEN_RE.sub(_redact_bearer_match, cleaned)


def apply_output_guard(text: str, system_prompt: str) -> tuple[str, bool]:
    """Scrub outgoing text. Returns (clean text, whether fully replaced).

    Runs last in postprocess so text injected by earlier guards is scanned
    too. A prompt leak replaces the whole answer, while stray secrets are
    redacted in place.
    """
    if not text:
        return text, False
    if _matches_leak_corpus(text, build_leak_corpus(system_prompt)):
        return REFUSAL_TEXT, True
    return redact_secrets(text), False


class StreamScrubber:
    """Rolling holdback so leaks are caught before they reach the SSE wire.

    Postprocess cannot retract an emitted delta, so the stream lags the model
    by a fixed tail. Any leak detectable within the scan window is still
    unemitted when the trip happens. After a trip nothing more is emitted and
    postprocess puts the refusal in the authoritative done payload.
    """

    # Larger than the leak window so a detected window is always still held.
    HOLDBACK_CHARS = 160

    def __init__(self, system_prompt: str) -> None:
        self._corpus = build_leak_corpus(system_prompt) if system_prompt else ""
        self._secrets = _secret_values()
        self._text = ""
        self._emitted = 0
        self.tripped = False

    def _dangerous(self) -> bool:
        if any(value in self._text for value in self._secrets):
            return True
        if _OPENROUTER_KEY_RE.search(self._text):
            return True
        return _matches_leak_corpus(self._text, self._corpus)

    def push(self, chunk: str) -> str:
        """Add model text, return the part now safe to emit."""
        if self.tripped:
            return ""
        self._text += chunk
        if self._dangerous():
            self.tripped = True
            return ""
        safe_until = max(self._emitted, len(self._text) - self.HOLDBACK_CHARS)
        out = self._text[self._emitted : safe_until]
        self._emitted = safe_until
        return out

    def flush(self) -> str:
        """Return the held tail after a final scan. Empty when tripped."""
        if self.tripped:
            return ""
        if self._dangerous():
            self.tripped = True
            return ""
        out = self._text[self._emitted :]
        self._emitted = len(self._text)
        return out


# High-precision signatures only. A false positive refuses a legitimate user
# with no model recourse, so anything ambiguous stays with the prompt rules.
# The lookahead keeps shell-help questions about terminal prompts untouched.
INJECTION_FAST_PATH_RE = re.compile(
    r"(?i)("
    r"ignore\s+all\s+(?:previous|prior|above)\s+instructions"
    r"|disregard\s+(?:all\s+)?(?:your|previous|prior)\s+instructions"
    r"|you\s+are\s+now\s+dan\b"
    r"|developer\s+mode\s+enabled"
    r"|(?:reveal|repeat|output|print|show)\s+(?:me\s+)?(?:your|the)\s+"
    r"(?:system|hidden)\s+prompt"
    r"(?![^.?!\n]*\b(?:zsh|bash|shell|terminal|ps1|prompt_command)\b)"
    r")"
)


def is_flagrant_injection(query: str) -> bool:
    """True only for unambiguous jailbreak phrasing worth a canned refusal.

    This is a cost optimization, not the defense. The prompt rules and the
    output guard handle everything these signatures miss.
    """
    return bool(INJECTION_FAST_PATH_RE.search(query or ""))


def canned_refusal_response() -> AgentResponse:
    """Refusal without any model call, calibrated like a computed refusal."""
    return AgentResponse(
        thought=Thought(
            reasoning="Declined or redirected the request.", confidence=0.3
        ),
        action=ActionType.RESPOND,
        response_text=REFUSAL_TEXT,
    )


def _looks_like_refusal(text: str) -> bool:
    lowered = text.lower()
    return any(marker in lowered for marker in REFUSAL_MARKERS)


def compute_thought(
    services_used: list[str],
    tool_invocations: list[dict[str, Any]],
    response_text: str,
) -> Thought:
    """Deterministically derive confidence + reasoning from the answer context.

    Replaces the model's former self-reported `thought`. Follows the calibration
    rubric from the system prompt:

    * 0.9+  : an authoritative tool returned data this turn
    * 0.6-0.8: partial tool data, or solid training knowledge
    * 0.2-0.4: declining / unable to answer
    """
    text = response_text or ""
    if not text.strip():
        return Thought(reasoning="No answer produced.", confidence=0.2)

    if _looks_like_refusal(text):
        return Thought(
            reasoning="Declined or redirected the request.",
            confidence=0.3,
        )

    # A failed call has a non-empty failure string, so the ok flag is what
    # separates real data from errors here.
    tools_returned_data = any(
        inv.get("ok", True)
        and isinstance(inv.get("result"), str)
        and inv["result"].strip()
        for inv in tool_invocations
    )
    if services_used and tools_returned_data:
        names = ", ".join(services_used)
        return Thought(
            reasoning=f"Answered using tool data from: {names}.",
            confidence=0.9,
        )
    if services_used:
        return Thought(
            reasoning="Tools returned limited data; answered with caveats.",
            confidence=0.5,
        )
    return Thought(
        reasoning="Answered from general knowledge.",
        confidence=0.7,
    )
