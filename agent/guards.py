"""Deterministic, framework-agnostic guards and metadata computation.

These helpers post-process the model's plain-Markdown answer. They contain the
real business logic that used to live inside the hand-rolled completion loop:

* tool-transparency: keep user-facing disclosure consistent with the
  authoritative `services_used` list computed by the graph (never the model's
  self-report).
* metadata: compute `thought` (confidence + reasoning) deterministically from
  the calibration rubric, since the model no longer emits a JSON envelope.

None of these functions touch the LLM or any framework; they operate on plain
data so they stay easy to unit-test.
"""

import re
from typing import Any

from .schema import AgentResponse, Thought

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

    tools_returned_data = any(
        isinstance(inv.get("result"), str) and inv["result"].strip()
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
