"""Input and output moderation via OpenAI's free moderation endpoint.

User messages are screened before the graph runs, and finished replies are
screened before the final payload is returned, replacing them retroactively
when a category trips. Self-harm signals take priority and route to campus
support resources rather than a refusal.
"""

import logging
import os
from typing import NamedTuple

import httpx

from agent.schema import ActionType, AgentResponse, Metadata, Thought

logger = logging.getLogger(__name__)

_MODERATION_URL = "https://api.openai.com/v1/moderations"
_MODERATION_MODEL = "omni-moderation-latest"
_TIMEOUT_SECONDS = 4.0

ALLOW = "allow"
REFUSE = "refuse"
SELF_HARM = "self_harm"

# Score thresholds (0..1) per category id as returned by the endpoint.
# Self-harm categories route to support resources instead of a refusal.
_SELF_HARM_THRESHOLDS = {
    "self-harm": 0.5,
    "self-harm/intent": 0.4,
    "self-harm/instructions": 0.4,
}

_REFUSE_THRESHOLDS = {
    "sexual/minors": 0.2,
    "sexual": 0.7,
    "hate/threatening": 0.5,
    "harassment/threatening": 0.6,
    "illicit/violent": 0.5,
    "violence/graphic": 0.7,
}

SELF_HARM_RESPONSE = (
    "It sounds like you might be going through something really difficult "
    "right now. You don't have to handle it alone, and support is available "
    "on campus whenever you're ready.\n\n"
    "- **CMU Counseling and Psychological Services (CaPS)**: 412-268-2922, "
    "available 24/7\n"
    "- **988 Suicide and Crisis Lifeline**: call or text 988, available 24/7\n"
    "- **If you are in immediate danger**: call 911, or CMU Police at "
    "412-268-2323\n\n"
    "If it would help, I can also point you to other campus support resources."
)

REFUSAL_RESPONSE = (
    "I can't help with that. If you have a question about campus, like "
    "courses, dining, or getting around CMU, I'm happy to help."
)

REDACTED_RESPONSE = (
    "This response was removed because it may have contained inappropriate "
    "content. If you have a question about campus, I'm happy to help."
)


class Verdict(NamedTuple):
    action: str
    categories: list[str]


_ALLOW_VERDICT = Verdict(ALLOW, [])


def _exceeded(scores: dict, thresholds: dict[str, float]) -> list[str]:
    flagged = []
    for category, threshold in thresholds.items():
        score = scores.get(category)
        if isinstance(score, int | float) and score >= threshold:
            flagged.append(category)
    return flagged


async def moderate_text(text: str) -> Verdict:
    """Classify ``text``, failing open to ALLOW on any error."""
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key or not text.strip():
        return _ALLOW_VERDICT
    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT_SECONDS) as client:
            res = await client.post(
                _MODERATION_URL,
                headers={"Authorization": f"Bearer {api_key}"},
                json={"model": _MODERATION_MODEL, "input": text},
            )
        if res.status_code != 200:
            logger.warning("moderation: OpenAI returned %s, allowing", res.status_code)
            return _ALLOW_VERDICT
        results = res.json().get("results")
        if not isinstance(results, list) or not results:
            logger.warning("moderation: unexpected response shape, allowing")
            return _ALLOW_VERDICT
        scores = results[0].get("category_scores")
        if not isinstance(scores, dict):
            logger.warning("moderation: unexpected response shape, allowing")
            return _ALLOW_VERDICT
        self_harm = _exceeded(scores, _SELF_HARM_THRESHOLDS)
        if self_harm:
            return Verdict(SELF_HARM, self_harm)
        refuse = _exceeded(scores, _REFUSE_THRESHOLDS)
        if refuse:
            return Verdict(REFUSE, refuse)
        return _ALLOW_VERDICT
    except Exception as exc:  # noqa: BLE001 - moderation must never break a turn
        logger.warning("moderation: check failed (%s), allowing", exc)
        return _ALLOW_VERDICT


def _canned_response(text: str) -> AgentResponse:
    return AgentResponse(
        thought=Thought(reasoning="Moderation policy response", confidence=1.0),
        action=ActionType.RESPOND,
        tool_calls=[],
        response_text=text,
        metadata=Metadata(),
    )


def blocked_input_response(action: str) -> AgentResponse:
    """The full reply for a blocked user message."""
    text = SELF_HARM_RESPONSE if action == SELF_HARM else REFUSAL_RESPONSE
    return _canned_response(text)


def redacted_output_response(action: str) -> AgentResponse:
    """The replacement for an agent reply that tripped moderation."""
    text = SELF_HARM_RESPONSE if action == SELF_HARM else REDACTED_RESPONSE
    return _canned_response(text)
