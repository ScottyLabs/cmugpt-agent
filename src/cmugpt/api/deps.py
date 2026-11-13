"""Checks shared by more than one route."""

import secrets
from http import HTTPStatus

from fastapi import Depends, HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from cmugpt.settings import get_settings

# Request bodies are rejected by header before parsing. Uvicorn imposes no default body limit, so oversized input is both a cost and an abuse vector.
_MAX_BODY_BYTES = 256 * 1024


# Shared-secret authentication for the agent and memory routers. Every request to them must carry a bearer token matching AGENT_SHARED_SECRET. Without a secret the routes fail closed, unless AGENT_ALLOW_UNAUTHENTICATED opts in for local development, so a deployment that forgets the secret is unusable rather than open to the internet. auto_error=False stops HTTPBearer from raising its own 403, so a missing header reaches require_shared_secret and gets the standard error envelope.
_bearer_scheme = HTTPBearer(auto_error=False)


def require_shared_secret(
    creds: HTTPAuthorizationCredentials | None = Depends(_bearer_scheme),  # noqa: B008 - Depends() as a default is FastAPI's convention
) -> None:
    settings = get_settings()
    expected = settings.agent_shared_secret
    if not expected:
        if settings.agent_allow_unauthenticated and not settings.is_production:
            return
        raise HTTPException(
            status_code=HTTPStatus.SERVICE_UNAVAILABLE,
            detail="Authentication is not configured.",
        )
    token_ok = (
        creds is not None
        and creds.scheme.lower() == "bearer"
        # Compared in constant time. An ordinary `!=` stops at the first wrong character, and that timing difference can reveal the secret one prefix at a time.
        and secrets.compare_digest(
            creds.credentials.encode("utf-8"), expected.encode("utf-8")
        )
    )
    if not token_ok:
        raise HTTPException(
            status_code=HTTPStatus.UNAUTHORIZED,
            detail="Invalid or missing bearer token.",
        )


def reject_oversized_body(request: Request) -> None:
    """Reject oversized bodies by header before request.json() parses them."""
    length = request.headers.get("content-length")
    if length is not None and length.isdigit() and int(length) > _MAX_BODY_BYTES:
        raise HTTPException(
            status_code=HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
            detail=f"Request body must be at most {_MAX_BODY_BYTES} bytes.",
        )
