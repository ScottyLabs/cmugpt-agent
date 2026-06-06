"""OAuth routes for per-user Gmail authorisation via Keycloak.

Uses Keycloak as the sole identity and token provider. After the user
authenticates via Keycloak → Google, we only store the user_id → Keycloak
subject mapping locally. All tokens live in Keycloak.

Provides ``/auth/gmail/start``, ``/auth/gmail/callback``, and
``/auth/gmail/status``.
"""

import hashlib
import hmac
import json
import logging
import os

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse

from email_interface.keycloak_auth import get_keycloak_client, keycloak_enabled
from email_interface.token_store import TokenStore

logger = logging.getLogger(__name__)

auth_router = APIRouter(prefix="/auth/gmail", tags=["auth"])

_token_store = TokenStore()

_STATE_SECRET = os.getenv("AGENT_SHARED_SECRET", "cmugpt-dev-fallback")


def _get_redirect_uri() -> str:
    base = os.getenv(
        "OAUTH_REDIRECT_BASE_URL",
        f"http://localhost:{os.getenv('PORT', '8000')}",
    )
    return f"{base}/auth/gmail/callback"


def _sign_state(user_id: str) -> str:
    """Create an HMAC-signed state parameter encoding user_id."""
    sig = hmac.new(
        _STATE_SECRET.encode(), user_id.encode(), hashlib.sha256
    ).hexdigest()[:16]
    return json.dumps({"uid": user_id, "sig": sig})


def _verify_state(state: str) -> str | None:
    """Verify the signed state. Returns ``user_id`` or ``None``."""
    try:
        data = json.loads(state)
    except (json.JSONDecodeError, TypeError):
        return None
    uid = data.get("uid", "")
    sig = data.get("sig", "")
    expected = hmac.new(
        _STATE_SECRET.encode(), uid.encode(), hashlib.sha256
    ).hexdigest()[:16]
    if not hmac.compare_digest(sig, expected):
        return None
    return uid


@auth_router.get("/start")
async def gmail_auth_start(
    user_id: str = Query(..., description="Caller's user ID"),
) -> RedirectResponse:
    """Redirect the user to Keycloak, which brokers Google OAuth consent."""
    if not keycloak_enabled():
        raise HTTPException(
            status_code=503,
            detail="Keycloak not configured. Set KEYCLOAK_SERVER_URL and KEYCLOAK_CLIENT_ID.",
        )

    kc = get_keycloak_client()
    redirect_uri = _get_redirect_uri()
    state = _sign_state(user_id)

    auth_url = kc.get_auth_url(redirect_uri=redirect_uri, state=state)
    logger.info("OAuth start for user %s → redirecting to Keycloak/Google", user_id)
    return RedirectResponse(auth_url)


@auth_router.get("/callback")
async def gmail_auth_callback(
    code: str = Query(...),
    state: str = Query(...),
) -> HTMLResponse:
    """Handle Keycloak's OAuth redirect after Google consent.

    Exchanges the code for Keycloak tokens, retrieves userinfo (including
    the Keycloak subject ID), and stores only the user mapping locally.
    Tokens remain in Keycloak.
    """
    user_id = _verify_state(state)
    if not user_id:
        raise HTTPException(status_code=400, detail="Invalid or tampered state parameter.")

    kc = get_keycloak_client()
    redirect_uri = _get_redirect_uri()

    try:
        token_response = await kc.exchange_code(code=code, redirect_uri=redirect_uri)
    except Exception as exc:
        logger.exception("Keycloak token exchange failed for user %s", user_id)
        raise HTTPException(
            status_code=400, detail=f"Token exchange failed: {exc}"
        ) from exc

    kc_access_token = token_response.get("access_token", "")
    if not kc_access_token:
        raise HTTPException(status_code=500, detail="No access token from Keycloak.")

    # Get user info — includes Keycloak 'sub' and email
    try:
        userinfo = await kc.get_userinfo(kc_access_token)
        keycloak_sub = userinfo.get("sub", "")
        email = userinfo.get("email", "")
    except Exception as exc:
        logger.exception("Failed to fetch userinfo for user %s", user_id)
        raise HTTPException(
            status_code=500, detail=f"Could not resolve user info: {exc}"
        ) from exc

    if not keycloak_sub:
        raise HTTPException(status_code=500, detail="Keycloak userinfo missing 'sub'.")
    if not email:
        raise HTTPException(status_code=500, detail="Keycloak userinfo returned no email.")

    # Ensure the user has the read-token role so the broker endpoint works
    await kc.ensure_broker_read_token(keycloak_sub)

    # Only store the mapping — tokens live in Keycloak
    _token_store.save_user(user_id=user_id, keycloak_sub=keycloak_sub, email=email)

    logger.info("Gmail auth complete for user %s → KC sub %s (%s)", user_id, keycloak_sub, email)

    return HTMLResponse(
        f"""<!DOCTYPE html>
<html><head><title>Gmail Authorised</title>
<style>
  body {{ font-family: system-ui, sans-serif; display: flex;
         justify-content: center; align-items: center; height: 100vh;
         margin: 0; background: #f5f5f5; }}
  .card {{ background: white; border-radius: 12px; padding: 2rem 3rem;
           box-shadow: 0 2px 8px rgba(0,0,0,0.1); text-align: center; }}
  .ok {{ color: #2e7d32; font-size: 3rem; }}
</style></head><body>
<div class="card">
  <div class="ok">&#10003;</div>
  <h2>Gmail authorised!</h2>
  <p>Signed in as <strong>{email}</strong>.</p>
  <p>You can close this tab and go back to the chat.</p>
</div></body></html>""",
        status_code=200,
    )


@auth_router.get("/status")
async def gmail_auth_status(
    user_id: str = Query(..., description="Caller's user ID"),
) -> JSONResponse:
    """Check whether a user has authorised Gmail via Keycloak."""
    email = _token_store.get_email(user_id)
    return JSONResponse(
        {"authorized": email is not None, "email": email},
    )
