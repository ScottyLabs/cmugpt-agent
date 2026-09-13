"""Assembles the FastAPI application: lifespan, CORS, error envelope, routers."""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from cmugpt.graph import drain_background_tasks
from cmugpt.memory import close_store, setup_store
from cmugpt.settings import get_settings

from .routes import agent, health, memory

logger = logging.getLogger(__name__)


def _validate_runtime_configuration() -> None:
    """Refuse to start a production deployment without a database or a shared
    secret. Starting anyway would run without durable memory or auth."""
    settings = get_settings()
    if not settings.is_production:
        return
    required = {
        "DATABASE_URL": settings.database_url,
        "AGENT_SHARED_SECRET": settings.agent_shared_secret,
    }
    missing = [name for name, value in required.items() if not value.strip()]
    if missing:
        raise RuntimeError(
            "Production configuration is missing required environment "
            f"variable(s): {', '.join(missing)}. Refusing to start with "
            "non-durable or unauthenticated user memory."
        )
    secret = settings.agent_shared_secret
    if secret != secret.strip():
        raise RuntimeError(
            "AGENT_SHARED_SECRET cannot have leading or trailing whitespace."
        )
    if len(secret) < 32:
        raise RuntimeError(
            "AGENT_SHARED_SECRET must be at least 32 characters in production."
        )


@asynccontextmanager
async def _lifespan(_: FastAPI) -> AsyncIterator[None]:
    """Open the memory store when the app starts. On shutdown, wait for any
    background memory writes to finish, then close the store."""
    _validate_runtime_configuration()
    if not get_settings().agent_shared_secret:
        logger.warning(
            "AGENT_SHARED_SECRET is not set: /agent/respond* and /memory/* "
            "are UNAUTHENTICATED. This is only acceptable in local dev."
        )
    await setup_store()
    try:
        yield
    finally:
        await drain_background_tasks()
        await close_store()


app = FastAPI(lifespan=_lifespan)

# CORS restricts only browser JavaScript running on another origin. It has no
# effect on direct requests from curl, scripts, or servers, so the bearer
# secret in deps.py is the access boundary. CORS exists to keep a malicious
# page from making a visitor's browser call this API.
_allowed_origins = get_settings().allowed_origin_list
app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_methods=["GET", "POST"],
    allow_headers=["Authorization", "Content-Type"],
)


@app.exception_handler(HTTPException)
async def _http_exception_handler(_: Request, exc: HTTPException) -> JSONResponse:
    """Emit both `error` and `detail` so older and newer clients both work."""
    detail = exc.detail if isinstance(exc.detail, str) else str(exc.detail)
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": detail, "detail": detail},
    )


app.include_router(health.router)
app.include_router(agent.router)
app.include_router(memory.router)


def main() -> None:
    # Uvicorn configures only its own loggers. Configure the root logger so
    # the application's cmugpt.* loggers emit too.
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    port = get_settings().port
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
