"""Unauthenticated health check."""

import time
from http import HTTPStatus

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from cmugpt.memory import store_is_ready, store_status

router = APIRouter()

_READY_TTL_SECONDS = 5.0
_ready_cache: tuple[float, bool] | None = None


@router.get("/api/health")
async def health() -> JSONResponse:
    # The memory block reports which backend this deployment is using and
    # whether it is answering queries. The readiness probe runs a real store
    # query, so the result is cached briefly to keep this unauthenticated
    # endpoint from generating database load.
    global _ready_cache
    now = time.monotonic()
    if _ready_cache is not None and now - _ready_cache[0] < _READY_TTL_SECONDS:
        ready = _ready_cache[1]
    else:
        ready = await store_is_ready()
        _ready_cache = (now, ready)
    memory = {**store_status(), "ready": ready}
    return JSONResponse(
        content={"status": "ok" if ready else "degraded", "memory": memory},
        status_code=HTTPStatus.OK if ready else HTTPStatus.SERVICE_UNAVAILABLE,
    )
