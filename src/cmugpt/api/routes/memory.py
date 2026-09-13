"""Endpoints behind the Surface's memory manager."""

from http import HTTPStatus
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import JSONResponse

from cmugpt.memory import (
    clear_memory,
    delete_memory_item,
    ensure_store,
    is_valid_user_id,
    list_memory_items,
)

from ..deps import require_shared_secret

router = APIRouter(prefix="/memory", dependencies=[Depends(require_shared_secret)])


def _require_valid_user_id(user_id: str) -> None:
    """Reject path-param user ids that are unsafe as a memory namespace key."""
    if not is_valid_user_id(user_id):
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST,
            detail="Invalid 'user_id'.",
        )


@router.get("/{user_id}")
async def get_memory(
    user_id: str,
    q: Annotated[str | None, Query(max_length=200)] = None,
    kind: Literal["learned", "remembered"] | None = None,
    limit: Annotated[int, Query(ge=1, le=200)] = 200,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> JSONResponse:
    """Search a user's learned and explicitly remembered facts."""
    _require_valid_user_id(user_id)
    store = await ensure_store()
    items, total = await list_memory_items(
        store,
        user_id,
        query=q,
        memory_type=kind,
        limit=limit,
        offset=offset,
    )
    return JSONResponse(
        content={
            "user_id": user_id,
            "items": items,
            "total": total,
            "limit": limit,
            "offset": offset,
        },
        status_code=HTTPStatus.OK,
    )


@router.delete("/{user_id}/items/{kind}/{item_id}")
async def delete_typed_memory_item(
    user_id: str,
    kind: Literal["learned", "remembered"],
    item_id: str,
) -> JSONResponse:
    """Delete one learned or explicitly remembered fact."""
    _require_valid_user_id(user_id)
    store = await ensure_store()
    deleted = await delete_memory_item(store, user_id, kind, item_id)
    if not deleted:
        raise HTTPException(
            status_code=HTTPStatus.NOT_FOUND,
            detail="Memory item not found.",
        )
    return JSONResponse(
        content={"status": "deleted", "id": item_id, "type": kind},
        status_code=HTTPStatus.OK,
    )


@router.delete("/{user_id}")
async def clear_user_memory(user_id: str) -> JSONResponse:
    """Delete all user memory, including any legacy raw-chat snippets."""
    _require_valid_user_id(user_id)
    store = await ensure_store()
    removed = await clear_memory(store, user_id)
    return JSONResponse(
        content={"status": "cleared", "removed": removed},
        status_code=HTTPStatus.OK,
    )
