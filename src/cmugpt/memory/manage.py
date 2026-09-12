"""The management API behind the Surface's memory manager.

Lists, deletes, and clears a user's facts. Search here is a literal substring
match, not semantic, so results are predictable and typing costs nothing.
"""

from __future__ import annotations

import asyncio
from typing import Any, Literal

from langgraph.store.base import BaseStore, Item, SearchItem

from .facts import MAX_FACTS
from .store import EPISODES, FACTS, is_valid_user_id, item_text, search

MemoryType = Literal["learned", "remembered"]


async def list_facts(
    store: BaseStore, user_id: str, limit: int = 100
) -> list[dict[str, Any]]:
    items = await search(
        store,
        (user_id, FACTS),
        None,
        limit,
        suppress_errors=False,
    )
    return [{"id": item.key, **item.value} for item in items]


def _memory_type(item: Item) -> MemoryType:
    return "learned" if item.value.get("source") == "extraction" else "remembered"


def _memory_item(item: SearchItem) -> dict[str, Any]:
    created_at = item.value.get("created_at") or item.value.get("ts")
    if not created_at:
        created_at = item.created_at.isoformat()
    return {
        "id": item.key,
        "type": _memory_type(item),
        "text": item_text(item),
        "created_at": str(created_at),
    }


async def list_memory_items(
    store: BaseStore,
    user_id: str,
    *,
    query: str | None = None,
    memory_type: MemoryType | None = None,
    limit: int = 200,
    offset: int = 0,
) -> tuple[list[dict[str, Any]], int]:
    """List learned and remembered facts with literal substring search.

    Deliberately not semantic: predictable for users and free of per-keystroke
    embedding cost.
    """
    if not is_valid_user_id(user_id):
        return [], 0
    batch = await search(
        store,
        (user_id, FACTS),
        None,
        MAX_FACTS,
        suppress_errors=False,
    )
    items = [_memory_item(item) for item in batch if item_text(item)]
    if memory_type is not None:
        items = [item for item in items if item["type"] == memory_type]
    needle = (query or "").strip().casefold()
    if needle:
        items = [item for item in items if needle in str(item["text"]).casefold()]
    items.sort(key=lambda item: str(item["created_at"]), reverse=True)
    total = len(items)
    return items[offset : offset + limit], total


async def delete_memory_item(
    store: BaseStore,
    user_id: str,
    memory_type: MemoryType,
    item_id: str,
) -> bool:
    """Delete one user-visible fact, returning whether it existed."""
    if not is_valid_user_id(user_id):
        return False
    namespace = (user_id, FACTS)
    existing = await store.aget(namespace, item_id)
    if existing is None or _memory_type(existing) != memory_type:
        return False
    await store.adelete(namespace, item_id)
    return True


async def clear_memory(store: BaseStore, user_id: str) -> int:
    """Delete all facts plus any legacy raw-chat snippets. Return count removed."""
    removed = 0
    for suffix in (FACTS, EPISODES):
        namespace = (user_id, suffix)
        while True:
            batch = await search(
                store,
                namespace,
                None,
                500,
                suppress_errors=False,
            )
            if not batch:
                break
            await asyncio.gather(
                *(store.adelete(namespace, item.key) for item in batch)
            )
            removed += len(batch)
    return removed
