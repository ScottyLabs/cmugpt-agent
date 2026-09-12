"""Persistent, per-user memory across chats.

Facts live under the namespace (user_id, "facts"). Raw chat turns are never
stored, and turns without a user_id run with memory off. Embeddings need
OPENAI_API_KEY, since OpenRouter has no embeddings API. Without it, recall
degrades from semantic search to a recency listing.

The public API is re-exported here. Submodules: store (the LangGraph store
singleton), facts (recall, save, forget), tools (the remember and forget
tools the model calls), extraction (background learning), and manage (the
list, delete, and clear endpoints).
"""

from .extraction import learn
from .facts import add_fact, forget, recall, resolve_forget_targets
from .manage import (
    MemoryType,
    clear_memory,
    delete_memory_item,
    list_facts,
    list_memory_items,
)
from .store import (
    close_store,
    ensure_store,
    is_valid_user_id,
    setup_store,
    store_is_ready,
    store_status,
)
from .tools import (
    FORGET_TOOL,
    INTERNAL_MEMORY_METADATA,
    REMEMBER_TOOL,
    MemoryWriteResult,
    build_memory_tools,
    is_internal_memory_tool,
)

__all__ = [
    "FORGET_TOOL",
    "INTERNAL_MEMORY_METADATA",
    "REMEMBER_TOOL",
    "MemoryType",
    "MemoryWriteResult",
    "add_fact",
    "build_memory_tools",
    "clear_memory",
    "close_store",
    "delete_memory_item",
    "ensure_store",
    "forget",
    "is_internal_memory_tool",
    "is_valid_user_id",
    "learn",
    "list_facts",
    "list_memory_items",
    "recall",
    "resolve_forget_targets",
    "setup_store",
    "store_is_ready",
    "store_status",
]
