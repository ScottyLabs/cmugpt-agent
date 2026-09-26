"""The remember and forget tools exposed to the model.

Both are bound to a single user_id at construction, so a call cannot reach
another user's memory. The graph identifies them by the metadata marker set
here rather than by name, since an MCP server could publish a tool that is
also named remember.
"""

from __future__ import annotations

from typing import TypedDict

from langchain_core.tools import BaseTool, StructuredTool
from langgraph.store.base import BaseStore
from pydantic import BaseModel, Field

from .facts import add_fact, forget, resolve_forget_targets
from .manage import clear_memory
from .store import is_valid_user_id

# Names of the model-callable memory tools. Trust is decided by the metadata
# marker below, never by name.
REMEMBER_TOOL = "remember"
FORGET_TOOL = "forget"

# BaseTool.metadata marker set by build_memory_tools. The graph trusts only
# tools carrying it: an MCP server could publish its own "remember" tool, and
# that one must stay untrusted.
INTERNAL_MEMORY_METADATA = "cmugpt_internal_memory"


def is_internal_memory_tool(tool: BaseTool) -> bool:
    """True only for the trusted per-user memory tools built in this module."""
    return bool((tool.metadata or {}).get(INTERNAL_MEMORY_METADATA))


class MemoryWriteResult(TypedDict):
    """Trusted result returned by the remember tool to the graph."""

    message: str
    memory_id: str
    fact: str


class _RememberArgs(BaseModel):
    fact: str = Field(
        ...,
        description=(
            "One concise, durable fact about the user worth remembering in "
            "future chats (a stable preference, identity, or ongoing context) "
            "- not transient details about the current question."
        ),
    )


class _ForgetArgs(BaseModel):
    facts: list[str] | None = Field(
        default=None,
        description=(
            "The remembered facts to remove, one entry per fact, each "
            "quoted or closely paraphrased (for example 'lives in Mudge "
            "House', not 'where I live'). Omit when everything is true."
        ),
    )
    everything: bool = Field(
        default=False,
        description=(
            "Set true ONLY when the user asks to forget everything you "
            "know about them, including 'forget all that' after seeing "
            "their remembered facts."
        ),
    )
    confirmed: bool = Field(
        default=False,
        description=(
            "Required when facts has more than one entry. Set true ONLY "
            "when the user themselves named each of these facts, or has "
            "just answered a question from you confirming exactly which "
            "facts to forget. Never set it on your own inference."
        ),
    )


def build_memory_tools(store: BaseStore, user_id: str) -> list[BaseTool]:
    """Model-callable remember/forget tools bound to one user's namespace.

    ``user_id`` is captured in the closure, never model-supplied, so the tools
    cannot touch another user's memory. An unsafe id yields no tools.
    """
    if not is_valid_user_id(user_id):
        return []

    async def _remember(fact: str) -> MemoryWriteResult:
        normalized_fact = " ".join(fact.split())
        memory_id, status = await add_fact(
            store, user_id, normalized_fact, source="tool"
        )
        if memory_id is None:
            raise ValueError("Memory could not be saved.")
        messages = {
            "saved": f"Saved to memory: {normalized_fact}",
            "updated": f"Updated memory: {normalized_fact}",
            "duplicate": f"Already in memory: {normalized_fact}",
        }
        return MemoryWriteResult(
            message=messages.get(status, f"Saved to memory: {normalized_fact}"),
            memory_id=memory_id,
            fact=normalized_fact,
        )

    async def _forget(
        facts: list[str] | None = None,
        everything: bool = False,
        confirmed: bool = False,
    ) -> str:
        if everything:
            removed = await clear_memory(store, user_id)
            if removed == 0:
                return "No matching memory found to forget."
            return f"Forgot all {removed} remembered facts about this user."
        queries = [q for q in (facts or []) if isinstance(q, str) and q.strip()]
        if not queries:
            return "No matching memory found to forget."
        # Removing several facts at once requires the user's explicit
        # confirmation. The prompt alone cannot enforce this: a model can
        # read a singular request ("forget my allergy") as a category and
        # expand it to every matching fact, so the tool refuses instead and
        # reports what would be removed.
        if len(queries) > 1 and not confirmed:
            resolved = await resolve_forget_targets(store, user_id, queries)
            if not resolved:
                return "No matching memory found to forget."
            listing = "; ".join(f"'{text}'" for text in resolved)
            return (
                f"Nothing was forgotten yet. This would remove: {listing}. "
                "Confirm with the user exactly which of these to forget, "
                "then call forget again with confirmed=true."
            )
        forgotten: list[str] = []
        ambiguous: list[str] = []
        unmatched = 0
        for q in queries:
            message = await forget(store, user_id, q)
            if message.startswith("Forgot: "):
                text = message.removeprefix("Forgot: ")
                if text not in forgotten:
                    forgotten.append(text)
            elif "several remembered facts could match" in message:
                ambiguous.append(message)
            else:
                unmatched += 1
        if not forgotten:
            # Ambiguity outranks a plain no-match: the model must ask the
            # user which fact they meant rather than report failure.
            return ambiguous[0] if ambiguous else "No matching memory found to forget."
        summary = "Forgot: " + "; ".join(forgotten)
        if ambiguous:
            summary += " Also: " + ambiguous[0]
        if unmatched:
            summary += (
                f" ({unmatched} other requested "
                f"fact{'s' if unmatched > 1 else ''} had no close match)"
            )
        return summary

    return [
        StructuredTool.from_function(
            coroutine=_remember,
            name=REMEMBER_TOOL,
            metadata={INTERNAL_MEMORY_METADATA: True},
            description=(
                "Save a durable fact about the user (a stable preference, "
                "identity, or ongoing context) so future chats can use it. Call "
                "this only when the user explicitly asks you to remember or "
                "save the fact."
            ),
            args_schema=_RememberArgs,
        ),
        StructuredTool.from_function(
            coroutine=_forget,
            name=FORGET_TOOL,
            metadata={INTERNAL_MEMORY_METADATA: True},
            description=(
                "Remove remembered facts about the user. Pass `facts` with "
                "one entry per fact to remove, quoting or closely "
                "paraphrasing each. When the user asks to forget everything "
                "you know about them, set `everything` to true instead."
            ),
            args_schema=_ForgetArgs,
        ),
    ]
