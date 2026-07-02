"""Offline tests for persistent user memory (agent/memory.py).

These run with no database and no network: an ``InMemoryStore`` backs the store
and a deterministic bag-of-words embedding stands in for OpenAI, so semantic
search, dedup, and forget behave deterministically. The live extraction pass
(``learn``) calls an LLM and is exercised by the manual E2E scripts instead.
"""

import asyncio
import re
import zlib
from collections.abc import Sequence
from math import sqrt
from typing import Any, cast

from langgraph.store.base import IndexConfig
from langgraph.store.memory import InMemoryStore

from agent import memory

_EMBED_DIMS = 64


def _fake_embed(texts: Sequence[str]) -> list[list[float]]:
    """Deterministic L2-normalized bag-of-words vectors.

    Texts that share words get high cosine similarity, which is all the memory
    logic needs to exercise semantic search, dedup, and forget without a model.
    """
    vectors: list[list[float]] = []
    for text in texts:
        vec = [0.0] * _EMBED_DIMS
        for word in re.findall(r"[a-z0-9]+", text.lower()):
            vec[zlib.crc32(word.encode()) % _EMBED_DIMS] += 1.0
        norm = sqrt(sum(value * value for value in vec)) or 1.0
        vectors.append([value / norm for value in vec])
    return vectors


def _indexed_store() -> InMemoryStore:
    index = cast(
        IndexConfig,
        {"dims": _EMBED_DIMS, "embed": _fake_embed, "fields": ["text"]},
    )
    return InMemoryStore(index=index)


def assert_true(condition: bool, label: str) -> None:
    if not condition:
        raise AssertionError(label)


def assert_equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise AssertionError(f"{label}: expected {expected!r}, got {actual!r}")


async def _test_recall_roundtrip_and_isolation() -> None:
    store = _indexed_store()
    await memory.add_fact(store, "u1", "Is a CS sophomore at CMU")
    await memory.add_fact(store, "u1", "Vegetarian, avoid meat dining recs")

    block = await memory.recall(store, "u1", "dining recommendations for tonight")
    assert_true("Vegetarian" in block, "recall surfaces a stored fact")
    assert_true("Memory about this user" in block, "recall has a header")

    # A different user shares nothing.
    other = await memory.recall(store, "u2", "anything at all")
    assert_equal(other, "", "recall is isolated per user")


async def _test_dedup_skips_duplicates() -> None:
    store = _indexed_store()
    first = await memory.add_fact(store, "u1", "Lives in Morewood")
    dup = await memory.add_fact(store, "u1", "lives in morewood")  # case variant
    assert_true(first is not None, "first fact stored")
    assert_true(dup is None, "duplicate fact skipped")
    facts = await memory.list_facts(store, "u1")
    assert_equal(len(facts), 1, "only one fact persisted")


async def _test_forget_removes_best_match() -> None:
    store = _indexed_store()
    await memory.add_fact(store, "u1", "Allergic to peanuts")
    await memory.add_fact(store, "u1", "Prefers window seats")

    message = await memory.forget(store, "u1", "allergic to peanuts")
    assert_true("Forgot" in message, "forget confirms removal")
    remaining = await memory.list_facts(store, "u1")
    assert_true(
        all("peanut" not in fact["text"].lower() for fact in remaining),
        "the peanut fact is gone",
    )
    assert_equal(len(remaining), 1, "the unrelated fact survives")

    # An unrelated query should not delete anything.
    miss = await memory.forget(store, "u1", "favorite programming language")
    assert_true("No matching" in miss, "weak match is a no-op")


async def _test_memory_tools_write_to_namespace() -> None:
    store = _indexed_store()
    tools = memory.build_memory_tools(store, "u1")
    assert_equal(
        {tool.name for tool in tools},
        {memory.REMEMBER_TOOL, memory.FORGET_TOOL},
        "both memory tools are built",
    )

    remember = next(tool for tool in tools if tool.name == memory.REMEMBER_TOOL)
    result = await remember.ainvoke({"fact": "Majors in ECE"})
    assert_true("Saved to memory" in result, "remember confirms")

    facts = await memory.list_facts(store, "u1")
    assert_true(any("ECE" in fact["text"] for fact in facts), "fact was written")

    # The user id is captured in the closure, so u2 never sees it.
    assert_equal(await memory.list_facts(store, "u2"), [], "tools are user-scoped")


async def _test_recall_without_index_degrades() -> None:
    store = InMemoryStore()  # no embeddings — recency fallback
    await memory.add_fact(store, "u1", "Plays club soccer on weekends")
    block = await memory.recall(store, "u1", "hobbies")
    assert_true("club soccer" in block, "recall works without an index")


async def _test_clear_memory() -> None:
    store = _indexed_store()
    await memory.add_fact(store, "u1", "Uses a standing desk")
    await memory.add_episode(store, "u1", "Where is the nearest gym on campus?", "…")
    removed = await memory.clear_memory(store, "u1")
    assert_true(removed >= 2, "clear removes facts and episodes")
    assert_equal(await memory.list_facts(store, "u1"), [], "nothing remains")


async def _test_forget_keyword_fallback_without_index() -> None:
    store = InMemoryStore()  # no embeddings
    await memory.add_fact(store, "u1", "Allergic to peanuts")
    await memory.add_fact(store, "u1", "Prefers window seats")

    miss = await memory.forget(store, "u1", "favorite color")
    assert_true("No matching" in miss, "no shared words -> no-op, not arbitrary")
    assert_equal(len(await memory.list_facts(store, "u1")), 2, "nothing deleted")

    hit = await memory.forget(store, "u1", "the peanuts thing")
    assert_true("peanuts" in hit, "keyword overlap picks the right fact")
    remaining = [f["text"] for f in await memory.list_facts(store, "u1")]
    assert_equal(remaining, ["Prefers window seats"], "only the match removed")


async def _test_growth_caps_prune_oldest() -> None:
    original_facts, original_episodes = memory._MAX_FACTS, memory._MAX_EPISODES
    memory._MAX_FACTS = memory._MAX_EPISODES = 3
    try:
        store = _indexed_store()
        for i in range(5):
            await memory.add_fact(store, "u1", f"Distinct standalone fact number {i}")
            await memory.add_episode(
                store, "u1", f"A sufficiently long question number {i}", "answer"
            )
        facts = [f["text"] for f in await memory.list_facts(store, "u1")]
        assert_equal(len(facts), 3, "facts capped")
        assert_true(
            all(f"number {i}" in " ".join(facts) for i in (2, 3, 4)),
            "newest facts survive, oldest evicted",
        )
        episodes = await memory._search(store, ("u1", "episodes"), None, 50)
        assert_equal(len(episodes), 3, "episodes capped")
    finally:
        memory._MAX_FACTS, memory._MAX_EPISODES = original_facts, original_episodes


async def _test_explicit_facts_evicted_last() -> None:
    """At the cap, auto-extracted facts are dropped before explicit saves,
    even when the explicit saves are older."""
    original = memory._MAX_FACTS
    memory._MAX_FACTS = 3
    try:
        store = _indexed_store()
        await memory.add_fact(store, "u1", "Shellfish allergy warning", source="tool")
        await memory.add_fact(store, "u1", "Morning class preference", source="tool")
        await memory.add_fact(
            store, "u1", "Note about dining halls", source="extraction"
        )
        await memory.add_fact(
            store, "u1", "Note about campus shuttles", source="extraction"
        )
        await memory.add_fact(
            store, "u1", "Note about library hours", source="extraction"
        )

        remaining = {f["text"] for f in await memory.list_facts(store, "u1")}
        assert_equal(len(remaining), 3, "cap enforced")
        assert_true(
            "Shellfish allergy warning" in remaining
            and "Morning class preference" in remaining,
            "older explicit saves outlive newer extracted facts",
        )
        assert_true(
            "Note about library hours" in remaining,
            "newest extracted fact survives",
        )
    finally:
        memory._MAX_FACTS = original


def _test_prompt_memory_section() -> None:
    from agent.prompts import build_system_prompt

    tools = memory.build_memory_tools(InMemoryStore(), "u1")
    with_memory = build_system_prompt(tools)
    assert_true(
        "Persistent user memory" in with_memory,
        "prompt teaches memory when tools present",
    )
    assert_true("`remember`" in with_memory, "prompt names the remember tool")

    without_memory = build_system_prompt([])
    assert_true(
        "Persistent user memory" not in without_memory,
        "anonymous turns get no memory section",
    )


def _test_worth_extracting_gate() -> None:
    assert_true(
        memory._worth_extracting("I'm a junior studying design"),
        "personal statement passes the gate",
    )
    assert_true(
        not memory._worth_extracting("where is gates building"),
        "impersonal lookup is skipped",
    )
    assert_true(not memory._worth_extracting("hi"), "trivial turn is skipped")


def _test_parse_facts_is_tolerant() -> None:
    assert_equal(memory._parse_facts('["a", "b"]'), ["a", "b"], "plain array")
    assert_equal(
        memory._parse_facts('Here you go:\n["x"]\nthanks'),
        ["x"],
        "array embedded in prose",
    )
    assert_equal(memory._parse_facts("not json at all"), [], "non-json -> empty")
    assert_equal(memory._parse_facts("[]"), [], "empty array -> empty")


async def _run_async() -> None:
    await _test_recall_roundtrip_and_isolation()
    await _test_dedup_skips_duplicates()
    await _test_forget_removes_best_match()
    await _test_memory_tools_write_to_namespace()
    await _test_recall_without_index_degrades()
    await _test_clear_memory()
    await _test_forget_keyword_fallback_without_index()
    await _test_growth_caps_prune_oldest()
    await _test_explicit_facts_evicted_last()


def run() -> None:
    asyncio.run(_run_async())
    _test_worth_extracting_gate()
    _test_parse_facts_is_tolerant()
    _test_prompt_memory_section()


if __name__ == "__main__":
    run()
    print("Memory tests passed.")
