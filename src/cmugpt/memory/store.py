"""Process-wide LangGraph store for user memory.

Uses Postgres with pgvector when DATABASE_URL is set and an in-memory store
otherwise. Owns setup, shutdown, the readiness check, and search(), the
single read path the other memory modules use.
"""

from __future__ import annotations

import asyncio
import logging
import re
from datetime import UTC, datetime
from typing import Any, cast

from langchain_openai import OpenAIEmbeddings
from langgraph.store.base import BaseStore, IndexConfig, SearchItem
from langgraph.store.memory import InMemoryStore

from ..settings import get_settings

logger = logging.getLogger(__name__)

FACTS = "facts"
# Raw chat turns from older builds. Nothing writes here anymore. The name is
# kept so cleanup can still delete what those builds stored.
EPISODES = "episodes"

# The user_id is the key that separates one user's stored memory from
# another's. LangGraph matches that key as a SQL LIKE pattern without
# escaping, so the allowlist excludes the wildcards ("%", "_") and the
# separator ".". A hostile user_id therefore cannot match another user's
# namespace. Every entry point checks it.
_USER_ID_RE = re.compile(r"^[A-Za-z0-9@:+=~-]{1,128}$")


def is_valid_user_id(user_id: str | None) -> bool:
    """True when ``user_id`` is safe to use as a memory namespace key."""
    return bool(user_id) and bool(_USER_ID_RE.match(user_id))


# Postgres connection pool size. Without a pool, langgraph shares one
# connection and every memory operation waits its turn on it.
_PG_POOL_MIN = 1
_PG_POOL_MAX = 10
_PG_SETUP_LOCK_ID = 4848217165257290356

_PG_SCHEMA = "agent_memory"

_EMBED_DIMS = 3072
_EMBED_MODEL = "text-embedding-3-large"
_PG_VECTOR_TYPE = "halfvec"


def _embeddings() -> OpenAIEmbeddings | None:
    """OpenAI embeddings for semantic search. None when OPENAI_API_KEY is unset.

    Without embeddings the store still works, but recall degrades to recency.
    """
    if not get_settings().openai_api_key:
        return None
    return OpenAIEmbeddings(model=_EMBED_MODEL, dimensions=_EMBED_DIMS)


def _index_config() -> IndexConfig | None:
    embeddings = _embeddings()
    if embeddings is None:
        return None
    return cast(
        IndexConfig,
        {
            "dims": _EMBED_DIMS,
            "embed": embeddings,
            "fields": ["text"],
            # pgvector's HNSW index supports at most 2,000 dimensions for
            # vector and 4,000 for halfvec. The large OpenAI model emits 3,072.
            "ann_index_config": {
                "kind": "hnsw",
                "vector_type": _PG_VECTOR_TYPE,
            },
        },
    )


_store: BaseStore | None = None
_pg_cm: Any = None
_store_lock = asyncio.Lock()


async def setup_store() -> BaseStore:
    """Create (once) and return the process-wide memory store.

    Postgres when ``DATABASE_URL`` is set, otherwise in-memory. Idempotent and
    concurrency-safe. Call from the app lifespan or lazily via ensure_store.
    """
    global _store, _pg_cm
    async with _store_lock:
        cached = _store
        if cached is not None:
            return cached
        index = _index_config()
        db_url = get_settings().database_url
        store: BaseStore
        if db_url:
            try:
                from langgraph.store.postgres import AsyncPostgresStore
            except ImportError as exc:  # pragma: no cover - optional extra
                raise RuntimeError(
                    "DATABASE_URL is set but the Postgres store is not "
                    "installed. Add 'langgraph-checkpoint-postgres' and "
                    "'psycopg[binary]' to the project dependencies."
                ) from exc
            _pg_cm = AsyncPostgresStore.from_conn_string(
                _conn_string_with_search_path(db_url),
                # _index_config's dict carries Postgres-specific keys beyond
                # the shared IndexConfig type.
                index=cast(Any, index),
                pool_config=cast(
                    Any, {"min_size": _PG_POOL_MIN, "max_size": _PG_POOL_MAX}
                ),
            )
            pg_store = await _pg_cm.__aenter__()
            try:
                await _setup_postgres_store(pg_store, db_url)
                await _verify_postgres_vector_dimensions(pg_store)
            except BaseException as exc:
                await _pg_cm.__aexit__(type(exc), exc, exc.__traceback__)
                _pg_cm = None
                raise
            store = pg_store
        else:
            store = InMemoryStore(index=index)
        _store = store
        return store


async def ensure_store() -> BaseStore:
    """Return the store, lazily setting it up if the lifespan hasn't run."""
    return _store if _store is not None else await setup_store()


async def close_store() -> None:
    """Tear down the store (e.g. the Postgres pool). Call from app shutdown."""
    global _store, _pg_cm
    async with _store_lock:
        if _pg_cm is not None:
            await _pg_cm.__aexit__(None, None, None)
        _pg_cm = None
        _store = None


def store_status() -> dict[str, Any]:
    """Report the active memory backend for /health. Never touches the DB.

    Before init, reports the backend implied by the environment.
    """
    settings = get_settings()
    if _store is not None:
        backend = "postgres" if _pg_cm is not None else "in-memory"
        initialized = True
    else:
        backend = "postgres" if settings.database_url else "in-memory"
        initialized = False
    return {
        "backend": backend,
        "initialized": initialized,
        "semantic_search": (
            has_index(_store) if _store is not None else bool(settings.openai_api_key)
        ),
        "embedding_model": _EMBED_MODEL if settings.openai_api_key else None,
    }


async def _verify_postgres_vector_dimensions(store: Any) -> None:
    """Fail at startup when the existing vector column has stale dimensions.

    Changing embedding models does not migrate an existing column. Without
    this check the mismatch would surface only on the first embedding write.
    """
    if not has_index(store):
        return
    pool = getattr(store, "conn", None)
    if pool is None or not callable(getattr(pool, "connection", None)):
        return
    async with cast(Any, pool).connection() as conn, conn.cursor() as cur:
        await cur.execute(
            """
                SELECT format_type(a.atttypid, a.atttypmod) AS vector_type
                FROM pg_attribute AS a
                JOIN pg_class AS c ON c.oid = a.attrelid
                WHERE c.relname = 'store_vectors'
                  AND a.attname = 'embedding'
                  AND a.attnum > 0
                  AND NOT a.attisdropped
                """
        )
        row = await cur.fetchone()
    if isinstance(row, dict):
        actual = str(row.get("vector_type", ""))
    else:
        actual = str(row[0]) if row else ""
    expected = f"{_PG_VECTOR_TYPE}({_EMBED_DIMS})"
    if actual and actual != expected:
        raise RuntimeError(
            "The existing memory vector index uses "
            f"{actual}, but {_EMBED_MODEL} requires {expected}. Rebuild the "
            "store_vectors/vector_migrations tables and re-index existing "
            "memory before starting the agent."
        )


def _conn_string_with_search_path(db_url: str) -> str:
    """Append search_path options to a URL-style conn string.

    An explicit `options=` already present in the URL is the operator's
    choice and wins.
    """
    if "options=" in db_url:
        return db_url
    options = f"options=-csearch_path%3D{_PG_SCHEMA},public"
    separator = "&" if "?" in db_url else "?"
    return f"{db_url}{separator}{options}"


async def _setup_postgres_store(store: Any, db_url: str) -> None:
    """Serialize LangGraph's first-run migrations across worker processes."""
    from psycopg import AsyncConnection

    # Dedicated autocommit connection for the advisory lock: borrowing from
    # the store's own pool during setup can deadlock a small pool.
    async with await AsyncConnection.connect(db_url, autocommit=True) as conn:
        try:
            # The schema name is written out here rather than composed from
            # _PG_SCHEMA, so nothing can be injected into this statement. Keep the
            # two in sync.
            await conn.execute("CREATE SCHEMA IF NOT EXISTS agent_memory")
        except Exception as exc:
            raise RuntimeError(
                f"Could not create memory schema {_PG_SCHEMA!r}. The "
                "database role needs either CREATE on the database or an "
                "existing schema it can write to (ask ops for: GRANT "
                f"USAGE, CREATE ON SCHEMA {_PG_SCHEMA} TO <role>)."
            ) from exc
        # Poll pg_try_advisory_lock instead of blocking in pg_advisory_lock:
        # LangGraph runs CREATE INDEX CONCURRENTLY, which waits on every older
        # open transaction, so a worker parked inside the blocking SELECT
        # would deadlock the worker holding the lock.
        while True:
            cursor = await conn.execute(
                "SELECT pg_try_advisory_lock(%s)", (_PG_SETUP_LOCK_ID,)
            )
            row = await cursor.fetchone()
            if row is not None and bool(row[0]):
                break
            await asyncio.sleep(0.1)
        try:
            await store.setup()
        finally:
            await conn.execute("SELECT pg_advisory_unlock(%s)", (_PG_SETUP_LOCK_ID,))


async def store_is_ready() -> bool:
    """Check that the configured store can answer a real query."""
    try:
        store = await ensure_store()
        await store.asearch(("healthcheck", FACTS), limit=1)
    except Exception:
        logger.warning("memory readiness check failed", exc_info=True)
        return False
    return True


def has_index(store: BaseStore) -> bool:
    return getattr(store, "index_config", None) is not None


def now_iso() -> str:
    return datetime.now(UTC).isoformat()


def item_text(item: SearchItem) -> str:
    return str(item.value.get("text", "")).strip()


async def search(
    store: BaseStore,
    namespace: tuple[str, str],
    query: str | None,
    limit: int,
    *,
    offset: int = 0,
    suppress_errors: bool = True,
) -> list[SearchItem]:
    """Semantic search when the store is indexed, else a recency listing.

    Never raises: a failed lookup returns ``[]`` so memory degrades instead of
    breaking the turn. Backend caveat: no-query listings are newest-first on
    Postgres but insertion-order (oldest-first) on the dev/CI InMemoryStore,
    so order-dependent behavior differs between local runs and production.
    """
    if not is_valid_user_id(namespace[0]):
        # Every caller has already validated the id, but every read passes
        # through here, so check again and never build a query from an unsafe
        # key.
        return []
    try:
        if query and has_index(store):
            return await store.asearch(
                namespace, query=query, limit=limit, offset=offset
            )
        return await store.asearch(namespace, limit=limit, offset=offset)
    except Exception:
        if not suppress_errors:
            raise
        logger.warning(
            "memory search failed for namespace %s", namespace[1:], exc_info=True
        )
        return []
