"""
Vector store layer — single module housing the abstract contract and both
concrete implementations.

Architecture:
    BaseVectorStore           (ABC — the interface contract)
         │
         ├── PgVectorStore   — asyncpg pool + pgvector extension (production)
         └── ChromaVectorStore — local persistent ChromaDB (dev / offline)

Design decisions:
    - asyncpg used directly (not via SQLAlchemy) for the hot query paths;
      the query builder is thin SQL strings, not an ORM abstraction.
    - ChromaDB's synchronous API is wrapped with asyncio.to_thread to keep
      the event loop unblocked.
    - Both implementations expose identical method signatures so callers can
      swap backends without changing a line of business logic.
    - DocumentChunk and SearchResult are Pydantic v2 models; callers only
      interact with these types, never with raw DB rows.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from abc import ABC, abstractmethod
from typing import Any

import asyncpg
import structlog
from asyncpg import Pool
from pydantic import BaseModel, Field

from backend.core.config import get_settings
from backend.core.exceptions import (
    MigrationError,
    SearchError,
    UpsertError,
    VectorStoreError,
)

# ---------------------------------------------------------------------------
# Domain models — every caller talks in these types
# ---------------------------------------------------------------------------


class DocumentChunk(BaseModel):
    """A single indexable unit of a source document."""

    chunk_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    document_id: str
    content: str
    embedding: list[float]
    metadata: dict[str, Any] = Field(default_factory=dict)


class SearchResult(BaseModel):
    """A retrieved chunk with its relevance score."""

    chunk_id: str
    document_id: str
    content: str
    score: float
    metadata: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class BaseVectorStore(ABC):
    """
    Protocol-aligned abstract base for all vector store backends.

    Every implementation must:
        1. Expose a class-level async factory (``create()`` or ``initialize_store()``).
        2. Be safe for concurrent access — all methods are coroutines.
        3. Raise typed exceptions from backend.core.exceptions; never naked Exception.
    """

    @abstractmethod
    async def upsert_embeddings(self, chunks: list[DocumentChunk]) -> list[str]:
        """
        Persist a batch of pre-embedded document chunks.

        Returns the list of chunk_ids that were written.
        Semantics are upsert: existing chunk_id → update, new → insert.
        """

    @abstractmethod
    async def similarity_search(
        self,
        query_embedding: list[float],
        top_k: int = 10,
        filter: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """
        ANN cosine-similarity retrieval against the stored embedding index.

        ``filter`` is applied BEFORE distance computation to reduce the
        candidate set. Format is implementation-dependent but both concrete
        classes accept flat ``{key: value}`` dicts.
        """

    @abstractmethod
    async def keyword_search(
        self,
        query: str,
        top_k: int = 10,
        filter: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """
        Lexical / full-text search over the ``content`` field.

        Postgres: uses tsvector BM25-ranked matching.
        ChromaDB: uses where_document $contains (substring match).
        """

    @abstractmethod
    async def delete_document(self, document_id: str) -> int:
        """
        Delete every chunk belonging to ``document_id``.
        Returns the count of deleted records.
        """

    @abstractmethod
    async def health_check(self) -> bool:
        """Return True if the backend is reachable and operational."""

    @abstractmethod
    async def close(self) -> None:
        """Release all resources. Idempotent; safe to call multiple times."""


# ---------------------------------------------------------------------------
# PostgreSQL + pgvector implementation
# ---------------------------------------------------------------------------

# DDL executed once during create() — idempotent due to IF NOT EXISTS guards.
_BOOTSTRAP_SQL = """
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;

CREATE TABLE IF NOT EXISTS document_chunks (
    chunk_id     TEXT          PRIMARY KEY,
    document_id  TEXT          NOT NULL,
    content      TEXT          NOT NULL,
    embedding    vector({dim}) NOT NULL,
    metadata     JSONB         NOT NULL DEFAULT '{{}}',
    created_at   TIMESTAMPTZ   NOT NULL DEFAULT NOW(),
    updated_at   TIMESTAMPTZ   NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_dc_document_id
    ON document_chunks (document_id);

CREATE INDEX IF NOT EXISTS idx_dc_embedding_hnsw
    ON document_chunks
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

CREATE INDEX IF NOT EXISTS idx_dc_content_fts
    ON document_chunks
    USING GIN (to_tsvector('english', content));

CREATE INDEX IF NOT EXISTS idx_dc_metadata_gin
    ON document_chunks USING GIN (metadata);
"""

_UPSERT_SQL = """
INSERT INTO document_chunks
    (chunk_id, document_id, content, embedding, metadata, updated_at)
VALUES
    ($1, $2, $3, $4::vector, $5::jsonb, NOW())
ON CONFLICT (chunk_id)
DO UPDATE SET
    document_id = EXCLUDED.document_id,
    content     = EXCLUDED.content,
    embedding   = EXCLUDED.embedding,
    metadata    = EXCLUDED.metadata,
    updated_at  = NOW();
"""

_VECTOR_SEARCH_SQL = """
SELECT
    chunk_id,
    document_id,
    content,
    metadata,
    1 - (embedding <=> $1::vector) AS score
FROM document_chunks
{where_clause}
ORDER BY embedding <=> $1::vector
LIMIT $2;
"""

_FTS_SEARCH_SQL = """
SELECT
    chunk_id,
    document_id,
    content,
    metadata,
    ts_rank_cd(
        to_tsvector('english', content),
        plainto_tsquery('english', $1)
    ) AS score
FROM document_chunks
WHERE to_tsvector('english', content) @@ plainto_tsquery('english', $1)
{extra_where}
ORDER BY score DESC
LIMIT $2;
"""


class PgVectorStore(BaseVectorStore):
    """
    Production PostgreSQL + pgvector implementation.

    Uses an asyncpg connection pool. Schema migration runs during ``create()``
    so there is no separate migration step required in development.

    Lifecycle:
        store = await PgVectorStore.create(embedding_dim=1536)
        chunks = [DocumentChunk(...)]
        await store.upsert_embeddings(chunks)
        results = await store.similarity_search(embedding, top_k=10)
        await store.close()
    """

    def __init__(self, pool: Pool, embedding_dim: int) -> None:
        self._pool = pool
        self._embedding_dim = embedding_dim
        self._log = structlog.get_logger(self.__class__.__name__)

    @classmethod
    async def create(
        cls,
        embedding_dim: int = 1536,
        *,
        dsn: str | None = None,
        min_size: int | None = None,
        max_size: int | None = None,
    ) -> "PgVectorStore":
        """
        Open an asyncpg pool and run schema bootstrap.

        All parameters fall back to values in DatabaseSettings if not provided.
        """
        cfg = get_settings().db
        resolved_dsn = dsn or cfg.dsn

        try:
            pool: Pool = await asyncpg.create_pool(
                dsn=resolved_dsn,
                min_size=min_size or cfg.pool_min_size,
                max_size=max_size or cfg.pool_max_size,
                max_queries=cfg.pool_max_queries,
                max_inactive_connection_lifetime=cfg.pool_max_inactive_connection_lifetime,
                statement_cache_size=100,
                command_timeout=30,
            )
        except Exception as exc:
            raise MigrationError(
                "Failed to open asyncpg pool",
                {"dsn_host": cfg.host, "error": str(exc)},
            ) from exc

        instance = cls(pool=pool, embedding_dim=embedding_dim)
        await instance._migrate()
        return instance

    async def _migrate(self) -> None:
        sql = _BOOTSTRAP_SQL.format(dim=self._embedding_dim)
        try:
            async with self._pool.acquire() as conn:
                await conn.execute(sql)
        except Exception as exc:
            raise MigrationError(
                "pgvector bootstrap DDL failed",
                {"embedding_dim": self._embedding_dim, "error": str(exc)},
            ) from exc
        self._log.info("pgvector.migrated", embedding_dim=self._embedding_dim)

    # ------------------------------------------------------------------
    # Write path
    # ------------------------------------------------------------------

    async def upsert_embeddings(self, chunks: list[DocumentChunk]) -> list[str]:
        if not chunks:
            return []
        self._log.debug("pgvector.upsert.start", count=len(chunks))
        records = [
            (
                c.chunk_id,
                c.document_id,
                c.content,
                c.embedding,
                json.dumps(c.metadata),
            )
            for c in chunks
        ]
        try:
            async with self._pool.acquire() as conn:
                async with conn.transaction():
                    await conn.executemany(_UPSERT_SQL, records)
        except asyncpg.PostgresError as exc:
            raise UpsertError(
                "pgvector batch upsert failed",
                {"chunk_count": len(chunks), "error": str(exc)},
            ) from exc
        self._log.info("pgvector.upsert.done", count=len(chunks))
        return [c.chunk_id for c in chunks]

    # ------------------------------------------------------------------
    # Read path — vector similarity (ANN)
    # ------------------------------------------------------------------

    async def similarity_search(
        self,
        query_embedding: list[float],
        top_k: int = 10,
        filter: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        where_clause, extra_params = self._build_jsonb_filter(filter, param_offset=2)
        sql = _VECTOR_SEARCH_SQL.format(where_clause=where_clause)
        self._log.debug("pgvector.similarity_search", top_k=top_k, has_filter=bool(filter))
        try:
            async with self._pool.acquire() as conn:
                rows = await conn.fetch(sql, query_embedding, top_k, *extra_params)
        except asyncpg.PostgresError as exc:
            raise SearchError(
                "pgvector similarity search failed",
                {"top_k": top_k, "error": str(exc)},
            ) from exc
        return [
            SearchResult(
                chunk_id=r["chunk_id"],
                document_id=r["document_id"],
                content=r["content"],
                score=float(r["score"]),
                metadata=dict(r["metadata"]),
            )
            for r in rows
        ]

    # ------------------------------------------------------------------
    # Read path — full-text keyword search (tsvector / BM25)
    # ------------------------------------------------------------------

    async def keyword_search(
        self,
        query: str,
        top_k: int = 10,
        filter: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        extra_where, extra_params = self._build_jsonb_filter(
            filter, param_offset=2, prefix="AND"
        )
        sql = _FTS_SEARCH_SQL.format(extra_where=extra_where)
        self._log.debug("pgvector.keyword_search", query=query[:80], top_k=top_k)
        try:
            async with self._pool.acquire() as conn:
                rows = await conn.fetch(sql, query, top_k, *extra_params)
        except asyncpg.PostgresError as exc:
            raise SearchError(
                "pgvector keyword search failed",
                {"query": query[:80], "error": str(exc)},
            ) from exc
        return [
            SearchResult(
                chunk_id=r["chunk_id"],
                document_id=r["document_id"],
                content=r["content"],
                score=float(r["score"]),
                metadata=dict(r["metadata"]),
            )
            for r in rows
        ]

    # ------------------------------------------------------------------
    # Delete
    # ------------------------------------------------------------------

    async def delete_document(self, document_id: str) -> int:
        try:
            async with self._pool.acquire() as conn:
                result = await conn.execute(
                    "DELETE FROM document_chunks WHERE document_id = $1",
                    document_id,
                )
        except asyncpg.PostgresError as exc:
            raise VectorStoreError(
                "pgvector document deletion failed",
                {"document_id": document_id, "error": str(exc)},
            ) from exc
        deleted = int(result.split()[-1])
        self._log.info("pgvector.delete_document", document_id=document_id, deleted=deleted)
        return deleted

    # ------------------------------------------------------------------
    # Admin
    # ------------------------------------------------------------------

    async def health_check(self) -> bool:
        try:
            async with self._pool.acquire() as conn:
                await conn.fetchval("SELECT 1")
            return True
        except Exception:
            return False

    async def close(self) -> None:
        await self._pool.close()
        self._log.info("pgvector.pool_closed")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_jsonb_filter(
        filter: dict[str, Any] | None,
        param_offset: int,
        prefix: str = "WHERE",
    ) -> tuple[str, list[Any]]:
        """
        Convert a flat metadata filter dict into a JSONB containment clause.

        Uses metadata @> $N::jsonb which leverages the GIN index on metadata.
        Returns (sql_fragment, [bind_params]).
        """
        if not filter:
            return "", []
        clause = f"{prefix} metadata @> ${param_offset + 1}::jsonb"
        return clause, [json.dumps(filter)]


# ---------------------------------------------------------------------------
# ChromaDB implementation — local dev / offline / CI
# ---------------------------------------------------------------------------


class ChromaVectorStore(BaseVectorStore):
    """
    Local persistent ChromaDB implementation.

    ChromaDB's Python client is synchronous; all calls are wrapped with
    asyncio.to_thread so the event loop remains unblocked.

    Lifecycle:
        store = await ChromaVectorStore.create()
        ...
        await store.close()
    """

    def __init__(self, client: Any, collection: Any) -> None:
        self._client = client
        self._collection = collection
        self._log = structlog.get_logger(self.__class__.__name__)

    @classmethod
    async def create(
        cls,
        collection_name: str | None = None,
        persist_directory: str | None = None,
    ) -> "ChromaVectorStore":
        import chromadb  # local import — optional dev dependency
        from chromadb import Settings as ChromaSettings  # type: ignore[attr-defined]

        cfg = get_settings().chroma
        resolved_dir = persist_directory or cfg.persist_directory
        resolved_collection = collection_name or cfg.collection_name

        def _init_sync() -> tuple[Any, Any]:
            client = chromadb.PersistentClient(
                path=resolved_dir,
                settings=ChromaSettings(anonymize_telemetry=False),
            )
            collection = client.get_or_create_collection(
                name=resolved_collection,
                metadata={"hnsw:space": "cosine"},
            )
            return client, collection

        try:
            client, collection = await asyncio.to_thread(_init_sync)
        except Exception as exc:
            raise MigrationError(
                "ChromaDB initialisation failed",
                {"persist_dir": resolved_dir, "collection": resolved_collection, "error": str(exc)},
            ) from exc

        instance = cls(client=client, collection=collection)
        instance._log.info(
            "chromadb.initialised",
            collection=resolved_collection,
            path=resolved_dir,
        )
        return instance

    # ------------------------------------------------------------------
    # Write path
    # ------------------------------------------------------------------

    async def upsert_embeddings(self, chunks: list[DocumentChunk]) -> list[str]:
        if not chunks:
            return []
        self._log.debug("chromadb.upsert.start", count=len(chunks))
        ids = [c.chunk_id for c in chunks]
        embeddings = [c.embedding for c in chunks]
        documents = [c.content for c in chunks]
        # Chroma metadata values must be str | int | float | bool
        metadatas = [
            {**{k: (v if isinstance(v, str | int | float | bool) else json.dumps(v))
               for k, v in c.metadata.items()},
             "document_id": c.document_id}
            for c in chunks
        ]
        try:
            await asyncio.to_thread(
                self._collection.upsert,
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas,
            )
        except Exception as exc:
            raise UpsertError(
                "ChromaDB upsert failed",
                {"chunk_count": len(chunks), "error": str(exc)},
            ) from exc
        self._log.info("chromadb.upsert.done", count=len(chunks))
        return ids

    # ------------------------------------------------------------------
    # Read path — vector similarity
    # ------------------------------------------------------------------

    async def similarity_search(
        self,
        query_embedding: list[float],
        top_k: int = 10,
        filter: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        self._log.debug("chromadb.similarity_search", top_k=top_k)
        query_kwargs: dict[str, Any] = {
            "query_embeddings": [query_embedding],
            "n_results": top_k,
            "include": ["documents", "metadatas", "distances"],
        }
        if filter:
            # Chroma $eq operator for each filter key
            query_kwargs["where"] = {k: {"$eq": v} for k, v in filter.items()}

        try:
            raw = await asyncio.to_thread(self._collection.query, **query_kwargs)
        except Exception as exc:
            raise SearchError(
                "ChromaDB similarity search failed",
                {"top_k": top_k, "error": str(exc)},
            ) from exc

        return self._parse_chroma_results(raw)

    # ------------------------------------------------------------------
    # Read path — keyword search
    # ------------------------------------------------------------------

    async def keyword_search(
        self,
        query: str,
        top_k: int = 10,
        filter: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        self._log.debug("chromadb.keyword_search", query=query[:80], top_k=top_k)
        query_kwargs: dict[str, Any] = {
            "query_texts": [query],
            "n_results": top_k,
            "include": ["documents", "metadatas", "distances"],
            "where_document": {"$contains": query},
        }
        if filter:
            query_kwargs["where"] = {k: {"$eq": v} for k, v in filter.items()}

        try:
            raw = await asyncio.to_thread(self._collection.query, **query_kwargs)
        except Exception as exc:
            raise SearchError(
                "ChromaDB keyword search failed",
                {"query": query[:80], "error": str(exc)},
            ) from exc

        return self._parse_chroma_results(raw)

    # ------------------------------------------------------------------
    # Delete
    # ------------------------------------------------------------------

    async def delete_document(self, document_id: str) -> int:
        try:
            existing = await asyncio.to_thread(
                self._collection.get,
                where={"document_id": {"$eq": document_id}},
            )
            ids_to_delete: list[str] = existing.get("ids", [])
            if ids_to_delete:
                await asyncio.to_thread(self._collection.delete, ids=ids_to_delete)
            self._log.info(
                "chromadb.delete_document",
                document_id=document_id,
                deleted=len(ids_to_delete),
            )
            return len(ids_to_delete)
        except Exception as exc:
            raise VectorStoreError(
                "ChromaDB document deletion failed",
                {"document_id": document_id, "error": str(exc)},
            ) from exc

    # ------------------------------------------------------------------
    # Admin
    # ------------------------------------------------------------------

    async def health_check(self) -> bool:
        try:
            await asyncio.to_thread(self._client.heartbeat)
            return True
        except Exception:
            return False

    async def close(self) -> None:
        # PersistentClient flushes to disk on GC; explicit no-op is safe.
        self._log.info("chromadb.closed")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_chroma_results(raw: dict[str, Any]) -> list[SearchResult]:
        ids: list[str] = (raw.get("ids") or [[]])[0]
        documents: list[str] = (raw.get("documents") or [[]])[0]
        metadatas: list[dict[str, Any]] = (raw.get("metadatas") or [[]])[0]
        distances: list[float] = (raw.get("distances") or [[]])[0]

        results: list[SearchResult] = []
        for chunk_id, content, meta, dist in zip(ids, documents, metadatas, distances):
            doc_id = str(meta.pop("document_id", ""))
            results.append(
                SearchResult(
                    chunk_id=chunk_id,
                    document_id=doc_id,
                    content=content,
                    score=1.0 - dist,  # cosine distance → cosine similarity
                    metadata=meta,
                )
            )
        return results
