"""
PostgreSQL + pgvector implementation — production backend.

Uses an asyncpg connection pool for all database access.
Schema migration runs during ``create()`` so there is no separate
migration step required in development.

Vector operators:
    - Cosine distance: embedding <=> $1  (returns distance 0.0 → 2.0)
    - Cosine similarity: 1 - (embedding <=> $1)
    - HNSW index with m=16, ef_construction=64 for ANN retrieval

Full-text search uses tsvector/tsquery with BM25 ranking via ts_rank_cd.
Hard metadata filtering via JSONB containment (@>) leverages a GIN index.

Lifecycle:
    store = await PgVectorStore.create(embedding_dim=1536)
    chunks = [DocumentChunk(...)]
    await store.upsert_embeddings(chunks)
    results = await store.similarity_search(embedding, top_k=10)
    await store.close()
"""

from __future__ import annotations

import json
from typing import Any

import asyncpg
import structlog
from asyncpg import Pool

from backend.core.config import get_settings
from backend.core.exceptions import (
    MigrationError,
    SearchError,
    UpsertError,
    VectorStoreError,
)
from backend.vector_store.base import BaseVectorStore, DocumentChunk, SearchResult

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
