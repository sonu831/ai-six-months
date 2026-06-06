"""
Vector store abstract base class and domain models.

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

import uuid
from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, Field


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
        1. Expose a class-level async factory (``create()``).
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
