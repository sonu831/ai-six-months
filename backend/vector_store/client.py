"""
Backward-compatible re-export module for vector store types.

Prefer importing directly from the canonical source files:
    from backend.vector_store.base import BaseVectorStore, DocumentChunk, SearchResult
    from backend.vector_store.chroma_store import ChromaVectorStore
    from backend.vector_store.pg_vector import PgVectorStore
"""

from __future__ import annotations

from backend.vector_store.base import BaseVectorStore, DocumentChunk, SearchResult
from backend.vector_store.chroma_store import ChromaVectorStore
from backend.vector_store.pg_vector import PgVectorStore

__all__ = [
    "BaseVectorStore",
    "ChromaVectorStore",
    "DocumentChunk",
    "PgVectorStore",
    "SearchResult",
]
