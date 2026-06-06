"""
ChromaDB local persistent implementation — development / offline / CI.

ChromaDB's Python client is synchronous; all calls are wrapped with
asyncio.to_thread so the event loop remains unblocked.

Lifecycle:
    store = await ChromaVectorStore.create()
    ...
    await store.close()
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import structlog

from backend.core.config import get_settings
from backend.core.exceptions import (
    MigrationError,
    SearchError,
    UpsertError,
    VectorStoreError,
)
from backend.vector_store.base import BaseVectorStore, DocumentChunk, SearchResult


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
