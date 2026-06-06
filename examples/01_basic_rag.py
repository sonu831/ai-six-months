"""
Example 01 — Basic RAG with ChromaDB (no Docker required).

Demonstrates:
    - ChromaVectorStore local persistent setup
    - Ingesting a markdown document as chunked DocumentChunks
    - Running a full RAG query using OpenAI for both embeddings and generation
    - ScorePassthroughReranker (no Cohere key needed)

Run:
    poetry run python examples/01_basic_rag.py
"""

from __future__ import annotations

import asyncio
import textwrap
import uuid
from pathlib import Path

from backend.core.config import get_settings
from backend.core.logging import configure_logging, get_logger
from backend.llm_client.openai_client import OpenAIClient
from backend.rag_pipeline.engine import (
    RAGConfig,
    RAGEngine,
    RAGRequest,
    ScorePassthroughReranker,
)
from backend.vector_store.client import ChromaVectorStore, DocumentChunk

DOCUMENT_PATH = Path(__file__).parent.parent / "data" / "documents" / "sample.md"
CHUNK_SIZE = 400        # characters per chunk
CHUNK_OVERLAP = 80


def _chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Sliding-window character chunking with overlap."""
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end].strip())
        start += chunk_size - overlap
    return [c for c in chunks if len(c) > 50]


async def _ingest_document(
    store: ChromaVectorStore,
    embedder: OpenAIClient,
    document_id: str,
    text: str,
) -> int:
    log = get_logger("ingest")
    raw_chunks = _chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP)
    log.info("ingest.chunks_created", count=len(raw_chunks))

    # Embed all chunks in one batched API call
    from backend.llm_client.base import EmbeddingRequest

    embed_response = await embedder.embed(EmbeddingRequest(texts=raw_chunks))

    chunks = [
        DocumentChunk(
            chunk_id=str(uuid.uuid4()),
            document_id=document_id,
            content=text_chunk,
            embedding=embedding,
            metadata={"source": "sample.md", "chunk_index": str(i)},
        )
        for i, (text_chunk, embedding) in enumerate(
            zip(raw_chunks, embed_response.embeddings)
        )
    ]

    await store.upsert_embeddings(chunks)
    log.info("ingest.upsert_done", document_id=document_id, chunks=len(chunks))
    return len(chunks)


async def main() -> None:
    settings = get_settings()
    configure_logging(log_level=settings.log_level, environment=settings.environment)
    log = get_logger("example_01")
    log.info("example_01.start")

    # -- Clients --
    openai_client = OpenAIClient.create()

    # -- Vector store --
    vector_store = await ChromaVectorStore.create()

    # -- Reranker (no external service needed) --
    reranker = ScorePassthroughReranker()

    # -- RAG Engine --
    config = RAGConfig(
        retrieval_top_k=10,
        rerank_top_k=3,
        query_transform_enabled=True,
        enable_hyde=True,
    )
    engine = RAGEngine(
        llm_client=openai_client,
        embedder=openai_client,
        vector_store=vector_store,
        reranker=reranker,
        config=config,
    )

    # -- Ingest --
    doc_text = DOCUMENT_PATH.read_text(encoding="utf-8")
    doc_id = "sample_md_v1"
    await _ingest_document(vector_store, openai_client, doc_id, doc_text)

    # -- Query --
    queries = [
        "What is the difference between HNSW and IVFFlat indexes?",
        "How does Hypothetical Document Embedding work?",
        "What metrics should I monitor in production RAG systems?",
    ]

    for query in queries:
        print(f"\n{'='*70}")
        print(f"QUERY: {query}")
        print("=" * 70)

        request = RAGRequest(query=query)
        response = await engine.execute(request)

        print(f"\nANSWER:\n{textwrap.fill(response.answer, width=80)}")
        print(f"\nQuery variants used ({len(response.query_variants)}):")
        for i, v in enumerate(response.query_variants, 1):
            print(f"  {i}. {v[:90]}")
        print(f"\nSource chunks: {response.final_context_count} / {response.retrieval_candidate_count} retrieved")
        for chunk in response.source_chunks:
            print(f"  - [{chunk.document_id}] score={chunk.score:.3f} | {chunk.content[:80]}...")

    await engine.close()
    log.info("example_01.complete")


if __name__ == "__main__":
    asyncio.run(main())
