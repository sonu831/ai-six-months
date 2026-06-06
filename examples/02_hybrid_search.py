"""
Example 02 — Hybrid Search Deep-Dive (requires PostgreSQL + pgvector).

Demonstrates:
    - PgVectorStore setup with asyncpg pool
    - Explicit side-by-side comparison: pure vector vs pure keyword vs hybrid RRF
    - Cohere re-ranking (falls back to cross-encoder if COHERE_API_KEY unset)
    - Observing how RRF fusion improves recall over either search strategy alone

Run:
    docker compose up -d postgres          # start pgvector instance
    poetry run python examples/02_hybrid_search.py
"""

from __future__ import annotations

import asyncio
import textwrap
import uuid
from pathlib import Path

from backend.core.config import get_settings
from backend.core.logging import configure_logging, get_logger
from backend.llm_client.base import EmbeddingRequest
from backend.llm_client.openai_client import OpenAIClient
from backend.rag_pipeline.engine import (
    CohereReranker,
    HuggingFaceCrossEncoderReranker,
    RAGConfig,
    RAGEngine,
    RAGRequest,
    ScorePassthroughReranker,
    reciprocal_rank_fusion,
)
from backend.vector_store.client import DocumentChunk, PgVectorStore

DOCUMENT_PATH = Path(__file__).parent.parent / "data" / "documents" / "sample.md"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
EMBEDDING_DIM = 1536


def _chunk_text(text: str, size: int, overlap: int) -> list[str]:
    chunks: list[str] = []
    start = 0
    while start < len(text):
        chunks.append(text[start : start + size].strip())
        start += size - overlap
    return [c for c in chunks if len(c) > 60]


async def _ingest(
    store: PgVectorStore,
    embedder: OpenAIClient,
    document_id: str,
    text: str,
) -> None:
    log = get_logger("ingest")
    raw_chunks = _chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP)
    embed_response = await embedder.embed(EmbeddingRequest(texts=raw_chunks))
    chunks = [
        DocumentChunk(
            chunk_id=str(uuid.uuid4()),
            document_id=document_id,
            content=chunk_text,
            embedding=emb,
            metadata={"source": "sample.md"},
        )
        for chunk_text, emb in zip(raw_chunks, embed_response.embeddings)
    ]
    await store.upsert_embeddings(chunks)
    log.info("ingest.done", document_id=document_id, chunks=len(chunks))


async def _compare_search_strategies(
    store: PgVectorStore,
    embedder: OpenAIClient,
    query: str,
    top_k: int = 5,
) -> None:
    """Run all three strategies and print a comparison table."""
    print(f"\n{'─'*70}")
    print(f"Query: {query}")
    print("─" * 70)

    embed_response = await embedder.embed(EmbeddingRequest(texts=[query]))
    query_embedding = embed_response.embeddings[0]

    # Strategy A — pure vector similarity
    vector_results = await store.similarity_search(
        query_embedding=query_embedding, top_k=top_k
    )
    print(f"\n[A] Vector-only (ANN cosine, top {top_k})")
    for i, r in enumerate(vector_results, 1):
        print(f"  {i}. score={r.score:.3f} | {r.content[:75]}...")

    # Strategy B — pure full-text keyword
    keyword_results = await store.keyword_search(query=query, top_k=top_k)
    print(f"\n[B] Keyword-only (tsvector BM25, top {top_k})")
    for i, r in enumerate(keyword_results, 1):
        print(f"  {i}. score={r.score:.4f} | {r.content[:75]}...")

    # Strategy C — RRF fusion of A + B
    fused = reciprocal_rank_fusion([vector_results, keyword_results])[:top_k]
    print(f"\n[C] Hybrid RRF (vector + keyword fused, top {top_k})")
    for i, r in enumerate(fused, 1):
        print(f"  {i}. rrf_score={r.score:.5f} | {r.content[:75]}...")

    # Overlap analysis
    vec_ids = {r.chunk_id for r in vector_results}
    kw_ids = {r.chunk_id for r in keyword_results}
    fused_ids = {r.chunk_id for r in fused}
    print(
        f"\n  Overlap: vector∩keyword={len(vec_ids & kw_ids)} | "
        f"new in fused={len(fused_ids - vec_ids - kw_ids)}"
    )


async def main() -> None:
    settings = get_settings()
    configure_logging(log_level=settings.log_level, environment=settings.environment)
    log = get_logger("example_02")
    log.info("example_02.start")

    openai_client = OpenAIClient.create()

    # pgvector store (requires Docker postgres running)
    store = await PgVectorStore.create(embedding_dim=EMBEDDING_DIM)

    doc_text = DOCUMENT_PATH.read_text(encoding="utf-8")
    await _ingest(store, openai_client, "sample_md_v1", doc_text)

    # --- Side-by-side search strategy comparison ---
    test_queries = [
        "HNSW index configuration parameters",   # technical/sparse — keyword wins
        "what makes vector search slow",          # semantic — vector wins
        "connection pool exhaustion risks",       # both relevant
    ]
    for query in test_queries:
        await _compare_search_strategies(store, openai_client, query)

    # --- Full RAG with Cohere re-ranking ---
    print(f"\n{'='*70}")
    print("Full RAG pipeline with re-ranker")
    print("=" * 70)

    cohere_key = settings.cohere.api_key
    if cohere_key and cohere_key.get_secret_value():
        reranker = CohereReranker(
            api_key=cohere_key.get_secret_value(),
            model=settings.cohere.rerank_model,
        )
        print("Reranker: Cohere API")
    else:
        reranker = HuggingFaceCrossEncoderReranker()
        print("Reranker: Local cross-encoder (Cohere key not set)")

    engine = RAGEngine(
        llm_client=openai_client,
        embedder=openai_client,
        vector_store=store,
        reranker=reranker,
        config=RAGConfig(retrieval_top_k=15, rerank_top_k=4),
    )

    response = await engine.execute(
        RAGRequest(query="How should I configure pgvector for a 5M document production dataset?")
    )
    print(f"\nAnswer:\n{textwrap.fill(response.answer, width=80)}")
    print(f"\nModel: {response.model_used} | Final chunks: {response.final_context_count}")

    await engine.close()
    await store.close()
    log.info("example_02.complete")


if __name__ == "__main__":
    asyncio.run(main())
