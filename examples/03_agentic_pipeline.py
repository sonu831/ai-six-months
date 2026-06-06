"""
Example 03 — Agentic Pipeline with ReAct Loop.

Demonstrates:
    - AgentOrchestrator with RAG search + calculator tools
    - Multi-step reasoning: agent decides when to invoke tools vs. answer directly
    - Registering a custom domain tool at runtime
    - Inspecting the step trace for observability

Run:
    poetry run python examples/03_agentic_pipeline.py
"""

from __future__ import annotations

import asyncio
import json
import textwrap
from pathlib import Path

from backend.agents.orchestrator import AgentOrchestrator, ToolDefinition
from backend.core.config import get_settings
from backend.core.logging import configure_logging, get_logger
from backend.llm_client.anthropic_client import AnthropicClient
from backend.llm_client.base import EmbeddingRequest
from backend.llm_client.openai_client import OpenAIClient
from backend.rag_pipeline.engine import (
    RAGConfig,
    RAGEngine,
    ScorePassthroughReranker,
)
from backend.vector_store.client import ChromaVectorStore, DocumentChunk

import uuid

DOCUMENT_PATH = Path(__file__).parent.parent / "data" / "documents" / "sample.md"


async def _bootstrap_rag_engine(embedder: OpenAIClient) -> RAGEngine:
    """Ingest the sample doc and build a ready RAGEngine."""
    store = await ChromaVectorStore.create(collection_name="agent_example")
    doc_text = DOCUMENT_PATH.read_text(encoding="utf-8")

    # Chunk and embed
    chunk_size = 400
    overlap = 80
    raw_chunks: list[str] = []
    start = 0
    while start < len(doc_text):
        raw_chunks.append(doc_text[start : start + chunk_size].strip())
        start += chunk_size - overlap
    raw_chunks = [c for c in raw_chunks if len(c) > 50]

    embed_response = await embedder.embed(EmbeddingRequest(texts=raw_chunks))
    chunks = [
        DocumentChunk(
            chunk_id=str(uuid.uuid4()),
            document_id="sample_md_v1",
            content=chunk_text,
            embedding=emb,
            metadata={"source": "sample.md"},
        )
        for chunk_text, emb in zip(raw_chunks, embed_response.embeddings)
    ]
    await store.upsert_embeddings(chunks)

    return RAGEngine(
        llm_client=AnthropicClient.create(),
        embedder=embedder,
        vector_store=store,
        reranker=ScorePassthroughReranker(),
        config=RAGConfig(retrieval_top_k=8, rerank_top_k=3),
    )


def _print_trace(steps: list[dict]) -> None:
    print("\n--- Agent Execution Trace ---")
    for entry in steps:
        print(
            f"  Step {entry['step']}: [{entry['tool']}] "
            f"args={json.dumps(entry['args'])[:80]} → "
            f"{entry['result'][:100]}..."
        )


async def main() -> None:
    settings = get_settings()
    configure_logging(log_level=settings.log_level, environment=settings.environment)
    log = get_logger("example_03")
    log.info("example_03.start")

    openai_client = OpenAIClient.create()
    anthropic_client = AnthropicClient.create()

    # Bootstrap the RAG engine that the agent will use as a tool
    print("Bootstrapping RAG engine...")
    rag_engine = await _bootstrap_rag_engine(openai_client)

    # Build orchestrator — the LLM driving the agent is Claude (Anthropic)
    orchestrator = AgentOrchestrator(
        llm=anthropic_client,
        rag_engine=rag_engine,
        max_steps=8,
        timeout_seconds=90.0,
    )

    # Register a custom domain tool: vector index cost estimator
    async def index_cost_estimator(args: dict) -> str:
        n_vectors = int(args.get("n_vectors", 0))
        dim = int(args.get("dimension", 1536))
        bytes_per_float = 4
        raw_bytes = n_vectors * dim * bytes_per_float
        hnsw_overhead = 1.4   # HNSW uses ~40% extra memory vs raw vectors
        total_gb = (raw_bytes * hnsw_overhead) / (1024 ** 3)
        return (
            f"Estimated HNSW index size for {n_vectors:,} vectors of dim {dim}: "
            f"{total_gb:.2f} GB. "
            f"Recommended postgres shared_buffers: {total_gb * 0.25:.2f} GB "
            f"(25% of index size)."
        )

    orchestrator.register_tool(
        ToolDefinition(
            name="index_cost_estimator",
            description=(
                "Estimate the memory footprint of an HNSW vector index. "
                "Use when asked about hardware sizing for a pgvector deployment."
            ),
            parameters_schema={
                "n_vectors": {"type": "integer", "description": "Number of vectors"},
                "dimension": {"type": "integer", "description": "Embedding dimension"},
            },
        ),
        index_cost_estimator,
    )

    # --- Run agentic queries ---
    agentic_queries = [
        (
            "I have a pgvector database with 2 million 1536-dimensional embeddings. "
            "Estimate the HNSW index size, then explain what HNSW parameters I should tune."
        ),
        (
            "What is the RRF formula for hybrid search, and what does the k=60 "
            "smoothing constant do? Compute the RRF score difference between a "
            "document ranked 1st vs one ranked 10th."
        ),
    ]

    for query in agentic_queries:
        print(f"\n{'='*70}")
        print(f"USER: {query}")
        print("=" * 70)

        result = await orchestrator.run(query)

        print(f"\nAGENT ANSWER:\n{textwrap.fill(result.final_answer, width=80)}")
        print(f"\nSteps taken: {result.step_count}")
        if result.tool_calls:
            _print_trace(result.tool_calls)

    await rag_engine.close()
    await anthropic_client.close()
    await openai_client.close()
    log.info("example_03.complete")


if __name__ == "__main__":
    asyncio.run(main())
