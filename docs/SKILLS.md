# Project Skills Map

This document defines your mastery path across architecture, implementation, testing, and operations.

Goal: become an AI master champion by repeatedly building, testing, and refining real systems.

## Skill Track 1: Core Reliability
- `backend/core/config.py`: robust settings and environment parsing
- `backend/core/exceptions.py`: domain error taxonomy
- `backend/core/logging.py`: structured observability baseline

## Skill Track 2: LLM Provider Engineering
- `backend/llm_client/base.py`: provider contract discipline
- `backend/llm_client/openai_client.py`: retries, timeouts, usage accounting
- `backend/llm_client/anthropic_client.py`: long-context and safety handling
- `backend/llm_client/deepseek_client.py` and `backend/llm_client/ollama_client.py`: provider portability and fallback

## Skill Track 3: Retrieval and Ranking
- `backend/vector_store/base.py`: abstract contract + Pydantic domain models
- `backend/vector_store/pg_vector.py`: PostgreSQL + pgvector with asyncpg pool, HNSW index, cosine distance `<=>`
- `backend/vector_store/chroma_store.py`: local persistent ChromaDB with async wrapping
- `backend/rag_pipeline/query_processor.py`: QueryTransformer — LLM rewriting + Hypothetical Document Embedding (HyDE)
- `backend/rag_pipeline/engine.py`: master RAGEngine — hybrid retrieval, RRF fusion, cross-encoder re-ranking, generation

## Skill Track 4: Agentic Workflows
- `backend/agents/orchestrator.py`: multi-step tool use and control loop behavior
- `examples/03_agentic_pipeline.py`: practical orchestration runbook

## Skill Track 5: Architecture & Production Design
- `docs/RAG_ARCHITECTURE.md`: exhaustive guide on chunking strategies, index topologies (HNSW vs IVFFlat), metadata filtering, RRF, cross-encoder re-ranking, security hardening

## Practice Surfaces

1. `python-learning/`: concept refresh and foundational Python drills (LinkedIn Learning notebooks, external resources).
2. `examples/`: end-to-end practical runs for each capability.
3. `tests/unit/`: behavior correctness and API contracts.
4. `tests/integration/`: cross-module confidence and workflow checks.

## Mastery Levels

1. Foundation: understand and run modules.
2. Builder: modify modules safely with tests.
3. Architect: design cross-layer improvements and trade-offs.
4. Operator: validate reliability, observability, and failure handling.

## Execution Rule
For each skill track update:
1. Implement in the correct layer.
2. Add or update tests.
3. Update docs when behavior changes.
4. Capture what you learned in the learning handbook.
