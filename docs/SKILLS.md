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
- `backend/vector_store/client.py`: retrieval backend abstraction
- `backend/rag_pipeline/engine.py`: query transformation, fusion, reranking, answer synthesis

## Skill Track 4: Agentic Workflows
- `backend/agents/orchestrator.py`: multi-step tool use and control loop behavior
- `examples/03_agentic_pipeline.py`: practical orchestration runbook

## Practice Surfaces

1. `python-learning/`: concept refresh and foundational drills.
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
