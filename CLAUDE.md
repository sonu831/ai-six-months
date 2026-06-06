# CLAUDE.md — Enterprise AI Sandbox

## Purpose

This repository is a production-realistic AI engineering sandbox focused on:
- Advanced RAG
- Multi-agent orchestration
- Provider-agnostic LLM integrations
- LLMOps-style reliability patterns

It is also the primary hands-on learning system for mastering modern AI engineering end to end.

## Source of Truth Order

1. `.ai/system_instructions.md`
2. `.claude/context_profile.json`
3. `docs/FOLDER_STRUCTURE.md`
4. `docs/SKILLS.md`
5. `docs/LEARNING_HANDBOOK.md`
6. `README.md`

## Authoritative Project Structure

```text
ai-six-months/
├── .ai/
├── .claude/
├── backend/{core,llm_client,vector_store,rag_pipeline,agents}
├── data/{documents,vector_db}
├── docs/{FOLDER_STRUCTURE.md,SKILLS.md,reference/}
├── python-learning/
├── examples/
└── tests/{unit,integration}
```

## Hard Placement Rules

- All application code lives in `backend/`.
- Shared primitives (config, exceptions, logging) live in `backend/core/`.
- LLM provider implementations live in `backend/llm_client/`.
- RAG orchestration lives in `backend/rag_pipeline/`.
- Vector DB adapters live in `backend/vector_store/`.
- Agent orchestration lives in `backend/agents/`.
- External learning reference material lives in top-level `python-learning/`.
- `python-learning/` is local learning material and should not be pushed.

## Runtime Artifacts

- `data/vector_db/` for local vector persistence
- `logs/` for local logs if introduced later

## Execution

```bash
poetry run python examples/01_basic_rag.py
poetry run python examples/02_hybrid_search.py
poetry run python examples/03_agentic_pipeline.py
```
