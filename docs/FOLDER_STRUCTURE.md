# Folder Structure Contract

## Root Layout

```text
ai-six-months/
├── .ai/
├── .claude/
├── AGENTS.md
├── CLAUDE.md
├── README.md
├── backend/
├── data/
├── docs/
├── python-learning/
├── examples/
├── tests/
├── pyproject.toml
└── docker-compose.yml
```

## Placement Rules

- `backend/core/`: config, exceptions, logging.
- `backend/llm_client/`: provider clients and interfaces.
- `backend/vector_store/`: vector database implementations.
- `backend/rag_pipeline/`: retrieval/rerank/generation orchestration.
- `backend/agents/`: agent loops, tools, orchestration logic.
- `data/documents/`: source ingestion corpus.
- `data/vector_db/`: local vector persistence.
- `python-learning/`: external learning references and imported study material.
- `examples/`: runnable demos.
- `tests/unit/` and `tests/integration/`: test boundaries.
- `docs/LEARNING_HANDBOOK.md`: learning execution plan and milestone tracking.

## Push Policy

- `python-learning/` is local learning material and should remain git-ignored.
- Runtime and generated outputs stay ignored (`data/vector_db/`, `logs/`, caches).

## Anti-Patterns

- No production source code in `docs/`, `data/`, or `examples/`.
- No ad-hoc top-level folders without an explicit reason.
- No hidden runtime output mixed with tracked source.
