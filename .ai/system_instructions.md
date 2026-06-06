# System Instructions — Enterprise AI Sandbox

## Operator Profile

- **User:** Elite engineer with 14+ years of experience in systems engineering and AI architecture.
- **Role:** Lead AI Architect / Principal Systems Engineer.
- **Standard:** Production-grade Python, strict typing, async-first I/O, observability-driven design.
- **Expectation:** Every change should be review-ready, testable, and production-hardened.

## Learning Vision

This repository is the user's practical AI learning handbook.
Each implementation should strengthen real-world skills in modern AI systems engineering.

## Non-Negotiable Engineering Rules

### 1. Type Discipline

- Use explicit type hints with `typing` and Pydantic v2 models for all structured data.
- `mypy` strict mode is enforced; all new code must pass typecheck without ignores.
- Never use `Any` as a lazy escape hatch — parametrize generics properly.

### 2. Async-First Boundaries

- All network, database, and file I/O paths **must** use native `async/await`.
- Never place a blocking call inside the event loop.
- Use `async with` context managers for all resource lifecycle (connections, sessions, pools).
- Always prefer connection pooling and `async with` over ad-hoc connection open/close.

### 3. Error Handling & Observability

- Never use naked `try/except` blocks.
- Implement custom domain exceptions via a single-root taxonomy (`EnterpriseAIError` hierarchy).
- Use the standard `logging` library configured with JSON formatting for structured log output.
- Structured logging via `structlog` with JSON renderer for production, ConsoleRenderer for development.
- Every exception must carry a typed context payload for post-mortem analysis.
- Never swallow exceptions silently — propagate or wrap with context.

### 4. Resource Lifecycle

- Manage clients, pools, and sessions with explicit creation and teardown.
- Use `async with` context managers and/or explicit `close()` methods for resource safety.
- Idempotent teardown: `close()` must be safe to call multiple times.
- Optimize for connection pooling (asyncpg pools, httpx shared clients, ChromaDB persistent clients).

### 5. Code Generation Standards

- Zero boilerplate placeholders — every module must be complete and operational.
- No commented-out code blocks; use version control for history.
- Prefer composition over inheritance; injection over hard-wired dependencies.

## Project Structure Contract (Authoritative)

```text
ai-six-months/
├── .ai/
├── .claude/
├── backend/{core,llm_client,vector_store,rag_pipeline,agents}
├── data/{documents,vector_db}
├── docs/{FOLDER_STRUCTURE.md,SKILLS.md,LEARNING_HANDBOOK.md,RAG_ARCHITECTURE.md,reference/}
├── python-learning/
├── examples/
├── tests/{unit,integration}
├── pyproject.toml
└── docker-compose.yml
```

## Placement Rules

- Keep `.ai/` and `.claude/` at root level.
- Keep production code inside `backend/` only.
- Keep vector store implementations separated: `base.py`, `chroma_store.py`, `pg_vector.py` inside `backend/vector_store/`.
- Keep query transformation logic in dedicated `backend/rag_pipeline/query_processor.py`.
- Keep external learning references in top-level `python-learning/`.
- Keep `python-learning/` local-only; do not commit or push it.
- Keep docs and decision records in `docs/`.
- Keep runtime artifacts out of source folders.

## Testing and Quality

- Run `ruff check` and `mypy` before finalizing major changes.
- Prefer small, composable modules with clear interfaces.
- Align significant work with `docs/SKILLS.md` and `docs/LEARNING_HANDBOOK.md`.
- Every major feature should have an example in `examples/` and tests in `tests/`.
