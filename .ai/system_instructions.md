# System Instructions — Enterprise AI Sandbox

## Operator Profile
- Role: Lead AI Architect / Principal Systems Engineer
- Standard: Production-grade Python, strict typing, async-first I/O
- Expectation: Every change should be review-ready and testable

## Learning Vision

This repository is the user's practical AI learning handbook.
Each implementation should strengthen real-world skills in modern AI systems engineering.

## Non-Negotiable Engineering Rules

1. Type discipline
- Use explicit type hints and mypy-compatible patterns.
- Use Pydantic v2 models for structured data.

2. Async-first boundaries
- All network, DB, and file I/O paths must be async-native where possible.
- Avoid blocking calls inside the event loop.

3. Error handling and observability
- Use a domain exception hierarchy rooted at `EnterpriseAIError`.
- Use structured logging through `structlog`.
- Never swallow exceptions silently.

4. Resource lifecycle
- Manage clients/sessions with explicit creation and teardown.
- Prefer context managers for resource safety.

## Project Structure Contract (Authoritative)

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

## Placement Rules

- Keep `.ai/` and `.claude/` at root level.
- Keep production code inside `backend/` only.
- Keep external learning references in top-level `python-learning/`.
- Keep `python-learning/` local-only; do not commit or push it.
- Keep docs and decision records in `docs/`.
- Keep runtime artifacts out of source folders.

## Testing and Quality

- Run lint/type/test checks before finalizing major changes.
- Prefer small, composable modules with clear interfaces.
- Align significant work with `docs/SKILLS.md` and `docs/LEARNING_HANDBOOK.md`.
