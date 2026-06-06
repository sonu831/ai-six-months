# Enterprise AI Sandbox

Production-grade Python sandbox for Advanced RAG, multi-provider LLM clients, and agentic orchestration.

## Vision

Build one practical learning platform that helps you become an AI master champion by doing real engineering work, not only theory.

This repository is your learning lab and execution ground where you can:
1. Learn the latest AI trends through guided implementation.
2. Practice every concept directly in code.
3. Build production-style instincts step by step.

## Canonical Structure

```text
ai-six-months/
├── .ai/
│   └── system_instructions.md
├── .claude/
│   └── context_profile.json
├── CLAUDE.md
├── AGENTS.md
├── backend/
│   ├── agents/
│   ├── core/
│   ├── llm_client/
│   ├── rag_pipeline/
│   └── vector_store/
├── data/
│   ├── documents/
│   └── vector_db/
├── docs/
│   ├── FOLDER_STRUCTURE.md
│   ├── SKILLS.md
│   └── reference/
├── python-learning/
├── examples/
├── tests/
│   ├── unit/
│   └── integration/
├── pyproject.toml
└── docker-compose.yml
```

## Folder Rules

1. Keep `.ai/` and `.claude/` at repository root.
2. Keep all production code in `backend/` only.
3. Keep runtime artifacts in `data/vector_db/`.
4. Keep external learning material in top-level `python-learning/`.
5. Keep examples runnable from `examples/`.
6. Do not push local learning assets from `python-learning/` to git.

## Quick Start

```bash
poetry install
docker compose up -d postgres
poetry run python examples/01_basic_rag.py
```

## Notes

- This repo targets Python 3.12 and async-first architecture.
- `docs/FOLDER_STRUCTURE.md` is the authoritative placement guide.
- `docs/SKILLS.md` defines project learning tracks and implementation focus.
- `docs/LEARNING_HANDBOOK.md` is your step-by-step learning and execution handbook.
- `python-learning/` is local learning material and is git-ignored by policy.
