# AGENTS Guide — Enterprise AI Sandbox

## Mission
Build and evolve a production-grade AI sandbox that serves as a hands-on learning system to become an AI master champion.

## Vision

Every task should improve both:
1. The codebase quality.
2. Your AI engineering capability.

Agents should treat this repository as a practical learning handbook where each change is a teachable, production-style step.

## Governance Priority
1. `.ai/system_instructions.md`
2. `.claude/context_profile.json`
3. `docs/FOLDER_STRUCTURE.md`
4. `docs/SKILLS.md`
5. `docs/LEARNING_HANDBOOK.md`
6. `README.md`

## Agent Rules

1. Keep `.ai/` and `.claude/` at root.
2. Keep code changes inside `backend/` unless task is docs/tests/examples.
3. Keep external imports and learning references in top-level `python-learning/`.
4. Do not move runtime artifacts into source folders.
5. For architecture changes, update README and folder-structure docs in the same change.
6. For any learning-flow change, update `docs/SKILLS.md` and `docs/LEARNING_HANDBOOK.md`.
7. Prefer changes that include practice points in `examples/` and validation points in `tests/`.
8. Do not commit `python-learning/`; it is local-only by project policy.

## Expected Delivery Quality

- Strict type hints
- Async-safe I/O
- Structured logging
- Domain exceptions
- Tests for behavior changes
- Clear learning value for each major change
