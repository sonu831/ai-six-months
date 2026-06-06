# Enterprise AI Sandbox

Production-grade Python sandbox for Advanced RAG, multi-provider LLM clients, and agentic orchestration.

## Vision

Build one practical learning platform that helps you become an AI master champion by doing real engineering work, not only theory.

This repository is your learning lab and execution ground where you can:
1. Learn the latest AI trends through guided implementation.
2. Practice every concept directly in code.
3. Build production-style instincts step by step.

What we are going to do in this project:
1. Build strong Python fundamentals for AI systems work.
2. Implement modern RAG and retrieval patterns with real trade-offs.
3. Practice multi-provider LLM integration and fallback design.
4. Build agent workflows that are observable, testable, and production-oriented.
5. Turn every module into repeatable learning + implementation milestones.

---

## Architecture Flow Diagram

```
                        USER QUERY
                            │
                            ▼
              ┌─────────────────────────┐
              │   Query Transformation   │  ← LLM (OpenAI / Anthropic)
              │  • 3× Query Rewrites     │
              │  • HyDE Passage Gen      │
              └────────────┬────────────┘
                           │  [original, rewrites..., hyde_passage]
                           ▼
              ┌─────────────────────────┐
              │   Embedding Generation   │  ← Batched API Call
              │   (all variants at once) │
              └────────────┬────────────┘
                           │  [emb_0, emb_1, ..., emb_n]
                           ▼
        ┌──────────────────────────────────────┐
        │         Hybrid Retrieval              │
        │                                      │
        │  ┌──────────────┐  ┌───────────────┐ │
        │  │ Vector ANN   │  │ Keyword BM25  │ │
        │  │ (pgvector /  │  │ (tsvector /   │ │
        │  │  ChromaDB)   │  │  ChromaDB)    │ │
        │  │ cosine <=>   │  │ $contains     │ │
        │  └──────┬───────┘  └──────┬────────┘ │
        │         │                 │          │
        │         └────────┬────────┘          │
        └──────────────────┼───────────────────┘
                           │  [list[SearchResult] × (2N variants)]
                           ▼
              ┌─────────────────────────┐
              │   RRF Fusion             │
              │   score(d) = Σ 1/(k+r)   │
              │   k=60, dedup by chunk_id│
              └────────────┬────────────┘
                           │  top-20 fused candidates
                           ▼
              ┌─────────────────────────┐
              │   Cross-Encoder Rerank   │
              │   • Cohere Rerank v3     │
              │   • HF CrossEncoder     │
              │   • Mock (deterministic) │
              └────────────┬────────────┘
                           │  top-5 re-ranked documents
                           ▼
              ┌─────────────────────────┐
              │   Context Assembly       │
              │   • Token budget check   │
              │   • Chunk-N annotation   │
              └────────────┬────────────┘
                           │  system_prompt + context
                           ▼
              ┌─────────────────────────┐
              │   LLM Generation         │
              │   (Claude / GPT / Ollama)│
              └────────────┬────────────┘
                           │
                           ▼
                     RAGResponse
              { answer, source_chunks,
                query_variants, metrics }
```

---

## Module Map

```
backend/
├── core/               Config, domain exceptions, structured logging
├── llm_client/          Multi-provider abstraction (OpenAI, Anthropic, DeepSeek, Ollama)
├── vector_store/
│   ├── base.py          Abstract BaseVectorStore + Pydantic domain models
│   ├── chroma_store.py  Local persistent ChromaDB (dev / offline / CI)
│   ├── pg_vector.py     PostgreSQL + pgvector (production, asyncpg pool)
│   └── client.py        Backward-compatible re-export
├── rag_pipeline/
│   ├── engine.py        Master RAGEngine: hybrid retrieval, RRF, reranking, generation
│   ├── query_processor.py  QueryTransformer: LLM rewriting + HyDE generation
│   └── __init__.py
└── agents/              ReAct loop, tool registry, agent orchestration
```

---

## Quick Start

### Prerequisites

- Python 3.12+
- [Poetry](https://python-poetry.org/docs/#installation)
- [Docker](https://docs.docker.com/get-docker/) (for PostgreSQL + pgvector)

### Setup

```bash
# Clone and enter the project
git clone <repo-url> && cd ai-six-months

# Install dependencies
poetry install

# Stand up PostgreSQL with pgvector pre-installed
docker compose up -d postgres

# Set your API keys (copy .env.example or set directly)
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
export COHERE_API_KEY="..."   # optional — for Cohere re-ranking

# Run the end-to-end advanced RAG pipeline
poetry run python examples/run_advanced_rag.py
```

### Infrastructure

```yaml
# docker-compose.yml — pgvector-powered PostgreSQL 16
version: "3.9"

services:
  postgres:
    image: pgvector/pgvector:pg16
    container_name: enterprise_ai_postgres
    restart: unless-stopped
    environment:
      POSTGRES_USER: ${DB_USER:-postgres}
      POSTGRES_PASSWORD: ${DB_PASSWORD:-changeme}
      POSTGRES_DB: ${DB_NAME:-enterprise_ai}
    ports:
      - "${DB_PORT:-5432}:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U ${DB_USER:-postgres} -d ${DB_NAME:-enterprise_ai}"]
      interval: 10s
      timeout: 5s
      retries: 5
    networks:
      - ai_net

volumes:
  postgres_data:

networks:
  ai_net:
    driver: bridge
```

Run individual examples:

```bash
poetry run python examples/01_basic_rag.py          # ChromaDB + OpenAI RAG
poetry run python examples/02_hybrid_search.py      # pgvector hybrid search + Cohere
poetry run python examples/03_agentic_pipeline.py   # ReAct agent with tools
poetry run python examples/run_advanced_rag.py      # Full pipeline with mock LLM
```

---

## Important Online References

Use these as trusted external learning references while building in this repository:

1. LinkedIn Learning: Python Essential Training (Ryan Mitchell)
   https://www.linkedin.com/learning/python-essential-training-18764650
2. LinkedIn Learning Instructor Profile: Ryan Mitchell
   https://www.linkedin.com/learning/instructors/ryan-mitchell
3. Python Official Documentation
   https://docs.python.org/3/
4. FastAPI Documentation
   https://fastapi.tiangolo.com/
5. Pydantic Documentation
   https://docs.pydantic.dev/

---

## 14+ Year Lead AI Engineer Progression Map

### Production Pitfalls & Defenses

| # | Pitfall | Defense Built In This Repo |
|---|---------|---------------------------|
| 1 | **Prompt Injection** — user input smuggles directives into system prompts | `_SYSTEM_PROMPT` uses strict `{context}` template; user query never interpolated into system instructions |
| 2 | **Embedding Model Drift** — model update makes all indexed embeddings semantically stale | `metadata` carries `embedding_model` version tag; ingestion pipeline separates model from data |
| 3 | **Token Cost Explosion** — unbounded context windows burn API budget | Hard `max_context_tokens` budget (8K default); tiktoken integration ready; cross-encoder prunes to top-5 |
| 4 | **Connection Pool Exhaustion** — high-concurrency vector ingestion starves pool | asyncpg pool with configurable min/max/query limits; batched upserts; explicit pool lifecycle |
| 5 | **Lost in the Middle** — LLM ignores documents in middle of context window | Cross-encoder re-ranking prunes to exactly top-5; every chunk sits at edge of context |
| 6 | **N+1 Embedding Calls** — embedding each query variant serially | Single batched `EmbeddingRequest(texts=all_variants)` call; all variants embedded concurrently |
| 7 | **Silent Retrieval Degradation** — HNSW recall drifts without monitoring | HNSW index with explicit `m`/`ef_construction`/`ef_search` parameters; REINDEX scheduling documented |
| 8 | **Hard Filter Miss** — metadata filter applied post-retrieval instead of pre-search | JSONB `@>` containment pushes to GIN index BEFORE ANN distance computation; partial index support |
| 9 | **Score Scale Incompatibility** — vector cosine vs. BM25 scores on different scales need normalization | Reciprocal Rank Fusion (RRF) with k=60; score-free rank aggregation; no normalization needed |
| 10 | **Naked Error Swallowing** — `try/except: pass` silently drops failures | Domain exception hierarchy (`EnterpriseAIError` → `VectorStoreError` → `UpsertError`); every exception carries typed context dict |

### Skill Progression Ladder

```
Level 1: FOUNDATION
  └─ Run examples, understand module boundaries, read docs/RAG_ARCHITECTURE.md

Level 2: BUILDER
  └─ Add a new LLM provider, write a custom reranker, extend chunking strategies

Level 3: ARCHITECT
  └─ Design cross-layer improvements: new index topology, streaming pipeline, cache layer

Level 4: OPERATOR
  └─ Add metrics, alerting, canary deployments, A/B testing for retrieval strategies
```

---

## AI Master Curriculum

Full 3-phase programme documented in [`docs/CURRICULUM.md`](docs/CURRICULUM.md).

| Phase | Focus | Weeks |
|-------|-------|-------|
| **1 — Agentic Tooling (AT)** | LLM APIs, tool use, function calling, prompt engineering | 1–4 |
| **2 — RAG** | Embeddings, vector stores, hybrid retrieval, reranking, RAGAS evaluation | 5–9 |
| **3 — Agents & LLMOps** | Multi-agent orchestration, circuit breakers, tracing, CRAG, GraphRAG | 10–16 |

**Learning loop:** Learn → Build → Run → Verify → Record (repeat every week).

**Teaching agents** (invoke by loading the relevant file into context):

| Agent | File | When |
|-------|------|------|
| Phase 1 Tutor | [`.ai/agents/phase1_at_tutor.md`](.ai/agents/phase1_at_tutor.md) | Weeks 1–4 |
| Phase 2 Tutor | [`.ai/agents/phase2_rag_tutor.md`](.ai/agents/phase2_rag_tutor.md) | Weeks 5–9 |
| Phase 3 Tutor | [`.ai/agents/phase3_ops_tutor.md`](.ai/agents/phase3_ops_tutor.md) | Weeks 10–16 |
| Eval Agent | [`.ai/agents/eval_agent.md`](.ai/agents/eval_agent.md) | End of any week — quiz + checkpoint |

**Session start:** tell Claude "I'm on Week N, use the [phase] tutor."
**Checkpoint:** tell Claude "Quiz me on Week N" — eval agent runs 5 questions + code audit.

---

## Canonical Structure

```text
ai-six-months/
├── .ai/
│   └── system_instructions.md          ← Operator profile & engineering rules
├── .claude/
│   └── context_profile.json            ← Structured AI context metadata
├── CLAUDE.md / AGENTS.md               ← Agent governance documents
├── backend/
│   ├── agents/                         ← ReAct loop, tool registry, orchestrator
│   ├── core/                           ← Config, exceptions, structured logging
│   ├── llm_client/                     ← Multi-provider clients (OpenAI, Anthropic, DeepSeek, Ollama)
│   ├── rag_pipeline/                   ← Query transformation, hybrid retrieval, RRF, reranking
│   └── vector_store/                   ← BaseVectorStore, ChromaDB, pgvector implementations
├── data/
│   ├── documents/                      ← Source ingestion corpus
│   └── vector_db/                      ← Local vector persistence (git-ignored)
├── docs/
│   ├── FOLDER_STRUCTURE.md             ← Authoritative placement guide
│   ├── SKILLS.md                       ← Project learning tracks & mastery levels
│   ├── LEARNING_HANDBOOK.md            ← Step-by-step learning execution plan
│   ├── RAG_ARCHITECTURE.md             ← Production RAG design guide
│   └── reference/                      ← External reference docs
├── python-learning/                    ← External learning material (git-ignored, local-only)
├── examples/                           ← Runnable end-to-end demos
├── tests/                              ← Unit & integration test boundaries
├── pyproject.toml                      ← Poetry config + mypy/ruff/pytest settings
└── docker-compose.yml                  ← Local infrastructure (pgvector, pgAdmin, Ollama)
```

---

## Folder Rules

1. Keep `.ai/` and `.claude/` at repository root.
2. Keep all production code in `backend/` only.
3. Keep vector store implementations separated: `base.py`, `chroma_store.py`, `pg_vector.py`.
4. Keep query transformation logic in dedicated `backend/rag_pipeline/query_processor.py`.
5. Keep runtime artifacts in `data/vector_db/`.
6. Keep external learning material in top-level `python-learning/`.
7. Keep examples runnable from `examples/`.
8. Do not push local learning assets from `python-learning/` to git.

---

## Testing & Quality

```bash
poetry run ruff check .          # Lint
poetry run mypy backend/         # Strict type check
poetry run pytest                # Run all tests with coverage
```

---

## Notes

- This repo targets Python 3.12 and async-first architecture.
- `docs/FOLDER_STRUCTURE.md` is the authoritative placement guide.
- `docs/SKILLS.md` defines project learning tracks and implementation focus.
- `docs/LEARNING_HANDBOOK.md` is your step-by-step learning and execution handbook.
- `docs/RAG_ARCHITECTURE.md` is the comprehensive production RAG design reference.
- `python-learning/` is local learning material and is git-ignored by policy.
