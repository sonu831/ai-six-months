# Enterprise AI Sandbox

> A production-grade AI engineering dojo. 16-week programme to master Agentic Tooling, Advanced RAG, and Multi-Agent LLMOps — by building real systems, not reading tutorials.

---

## Mission

This is not a demo repo. Every implementation is production-realistic: typed, async-first, tested, observable.

The goal is to reach **Operator-level AI engineering mastery** — the ability to design, build, evaluate, and operate advanced AI systems in production.

---

## 3-Phase Curriculum

Full curriculum: [`docs/CURRICULUM.md`](docs/CURRICULUM.md)

| Phase | Focus | Weeks | Mastery Target |
|-------|-------|-------|----------------|
| **1 — Agentic Tooling (AT)** | LLM APIs, provider abstraction, tool use, function calling, prompt engineering | 1–4 | Builder |
| **2 — RAG** | Embeddings, vector stores, hybrid retrieval, reranking, RAGAS evaluation | 5–9 | Architect |
| **3 — Agents & LLMOps** | Multi-agent orchestration, circuit breakers, tracing, CRAG, GraphRAG | 10–16 | Operator |

### Learning Loop (every week)

```
Learn → Build → Run → Verify → Record
```

1. **Learn** — read concept material in this doc or `python-learning/`
2. **Build** — implement in the correct `backend/` layer
3. **Run** — validate via an `examples/` script
4. **Verify** — add/update tests in `tests/`
5. **Record** — log insights in `docs/LEARNING_HANDBOOK.md`

### Teaching Agents

| Agent | File | When |
|-------|------|------|
| Phase 1 Tutor | [`.ai/agents/phase1_at_tutor.md`](.ai/agents/phase1_at_tutor.md) | Weeks 1–4 — LLM APIs, tool use |
| Phase 2 Tutor | [`.ai/agents/phase2_rag_tutor.md`](.ai/agents/phase2_rag_tutor.md) | Weeks 5–9 — RAG pipeline |
| Phase 3 Tutor | [`.ai/agents/phase3_ops_tutor.md`](.ai/agents/phase3_ops_tutor.md) | Weeks 10–16 — Agents, LLMOps |
| Eval Agent | [`.ai/agents/eval_agent.md`](.ai/agents/eval_agent.md) | End of any week — quiz + checkpoint |

**Starting a session:** "I'm on Week N. Use the Phase X tutor."
**Weekly checkpoint:** "Quiz me on Week N." — eval agent runs 5 questions + code audit → PASS / FAIL.

---

## Architecture

Full pipeline from user query to grounded answer:

```
                        USER QUERY
                            │
                            ▼
              ┌─────────────────────────┐
              │   Query Transformation   │  ← QueryTransformer (LLM)
              │  • 3× Query Rewrites     │     ┌─ _REWRITE_SYSTEM prompt
              │  • HyDE Passage Gen      │     └─ _HYDE_SYSTEM prompt
              └────────────┬────────────┘
                           │  [original, rewrites..., hyde_passage]
                           ▼
              ┌─────────────────────────┐
              │   Embedding Generation   │  ← Batched EmbeddingRequest
              │   (all variants at once) │     embed(texts=all_variants)
              └────────────┬────────────┘
                           │  [emb_0, emb_1, ..., emb_n]
                           ▼
        ┌──────────────────────────────────────┐
        │         Hybrid Retrieval              │
        │  ┌──────────────┐  ┌───────────────┐ │
        │  │ Vector ANN   │  │ Keyword BM25  │ │
        │  │ pgvector:    │  │ pgvector:     │ │
        │  │  embedding   │  │  ts_rank_cd() │ │
        │  │  <=> $1      │  │  @@ tsquery   │ │
        │  │ ChromaDB:    │  │ ChromaDB:     │ │
        │  │  cosine dist │  │  $contains    │ │
        │  └──────┬───────┘  └──────┬────────┘ │
        │         └────────┬────────┘          │
        └──────────────────┼───────────────────┘
                           │  2N concurrent search tasks
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
              │   • Cohere Rerank v3     │  (API)
              │   • HF CrossEncoder      │  (local)
              │   • ScorePassthrough     │  (no-op)
              └────────────┬────────────┘
                           │  top-5 re-ranked documents
                           ▼
              ┌─────────────────────────┐
              │   Context Assembly       │
              │   • Token budget (8K)    │
              │   • [Chunk-N] annotation │
              └────────────┬────────────┘
                           ▼
              ┌─────────────────────────┐
              │   LLM Generation         │
              │   Claude / GPT / Ollama  │
              └────────────┬────────────┘
                           ▼
                     RAGResponse
              { answer, source_chunks,
                query_variants, metrics }
```

---

## What's Built

### Layer Map

```
backend/
├── core/               Config (pydantic-settings v2), 22-class exception hierarchy, structlog
├── llm_client/         Provider abstraction — OpenAI, Anthropic, DeepSeek, Ollama
├── vector_store/       BaseVectorStore + ChromaDB (dev) + pgvector (production)
├── rag_pipeline/       QueryTransformer (HyDE + rewrites) + RAGEngine (hybrid retrieval, RRF, rerank)
└── agents/             LangGraph ReAct loop — ToolRegistry, step guard, max_steps safety
```

### Provider Matrix

| Capability | Providers |
|-----------|-----------|
| Generation | OpenAI · Anthropic · DeepSeek · Ollama |
| Embeddings | OpenAI · Ollama |
| Re-ranking | Cohere Rerank v3 · HuggingFace CrossEncoder · ScorePassthrough |
| Vector DB | pgvector (asyncpg + HNSW) · ChromaDB (local persistent) |

Switch provider without code changes:
```bash
PROVIDER=deepseek poetry run python examples/01_basic_rag.py
```

### Examples

| Script | What it shows | Requirements |
|--------|---------------|--------------|
| `examples/01_basic_rag.py` | ChromaDB + OpenAI: chunking, embeddings, full RAG | OpenAI key |
| `examples/02_hybrid_search.py` | pgvector: vector vs keyword vs hybrid RRF side-by-side | Docker + OpenAI key |
| `examples/03_agentic_pipeline.py` | ReAct agent: RAG tool + calculator, multi-step reasoning | OpenAI + Anthropic keys |
| `examples/run_advanced_rag.py` | Self-contained mock pipeline — no API keys needed | **None** |

---

## Quick Start

**Prerequisites:** Python 3.12+, [Poetry](https://python-poetry.org), [Docker](https://docs.docker.com/get-docker/)

```bash
# Install
git clone <repo-url> && cd ai-six-months
poetry install

# Infrastructure (PostgreSQL + pgvector)
docker compose up -d postgres

# API keys
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
export COHERE_API_KEY="..."       # optional — for Cohere reranking

# Run the no-key demo first
poetry run python examples/run_advanced_rag.py

# Then the full pipeline
poetry run python examples/01_basic_rag.py
```

---

## Production Patterns Built In

Every pitfall below is handled in the codebase — not documented as future work.

| Pitfall | Defense |
|---------|---------|
| **Prompt injection** | `_SYSTEM_PROMPT` uses strict `{context}` template; user query never interpolated into system instructions |
| **Embedding model drift** | Metadata carries model version; re-embedding architecture decoupled from data |
| **Token cost explosion** | Hard `max_context_tokens=8k` budget; tiktoken; cross-encoder prunes to top-5 |
| **Connection pool exhaustion** | asyncpg pool with `min=5/max=20/max_queries=50k`; explicit lifecycle |
| **Lost in the middle** | Cross-encoder prunes to top-5; `[Chunk-N]` citation format |
| **N+1 embedding calls** | Single batched `EmbeddingRequest(texts=all_variants)` per pipeline run |
| **Silent retrieval degradation** | HNSW `m/ef_construction/ef_search` explicit; REINDEX schedule in `docs/RAG_ARCHITECTURE.md` |
| **Metadata filter miss** | JSONB `@>` pushes to GIN index before ANN distance; filter is pre-retrieval |
| **RRF score incompatibility** | RRF k=60; rank-based fusion — no scale normalization needed |
| **Naked exception swallowing** | 22-class `EnterpriseAIError` hierarchy; every exception carries typed `context: dict` |

---

## Mastery Progression

```
Level 1 — FOUNDATION    Run all examples. Understand module boundaries.
Level 2 — BUILDER       Add a provider, write a custom reranker, extend chunking.
Level 3 — ARCHITECT     Design cross-layer improvements, new index topology, streaming.
Level 4 — OPERATOR      Add metrics, alerting, eval harness, drift detection, A/B retrieval.
```

Key files to master (in order):

```
1.  docs/LEARNING_HANDBOOK.md            ← How to use this repo
2.  docs/CURRICULUM.md                   ← 16-week programme
3.  backend/core/exceptions.py           ← Error taxonomy
4.  backend/core/config.py               ← Settings architecture
5.  backend/llm_client/base.py           ← Provider abstraction contract
6.  backend/llm_client/openai_client.py  ← Concrete provider with retry
7.  backend/vector_store/base.py         ← Vector store contract
8.  backend/vector_store/pg_vector.py    ← Production vector backend
9.  backend/rag_pipeline/query_processor.py  ← HyDE + query rewriting
10. backend/rag_pipeline/engine.py       ← Full pipeline orchestration
11. backend/agents/orchestrator.py       ← ReAct agent with LangGraph
12. docs/RAG_ARCHITECTURE.md             ← Deep theory + trade-offs
```

---

## Quality

```bash
poetry run ruff check .        # Lint
poetry run mypy backend/       # Strict type check (Python 3.12)
poetry run pytest              # Tests with coverage (target: 80%)
```

---

## Reference Docs

| Doc | Purpose |
|-----|---------|
| [`docs/CURRICULUM.md`](docs/CURRICULUM.md) | 3-phase programme, weekly exercises, checkpoint questions |
| [`docs/RAG_ARCHITECTURE.md`](docs/RAG_ARCHITECTURE.md) | Chunking, HNSW vs IVFFlat, RRF, HyDE, security hardening |
| [`docs/SKILLS.md`](docs/SKILLS.md) | 5 skill tracks, mastery levels, execution rule |
| [`docs/LEARNING_HANDBOOK.md`](docs/LEARNING_HANDBOOK.md) | Step-by-step learning execution plan |
| [`docs/FOLDER_STRUCTURE.md`](docs/FOLDER_STRUCTURE.md) | Authoritative file placement rules |
| [`.ai/system_instructions.md`](.ai/system_instructions.md) | Engineering non-negotiables (types, async, errors, logging) |

---

## Governance

Source of truth order (descending):
1. `.ai/system_instructions.md`
2. `.claude/context_profile.json`
3. `docs/FOLDER_STRUCTURE.md`
4. `docs/SKILLS.md`
5. `docs/LEARNING_HANDBOOK.md`
6. `README.md`

Hard rules:
- All production code lives in `backend/` only
- `python-learning/` is local-only and gitignored — never push it
- No `print()` — structlog only
- No naked exceptions — every error carries a typed context payload
- No blocking I/O in the event loop — everything is `async/await`
