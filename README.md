# Enterprise AI Sandbox

Production-grade Python sandbox for Advanced RAG, multi-provider LLM clients, and agentic orchestration.

## Vision

Build one practical learning platform that helps you become an **AI master champion** by doing real engineering work, not only theory. Every module is a teachable, production-style implementation.

---

## Architecture Flow Diagram

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
        │                                      │
        │  ┌──────────────┐  ┌───────────────┐ │
        │  │ Vector ANN   │  │ Keyword BM25  │ │
        │  │ pgvector:    │  │ pgvector:     │ │
        │  │  embedding   │  │  ts_rank_cd() │ │
        │  │  <=> $1      │  │  @@ tsquery   │ │
        │  │ ChromaDB:    │  │ ChromaDB:     │ │
        │  │  cosine dist │  │  $contains    │ │
        │  └──────┬───────┘  └──────┬────────┘ │
        │         │                 │          │
        │         └────────┬────────┘          │
        └──────────────────┼───────────────────┘
                           │  2N concurrent search tasks
                           ▼
              ┌─────────────────────────┐
              │   RRF Fusion             │
              │   score(d) = Σ 1/(k+r)   │
              │   k=60, dedup by chunk_id│
              └────────────┬────────────┘
                           │  top-20 fused SearchResults
                           ▼
              ┌─────────────────────────┐
              │   Cross-Encoder Rerank   │
              │   • Cohere Rerank v3     │  (API)
              │   • HF CrossEncoder     │  (local, ms-marco-MiniLM)
              │   • ScorePassthrough    │  (no-op, by fusion score)
              └────────────┬────────────┘
                           │  top-5 re-ranked documents
                           ▼
              ┌─────────────────────────┐
              │   Context Assembly       │
              │   • Token budget (8K)    │
              │   • [Chunk-N] annotation │
              │   • 0.75 words/token est │
              └────────────┬────────────┘
                           │  system_prompt + context
                           ▼
              ┌─────────────────────────┐
              │   LLM Generation         │
              │   • Claude / GPT-4o     │
              │   • DeepSeek / Ollama   │
              │   • temp=0.1, max=1024  │
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
├── __init__.py               ← Package root
├── core/                     ← Config, exceptions, logging
│   ├── config.py             ← Settings (pydantic-settings v2) + sub-configs
│   ├── exceptions.py         ← 22-class domain exception hierarchy
│   └── logging.py            ← structlog bootstrap (JSON/Console renderer)
├── llm_client/               ← Multi-provider abstraction layer
│   ├── base.py               ← BaseLLMClient ABC + Pydantic request/response models
│   ├── openai_client.py      ← OpenAI (GPT-4o, tenacity retry, batched embeddings)
│   ├── anthropic_client.py   ← Anthropic (Claude, system/turn split, no embeddings)
│   ├── deepseek_client.py    ← DeepSeek (OpenAI-compatible, own settings class)
│   └── ollama_client.py      ← Ollama (local REST API, httpx, per-text embedding)
├── vector_store/             ← Vector DB abstractions + implementations
│   ├── base.py               ← BaseVectorStore ABC + DocumentChunk / SearchResult
│   ├── pg_vector.py          ← PostgreSQL+pgvector (asyncpg pool, HNSW, cosine <=>)
│   ├── chroma_store.py       ← ChromaDB (local persistent, asyncio.to_thread)
│   └── client.py             ← Backward-compatible re-export
├── rag_pipeline/             ← Retrieval & synthesis orchestration
│   ├── engine.py             ← RAGEngine, reranker hierarchy, RRF, models
│   └── query_processor.py    ← QueryTransformer (rewrites + HyDE)
└── agents/                   ← Agentic orchestration
    └── orchestrator.py       ← AgentOrchestrator, ToolRegistry, ReAct loop (LangGraph)

data/
├── documents/sample.md       ← 139-line enterprise knowledge base
└── vector_db/                ← Local vector persistence (git-ignored)

docs/
├── FOLDER_STRUCTURE.md       ← Authoritative placement contract
├── SKILLS.md                 ← 5 skill tracks + mastery levels
├── LEARNING_HANDBOOK.md      ← Step-by-step learning execution plan
├── RAG_ARCHITECTURE.md       ← 394-line production RAG design reference
└── reference/README.md       ← External reference notes

examples/
├── 01_basic_rag.py           ← ChromaDB + OpenAI RAG (no Docker needed)
├── 02_hybrid_search.py       ← pgvector hybrid search + Cohere/CrossEncoder rerank
├── 03_agentic_pipeline.py    ← ReAct agent with RAG + custom tools
└── run_advanced_rag.py       ← Self-contained mock pipeline (no API keys)

tests/                        ← Unit + integration test boundaries
python-learning/              ← External learning material (git-ignored, local-only)
```

---

## Quick Start

### Prerequisites

| Tool | Version | Purpose |
|------|---------|---------|
| Python | 3.12+ | Runtime |
| Poetry | latest | Dependency management |
| Docker | latest | PostgreSQL + pgvector infrastructure |
| API Keys | — | OpenAI, Anthropic, Cohere (optional) |

### Setup

```bash
# Clone and enter the project
git clone <repo-url> && cd ai-six-months

# Install all dependencies (core + dev)
poetry install

# Optional: install reranking extras
poetry install --extras "cohere-rerank"
poetry install --extras "local-rerank"

# Stand up PostgreSQL 16 with pgvector pre-installed
docker compose up -d postgres

# Verify postgres is healthy
docker compose ps

# Optional: pgAdmin web UI at http://localhost:5050 (credentials: admin@local.dev / admin)
docker compose --profile dev-tools up -d pgadmin

# Optional: Ollama for local LLM inference
docker compose --profile local-llm up -d ollama
docker exec enterprise_ai_ollama ollama pull llama3.2
```

### Environment

```bash
# Required — copy and fill in your keys
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."

# Optional — Cohere re-ranking
export COHERE_API_KEY="..."

# Optional — override PostgreSQL defaults
export DB_HOST="localhost"
export DB_PORT="5432"
export DB_NAME="enterprise_ai"
export DB_USER="postgres"
export DB_PASSWORD="changeme"

# Optional — override model selection
export OPENAI_DEFAULT_MODEL="gpt-4o"
export ANTHROPIC_DEFAULT_MODEL="claude-sonnet-4-6"
```

### Run Examples

```bash
# Example 1: ChromaDB local RAG (no Docker, works offline)
# Uses: OpenAI for embeddings + generation, ScorePassthroughReranker
poetry run python examples/01_basic_rag.py

# Example 2: pgvector hybrid search deep-dive (needs Docker postgres)
# Compares vector-only vs keyword-only vs hybrid RRF side-by-side
poetry run python examples/02_hybrid_search.py

# Example 3: Agentic pipeline with ReAct loop
# Multi-step reasoning: agent calls RAG + calculator tools autonomously
poetry run python examples/03_agentic_pipeline.py

# Example 4: Self-contained advanced RAG (no API keys needed)
# Mock LLM + deterministic embeddings, 6 documents, 4 queries, formatted metrics
poetry run python examples/run_advanced_rag.py
```

---

## Infrastructure

```yaml
# docker-compose.yml — full local infrastructure stack
version: "3.9"

services:
  # PostgreSQL 16 with pgvector extension pre-installed
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

  # pgAdmin 4 web UI — http://localhost:5050 (profile: dev-tools)
  pgadmin:
    image: dpage/pgadmin4:8
    container_name: enterprise_ai_pgadmin
    environment:
      PGADMIN_DEFAULT_EMAIL: admin@local.dev
      PGADMIN_DEFAULT_PASSWORD: admin
    ports:
      - "5050:80"
    depends_on:
      postgres:
        condition: service_healthy
    profiles:
      - dev-tools

  # Ollama local LLM server — port 11434 (profile: local-llm)
  ollama:
    image: ollama/ollama:latest
    container_name: enterprise_ai_ollama
    ports:
      - "11434:11434"
    volumes:
      - ollama_data:/root/.ollama
    profiles:
      - local-llm

volumes:
  postgres_data:
  pgadmin_data:
  ollama_data:

networks:
  ai_net:
    driver: bridge
```

---

## Module Reference

### `backend/core/` — Foundation Layer

#### Config (`config.py`)

Class hierarchy using `pydantic-settings` v2 with env-prefix groups:

| Settings Class | Env Prefix | Key Fields | Defaults |
|---|---|---|---|
| `DatabaseSettings` | `DB_` | host, port, name, user, password, pool_min/max_size | `localhost:5432/enterprise_ai` |
| `OpenAISettings` | `OPENAI_` | api_key, default_model, embedding_model, max_retries | `gpt-4o`, `text-embedding-3-large` |
| `AnthropicSettings` | `ANTHROPIC_` | api_key, default_model, max_retries | `claude-sonnet-4-6` |
| `OllamaSettings` | `OLLAMA_` | base_url, default_model, embedding_model | `localhost:11434`, `llama3.2` |
| `CohereSettings` | `COHERE_` | api_key, rerank_model | `rerank-english-v3.0` |
| `ChromaSettings` | `CHROMA_` | persist_directory, collection_name | `./data/vector_db/chroma` |
| `Settings` | (root) | environment, log_level + all sub-configs | `development`, `INFO` |

`get_settings()` returns a process-wide `@lru_cache` singleton. Reset in tests with `monkeypatch.setattr("backend.core.config._settings", None)`.

#### Exceptions (`exceptions.py`)

22-class typed hierarchy rooted at `EnterpriseAIError(message, context: dict | None)`:

```
EnterpriseAIError                   ← root, carries {key: value} context
├── ConfigurationError
├── VectorStoreError
│   ├── EmbeddingError              (dimension mismatch)
│   ├── UpsertError                 (batch write failure)
│   ├── SearchError                 (similarity/keyword search failure)
│   └── MigrationError              (DDL execution failure)
├── LLMClientError
│   ├── RateLimitError              (HTTP 429)
│   ├── AuthenticationError         (HTTP 401/403)
│   ├── ContextLengthError          (prompt exceeds window)
│   ├── ProviderUnavailableError    (HTTP 5xx / unreachable)
│   └── ModelNotFoundError          (model doesn't exist)
├── RAGPipelineError
│   ├── QueryTransformationError    (rewrite/HyDE failure)
│   ├── RetrievalError              (zero usable candidates)
│   ├── ReRankingError              (cross-encoder/Cohere failure)
│   └── ContextAssemblyError        (token budget exceeded)
└── AgentError
    ├── AgentLoopDetectedError      (max_steps exceeded)
    ├── AgentTimeoutError           (wall-clock timeout)
    └── ToolExecutionError          (tool raised exception)
```

Every exception carries a typed `context: dict[str, Any]` for post-mortem analysis. `__str__` formats as `message [key=value, ...]`.

#### Logging (`logging.py`)

- `configure_logging(log_level, environment)` — call once at startup.
- Development: `ConsoleRenderer(colors=True)` for human-friendly output.
- Production: `JSONRenderer` for machine-parseable structured logs.
- Silences noisy third-party loggers (`httpx`, `httpcore`, `asyncpg`, `chromadb`, `openai`, `anthropic`) to `WARNING`.
- `get_logger(name)` — returns a `structlog.stdlib.BoundLogger`.

---

### `backend/llm_client/` — Multi-Provider LLM Layer

#### Base Contract (`base.py`)

Pydantic v2 domain models:
- `ChatMessage(role: "system"|"user"|"assistant", content: str)`
- `ChatCompletionRequest(messages, model, temperature, max_tokens, stream)`
- `ChatCompletionResponse(content, model, prompt_tokens, completion_tokens, total_tokens)`
- `EmbeddingRequest(texts, model)` / `EmbeddingResponse(embeddings, model, total_tokens)`
- `ProviderInfo(name, default_model, embedding_model, supports_streaming, max_context_tokens)`

Abstract class `BaseLLMClient` with 6 methods: `chat_complete`, `stream_chat`, `embed`, `health_check`, `provider_info` (sync), `close`.

#### Provider Comparison

| Feature | OpenAI | Anthropic | DeepSeek | Ollama |
|---------|--------|-----------|----------|--------|
| **Chat** | GPT-4o | Claude Sonnet | deepseek-chat | llama3.2 |
| **Embeddings** | text-embedding-3-large | — (not supported) | — (not supported) | nomic-embed-text |
| **Streaming** | Yes | Yes (stream manager) | Yes | Yes (SSE lines) |
| **Retry** | tenacity (3 retries, exp backoff) | tenacity (3 retries, exp backoff) | tenacity (3 retries, exp backoff) | httpx timeout only |
| **System prompt** | In messages array | Separate `system` param | In messages array | In messages array |
| **HTTP client** | `openai.AsyncOpenAI` | `anthropic.AsyncAnthropic` | `openai.AsyncOpenAI` (wrapped) | `httpx.AsyncClient` |
| **Config source** | `get_settings().openai` | `get_settings().anthropic` | Own `DeepSeekSettings()` | `get_settings().ollama` |
| **Exported** | Yes | Yes | No | Yes |

**OpenAI** (`openai_client.py`): SDK `max_retries=0` so tenacity manages all retry. Error mapping: `RateLimitError`, `AuthenticationError`, `ModelNotFoundError`, `ContextLengthError` (BadRequestError), `ProviderUnavailableError` (APIConnectionError). Embedding calls are batched in a single request.

**Anthropic** (`anthropic_client.py`): `_split_messages` transparently separates system messages from conversation turns (Anthropic's API requires separate `system` kwarg). Default `max_tokens=4096`. Content concatenated from multiple `TextBlock` blocks.

**DeepSeek** (`deepseek_client.py`): Uses the OpenAI SDK pointed at `https://api.deepseek.com/v1`. Has its own local `DeepSeekSettings` class (env prefix `DEEPSEEK_`). Not exported from `__init__.py`. Creates settings on every `chat_complete` call.

**Ollama** (`ollama_client.py`): REST API via `httpx.AsyncClient` with connection pooling (`max_connections=20`, `max_keepalive=10`). Embeddings done **one text at a time** in a sequential loop. Token count approximated by word count. No retry mechanism.

---

### `backend/vector_store/` — Vector Database Layer

#### Domain Models (`base.py`)

- `DocumentChunk(chunk_id, document_id, content, embedding, metadata)` — the indexable unit.
- `SearchResult(chunk_id, document_id, content, score, metadata)` — retrieved chunk with relevance score.
- `BaseVectorStore(ABC)` — 6 abstract methods: `upsert_embeddings`, `similarity_search`, `keyword_search`, `delete_document`, `health_check`, `close`.

#### Backend Comparison

| Feature | PgVectorStore | ChromaVectorStore |
|---------|--------------|-------------------|
| **Target env** | Production | Dev / offline / CI |
| **Connection** | asyncpg connection pool | Local persistent file |
| **ANN index** | HNSW (m=16, ef_construction=64) | HNSW (cosine space) |
| **Distance operator** | `embedding <=> $1` (cosine) | Cosine distance in SDK |
| **Keyword search** | tsvector BM25 (`ts_rank_cd`) | `$contains` substring |
| **Metadata filter** | JSONB `@>` with GIN index | `$eq` operator per key |
| **Deduplication** | `ON CONFLICT (chunk_id) DO UPDATE` | Native upsert |
| **Async safety** | Native async (asyncpg) | `asyncio.to_thread` wrapping |
| **Schema migration** | Auto in `create()` (DDL) | Auto in `create()` (get_or_create) |
| **Transaction** | Yes (`conn.transaction()`) | No |

**PgVectorStore** (`pg_vector.py`):
- Table: `document_chunks(chunk_id TEXT PK, document_id TEXT, content TEXT, embedding vector(N), metadata JSONB, created_at, updated_at)`.
- 4 indexes: btree on `document_id`, HNSW on `embedding` (cosine), GIN on `to_tsvector('english', content)`, GIN on `metadata`.
- `_build_jsonb_filter` converts flat metadata dicts to parameterized `metadata @> $N::jsonb` clauses.
- Connection pool config: `min_size=5`, `max_size=20`, `max_queries=50,000`, `statement_cache_size=100`, `command_timeout=30`.

**ChromaVectorStore** (`chroma_store.py`):
- All sync operations via `asyncio.to_thread`.
- Collection created with `{"hnsw:space": "cosine"}` metadata.
- Metadata serialization: non-primitive values `json.dumps()`-ed.
- `close()` is a no-op (relies on PersistentClient GC).

---

### `backend/rag_pipeline/` — RAG Orchestration Layer

#### Models (`engine.py`)

- `RAGConfig(retrieval_top_k=20, rerank_top_k=5, hybrid_alpha=0.7, max_context_tokens=8_000, query_transform_enabled=True, transform_variants=3, enable_hyde=True)`.
- `RAGRequest(query, document_filter, config_override, conversation_history)`.
- `RAGResponse(answer, source_chunks, query_variants, retrieval_candidate_count, final_context_count, model_used)`.

#### Reranker Hierarchy

| Class | Strategy | Requirements | Score Range |
|---|---|---|---|
| `BaseReranker` (ABC) | Contract: `rerank(query, candidates, top_k)` | — | — |
| `CohereReranker` | Cohere Rerank v3 API | `COHERE_API_KEY` | 0.0–1.0 (relevance) |
| `HuggingFaceCrossEncoderReranker` | Local `CrossEncoder` model | `pip install sentence-transformers` | 0.0–1.0 (logit) |
| `ScorePassthroughReranker` | No-op: sorts by existing fusion score | None | RRF score |

#### Query Transformer (`query_processor.py`)

- `QueryTransformer(llm_client, config)` — produces augmented query set.
- `transform(original_query)` → `[original_query, *rewrites, hyde_passage?]`.
- Rewriting: temperature=0.7, max_tokens=512, parses JSON array from LLM response (strips markdown fences).
- HyDE: temperature=0.3, max_tokens=256, generates 2-4 sentence plausible answer.
- Both tasks run in parallel via `asyncio.gather`; individual failures gracefully skipped.

#### RAG Engine (`engine.py`)

`RAGEngine` injection-based design: swap LLM, embedder, vector store, and reranker at construction time.

Pipeline stages:
1. **Query Transformation** → `QueryTransformer.transform()` parallel rewrite + HyDE.
2. **Hybrid Retrieval** → Embed all variants in one batched call → fan out N vector + N keyword searches concurrently → `asyncio.gather` with exception tolerance.
3. **RRF Fusion** → `reciprocal_rank_fusion(result_lists, k=60)` deduplicates by `chunk_id`.
4. **Cross-Encoder Rerank** → Top-20 → top-5 via configured reranker.
5. **Context Assembly** → Token budget check (0.75 words/token), `[Chunk-N]` annotation, `\n\n---\n\n` separators.
6. **Generation** → System prompt with strict `{context}` template → LLM call (temperature=0.1, max_tokens=1,024).

Graceful degradation: query transform failure → original query only. Rerank failure → top-k by fusion score. Context budget exceeded → warning log, truncate.

---

### `backend/agents/` — Agentic Orchestration

**`orchestrator.py`** implements a stateful ReAct agent using LangGraph:

- **State:** `AgentState(TypedDict)` with `messages`, `step_count`, `tool_calls_log`, `final_answer`, `error`.
- **Tools:** `ToolRegistry` with `register(definition, fn)` and `invoke(name, args)`.
- **Graph topology:** `START → reason → [end | act] → reason → ...` (compiled StateGraph).
- **Safety invariants:** `max_steps=10`, `timeout_seconds=120.0`, tool whitelisting, JSON-only action parsing, append-only trace.
- **Default tools:** `rag_search` (calls `RAGEngine.execute()`), `calculator` (safe `eval()` with math functions, no builtins).
- **LLM parsing:** System prompt with all tool definitions as JSON schemas. Agent outputs `{"tool": "...", "args": {...}}` or `{"final_answer": "..."}`.

---

## Database Schema (pgvector)

```sql
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;

CREATE TABLE IF NOT EXISTS document_chunks (
    chunk_id     TEXT          PRIMARY KEY,
    document_id  TEXT          NOT NULL,
    content      TEXT          NOT NULL,
    embedding    vector(1536)  NOT NULL,   -- configurable dimension
    metadata     JSONB         NOT NULL DEFAULT '{}',
    created_at   TIMESTAMPTZ   NOT NULL DEFAULT NOW(),
    updated_at   TIMESTAMPTZ   NOT NULL DEFAULT NOW()
);

-- Indexes
CREATE INDEX idx_dc_document_id    ON document_chunks (document_id);
CREATE INDEX idx_dc_embedding_hnsw ON document_chunks USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64);
CREATE INDEX idx_dc_content_fts    ON document_chunks USING GIN (to_tsvector('english', content));
CREATE INDEX idx_dc_metadata_gin   ON document_chunks USING GIN (metadata);
```

---

## Example Scripts

| File | What It Demonstrates | Requirements |
|------|---------------------|--------------|
| `01_basic_rag.py` | ChromaDB local ingest + chunking + OpenAI embeddings + full RAG query (3 queries: HNSW vs IVFFlat, HyDE, production metrics) | OpenAI API key |
| `02_hybrid_search.py` | pgvector side-by-side: vector-only vs keyword-only vs hybrid RRF. Cohere or HF reranker. 3 test queries showing overlap analysis | Docker postgres + OpenAI key |
| `03_agentic_pipeline.py` | ReAct agent: RAG tool + custom `index_cost_estimator` tool. Multi-step reasoning: agent decides when to invoke tools vs answer. Step trace printed | OpenAI + Anthropic keys |
| `run_advanced_rag.py` | Self-contained full pipeline: 6 mock enterprise docs, deterministic embeddings, mock LLM, 4 queries, formatted per-stage metrics | **None** (no API keys) |

---

## 14+ Year Lead AI Engineer Progression Map

### Production Pitfalls & Defenses

| # | Pitfall | Risk | Defense Built In This Repo |
|---|---------|------|---------------------------|
| 1 | **Prompt Injection** — user input smuggles directives into system prompts | High | `_SYSTEM_PROMPT` uses strict `{context}` template; user query never interpolated into system instructions |
| 2 | **Embedding Model Drift** — model update makes all indexed embeddings semantically stale | High | Metadata carries model version; ingestion pipeline separates model from data; re-embedding architecture documented |
| 3 | **Token Cost Explosion** — unbounded context windows burn API budget | High | Hard `max_context_tokens` budget (8K default); tiktoken integration; cross-encoder prunes to top-5 |
| 4 | **Connection Pool Exhaustion** — high-concurrency vector ingestion starves pool | Medium | asyncpg pool with configurable min/max/query limits; `statement_cache_size=100`; explicit pool lifecycle |
| 5 | **Lost in the Middle** — LLM ignores documents in middle of context window | High | Cross-encoder re-ranking prunes to exactly top-5; every chunk at edge of context; `[Chunk-N]` citation format |
| 6 | **N+1 Embedding Calls** — embedding each query variant serially | Medium | Single batched `EmbeddingRequest(texts=all_variants)`; all variants embedded concurrently in one API call |
| 7 | **Silent Retrieval Degradation** — HNSW recall drifts without monitoring | Medium | HNSW index with explicit `m`/`ef_construction`/`ef_search`; REINDEX scheduling documented in RAG_ARCHITECTURE.md |
| 8 | **Hard Filter Miss** — metadata filter applied post-retrieval instead of pre-search | Medium | JSONB `@>` containment pushes to GIN index BEFORE ANN distance computation; partial index support documented |
| 9 | **Score Scale Incompatibility** — vector cosine vs BM25 scores on different scales | Low | Reciprocal Rank Fusion (RRF) with k=60; score-free rank aggregation; no normalization needed |
| 10 | **Naked Error Swallowing** — `try/except: pass` silently drops failures | Critical | 22-class domain exception hierarchy; every exception carries typed `context` dict; structlog captures all errors |

### Skill Progression Ladder

```
Level 1: FOUNDATION (Read & Run)
  └─ Read docs/RAG_ARCHITECTURE.md, run all 4 examples, understand module boundaries

Level 2: BUILDER (Modify & Extend)
  └─ Add a new LLM provider client, implement a custom reranker, extend chunking strategy

Level 3: ARCHITECT (Design & Optimize)
  └─ Add streaming response pipeline, implement caching layer, design new index topology

Level 4: OPERATOR (Harden & Monitor)
  └─ Add Prometheus metrics, implement canary deployment, A/B test retrieval strategies,
     build re-embedding migration pipeline
```

### Key Files to Master (in order)

```
 1. docs/LEARNING_HANDBOOK.md        ← Understand the learning loop
 2. docs/FOLDER_STRUCTURE.md         ← Know where everything lives
 3. backend/core/exceptions.py       ← Internalize the error taxonomy
 4. backend/core/config.py           ← Configure every service
 5. backend/core/logging.py          ← Set up observability
 6. backend/llm_client/base.py       ← Provider contract pattern
 7. backend/llm_client/openai_client.py  ← Concrete implementation with retry
 8. backend/vector_store/base.py     ← Vector store contract
 9. backend/vector_store/pg_vector.py   ← Production vector backend
10. backend/rag_pipeline/query_processor.py  ← Query augmentation techniques
11. backend/rag_pipeline/engine.py   ← Full pipeline orchestration
12. backend/agents/orchestrator.py   ← Agentic workflow patterns
13. docs/RAG_ARCHITECTURE.md         ← Deep theory + production trade-offs
```

---

## Important Online References

| Resource | URL |
|----------|-----|
| LinkedIn Learning: Python Essential Training | https://www.linkedin.com/learning/python-essential-training-18764650 |
| Python Official Documentation | https://docs.python.org/3/ |
| FastAPI Documentation | https://fastapi.tiangolo.com/ |
| Pydantic Documentation | https://docs.pydantic.dev/ |
| pgvector Documentation | https://github.com/pgvector/pgvector |
| structlog Documentation | https://www.structlog.org/ |
| LangGraph Documentation | https://langchain-ai.github.io/langgraph/ |

---

## Canonical Structure

```text
ai-six-months/
├── .ai/system_instructions.md       ← Operator profile & engineering rules
├── .claude/context_profile.json     ← Structured AI context metadata
├── AGENTS.md / CLAUDE.md            ← Agent governance documents
├── README.md                        ← This file
├── pyproject.toml                   ← Poetry config + mypy/ruff/pytest settings
├── docker-compose.yml               ← Local infrastructure (pgvector, pgAdmin, Ollama)
├── .gitignore                       ← Excludes __pycache__, .venv, .env, python-learning/
├── backend/                         ← All production source code
│   ├── agents/                      ← ReAct loop, tool registry, orchestrator
│   ├── core/                        ← Config, domain exceptions, structured logging
│   ├── llm_client/                  ← OpenAI, Anthropic, DeepSeek, Ollama clients
│   ├── rag_pipeline/                ← Query transformation, hybrid retrieval, RRF, reranking
│   └── vector_store/                ← BaseVectorStore, ChromaDB, pgvector implementations
├── data/
│   ├── documents/sample.md          ← Source ingestion corpus (139-line enterprise KB)
│   └── vector_db/                   ← Local vector persistence (git-ignored)
├── docs/
│   ├── FOLDER_STRUCTURE.md          ← Authoritative placement guide
│   ├── SKILLS.md                    ← 5 skill tracks + mastery levels
│   ├── LEARNING_HANDBOOK.md         ← Step-by-step learning execution plan
│   ├── RAG_ARCHITECTURE.md          ← 394-line production RAG design reference
│   └── reference/README.md          ← External reference notes
├── examples/                        ← 4 runnable end-to-end demos
├── tests/                           ← Unit & integration test boundaries
├── python-learning/                 ← External learning material (git-ignored, local-only)
└── logs/                            ← Application log files (git-ignored)
```

---

## Governance & Rules

### Governance Priority (descending)
1. `.ai/system_instructions.md`
2. `.claude/context_profile.json`
3. `docs/FOLDER_STRUCTURE.md`
4. `docs/SKILLS.md`
5. `docs/LEARNING_HANDBOOK.md`
6. `README.md`

### Folder Rules
1. Keep `.ai/` and `.claude/` at repository root.
2. Keep all production code in `backend/` only.
3. Keep vector store implementations separated: `base.py`, `chroma_store.py`, `pg_vector.py`.
4. Keep query transformation logic in dedicated `backend/rag_pipeline/query_processor.py`.
5. Keep runtime artifacts in `data/vector_db/`.
6. Keep external learning material in top-level `python-learning/`.
7. Keep examples runnable from `examples/`.
8. Do not push local learning assets from `python-learning/` to git.

### Anti-Patterns
- No production source code in `docs/`, `data/`, or `examples/`.
- No ad-hoc top-level folders without an explicit reason.
- No hidden runtime output mixed with tracked source.
- No monolithic module files — separate implementations by concern.
- No commented-out code blocks; use version control for history.

---

## Testing & Quality

```bash
poetry run ruff check .            # Lint (E, W, F, I, N, UP, ANN, ASYNC, S, B, C4, PT)
poetry run mypy backend/           # Strict type check (Python 3.12, strict=true)
poetry run pytest                  # Run all tests with coverage (target: 80%)
```

**mypy:** strict mode, ignores missing imports (3rd-party SDKs), excludes `tests/`.
**ruff:** 100-char line length, ignores ANN101 (missing self type), ANN102 (missing cls type), S101 (assert).
**pytest:** asyncio auto mode, coverage reports with `--cov-fail-under=80`.

---

## Notes

- This repo targets Python 3.12 and async-first architecture.
- `python-learning/` contains LinkedIn Learning Jupyter notebooks and is git-ignored by policy.
- `docs/RAG_ARCHITECTURE.md` is the comprehensive production RAG design reference (chunking, HNSW vs IVFFlat, RRF, HyDE, security).
- The `DeepSeekClient` is implemented but not exported from `llm_client/__init__.py` — import directly from `backend.llm_client.deepseek_client`.
- `backend/__init__.py` has a stale import path (`backend.config` → should be `backend.core.config`) — kept for backward compatibility.
- `RAGConfig.hybrid_alpha` is defined but not currently wired into the engine's fusion logic.
