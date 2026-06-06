# AI Master Curriculum — 3-Phase Programme

## Structure at a Glance

| Phase | Focus | Duration | Mastery Target |
|-------|-------|----------|----------------|
| **1 — Agentic Tooling (AT)** | LLM APIs, tool use, provider engineering | Weeks 1–4 | Builder |
| **2 — RAG** | Embeddings, retrieval, reranking, evaluation | Weeks 5–9 | Architect |
| **3 — Agents & LLMOps** | Orchestration, reliability, observability, advanced patterns | Weeks 10–16 | Operator |

---

## Learning Loop (every module)

```
Learn → Build → Run → Verify → Record
```

1. **Learn** — read concept brief in this doc or `python-learning/`
2. **Build** — implement or improve one capability in `backend/`
3. **Run** — validate through `examples/`
4. **Verify** — add/update tests in `tests/`
5. **Record** — capture insights in `docs/LEARNING_HANDBOOK.md`

Call the eval agent at the end of each week: ask it to quiz you on the week's material.

---

## Phase 1: Agentic Tooling (AT) — Weeks 1–4

**Goal:** Own the LLM-as-API layer end to end. Understand how every request flows
from your code to the model and back. Implement tool use, function calling, and
structured outputs from scratch.

### Week 1 — LLM API Fundamentals

**Concepts:**
- Token economics: tokens ≠ characters; context window = hard cap
- Temperature, top-p, top-k — what they actually change vs. common lore
- Synchronous vs streaming responses; async HTTP lifecycle
- The provider abstraction: why `base.py` exists and what it enforces

**Codebase targets:**
- `backend/llm_client/base.py` — read every line, understand the contract
- `backend/core/config.py` — how settings flow to clients at runtime
- `backend/core/logging.py` — structlog setup, JSON vs console renderer

**Exercise:**
1. Trace a complete request from `examples/01_basic_rag.py` through the LLM client layer.
   Map each function call and log line on paper.
2. Change the provider via env var (`PROVIDER=ollama`) without touching code. Confirm the factory routes correctly.
3. Write a unit test: mock the HTTP layer, assert the structured response is parsed correctly.

**Checkpoint question:** What enforces that all providers return the same response shape?

---

### Week 2 — Multi-Provider Engineering

**Concepts:**
- OpenAI, Anthropic, Ollama, DeepSeek API differences (auth, payload schema, streaming protocol)
- Retry strategies: exponential backoff + jitter vs. fixed interval
- Timeout budgets: connect timeout vs. read timeout vs. total budget
- Usage accounting: token counts, cost estimation, budget guards

**Codebase targets:**
- `backend/llm_client/openai_client.py`
- `backend/llm_client/anthropic_client.py`
- `backend/llm_client/deepseek_client.py`
- `backend/llm_client/ollama_client.py`

**Exercise:**
1. Add a new provider: implement a `gemini_client.py` following the base contract.
   Route it through `factory.py`. No change to callers.
2. Inject a 100% failure rate on one provider (monkeypatch). Confirm retries fire and the
   correct exception propagates with context payload.
3. Add token usage logging to the base class so every call emits a structured log line with
   model, provider, prompt_tokens, completion_tokens.

**Checkpoint question:** What is the difference between `httpx.TimeoutException` and a
provider-level rate-limit 429? How does your client distinguish them?

---

### Week 3 — Tool Use & Function Calling

**Concepts:**
- Tool schema: JSON Schema definition, required vs optional params
- The tool-call loop: LLM emits tool call → you execute → result injected into context
- Parallel tool calls vs sequential
- Structured outputs: Pydantic model ↔ JSON Schema round-trip

**Codebase targets:**
- `backend/agents/orchestrator.py` — locate where tool results are injected
- `examples/03_agentic_pipeline.py`

**Exercise:**
1. Define 3 tools (web_search, calculator, get_current_time) as Pydantic models with
   JSON Schema export. Wire them into the orchestrator.
2. Run the agentic pipeline with a multi-step question that requires 2 tool calls in sequence.
   Log the full conversation turn by turn.
3. Handle tool execution failure gracefully: if a tool raises, inject a typed error result
   rather than crashing the loop.

**Checkpoint question:** What prevents an infinite tool-call loop? How do you detect
the model is spinning and cut it off?

---

### Week 4 — Prompt Engineering at Scale

**Concepts:**
- System prompt anatomy: role, constraints, output format, examples
- Few-shot prompting: when it helps, when it hurts (context bloat)
- Chain-of-thought: scratchpad technique and its latency cost
- Context window management: sliding window, summary compression, eviction strategies

**Exercise:**
1. Build a `PromptTemplate` Pydantic model with: system_prompt, examples: list, output_schema.
   Render it to the provider's message format automatically.
2. Implement context window compression: when messages exceed 80% of the model's context,
   summarize old turns with a cheap model call before continuing.
3. Compare CoT vs direct answer on 10 reasoning questions. Measure accuracy and latency.

**Checkpoint question:** You have a 128k context window. How do you decide what to keep
vs evict when it fills up?

---

## Phase 2: RAG — Weeks 5–9

**Goal:** Own retrieval end to end. Understand every decision that changes what the model
sees — chunking, embedding, indexing, query transformation, fusion, reranking, generation.
Evaluate your own pipeline quantitatively.

### Week 5 — Embeddings & Vector Mathematics

**Concepts:**
- What embeddings encode and what they don't
- Cosine similarity vs dot product vs Euclidean — when each is correct
- MTEB benchmark: how to choose an embedding model (dimensions, task type, speed)
- Embedding model context windows (512 tokens for most; 8k for newer models)

**Codebase targets:**
- `backend/vector_store/base.py` — the `Document` and `SearchResult` models
- `backend/vector_store/chroma_store.py` — how embeddings are stored and queried

**Exercise:**
1. Embed the same 20 sentences with 2 different models. Compute pairwise cosine similarity.
   Visualize with a heatmap. Identify where models disagree.
2. Measure embedding throughput: how many chunks/sec can you embed before the API rate-limits?
   Build a batching + throttling wrapper.
3. Implement embedding caching: hash chunk content → store embedding → skip re-embedding.

**Checkpoint question:** A semantic search returns a result with 0.97 cosine similarity
but the content is wrong. What caused this and how do you fix retrieval?

---

### Week 6 — Vector Stores & Index Tuning

**Concepts:**
- ChromaDB: local persistence, collection config, metadata filtering
- pgvector: HNSW vs IVFFlat — construction cost, query latency, recall trade-off
- HNSW parameters: `m` (graph connections), `ef_construction` (build quality), `ef_search` (query recall)
- Metadata filtering: pre-filter vs post-filter, cardinality effects on recall

**Codebase targets:**
- `backend/vector_store/pg_vector.py` — HNSW index creation, asyncpg pool
- `backend/vector_store/chroma_store.py`

**Exercise:**
1. Benchmark `ef_search` at values 64, 128, 256, 512 on 10k vectors.
   Plot recall@10 vs latency. Find your acceptable trade-off point.
2. Add a metadata filter to the pgvector store: `filter_by_source(source_url: str)`.
   Confirm it uses the index (check `EXPLAIN ANALYZE`).
3. Run the same query against Chroma and pgvector. Compare results, latency, and memory usage.

**Checkpoint question:** When would you choose IVFFlat over HNSW in production?

---

### Week 7 — Chunking & Ingestion

**Concepts:**
- Fixed-size recursive chunking with token overlap (see `docs/RAG_ARCHITECTURE.md §1`)
- Semantic chunking: embedding similarity boundary detection
- Hierarchical chunking: parent-child — retrieve child, pass parent to LLM
- Metadata extraction: source, section, page, timestamp — each becomes a filter axis

**Codebase targets:**
- `backend/rag_pipeline/engine.py` — ingest path
- `data/documents/sample.md` — baseline corpus

**Exercise:**
1. Ingest `sample.md` with 3 strategies: fixed-256, fixed-512, semantic.
   For each, run 10 queries and score answer quality (1–5 manually).
2. Implement hierarchical chunking: split into paragraphs (children) and sections (parents).
   Retrieve by child, return parent text to LLM.
3. Add `source`, `section`, `chunk_index` metadata to every chunk. Filter by section in a query.

**Checkpoint question:** Why does excessive chunk overlap hurt retrieval even though
it preserves semantic continuity?

---

### Week 8 — Query Processing & Hybrid Retrieval

**Concepts:**
- Query rewriting: LLM improves poorly-phrased queries
- HyDE (Hypothetical Document Embedding): generate a fake answer, embed it, search
- Multi-query: generate N query variants, union results, deduplicate
- BM25 keyword search: TF-IDF math, why it catches exact terms dense search misses
- Reciprocal Rank Fusion (RRF): how it merges ranked lists from heterogeneous retrievers

**Codebase targets:**
- `backend/rag_pipeline/query_processor.py` — QueryTransformer
- `backend/rag_pipeline/engine.py` — hybrid retrieval + RRF section

**Exercise:**
1. Run the same ambiguous query with and without HyDE. Compare top-5 results.
2. Implement multi-query: 3 variants via LLM, retrieve 5 per variant, RRF-fuse, return top-10.
3. Add BM25 as a second retrieval arm (use `rank_bm25`). Fuse with dense results via RRF.
   Compare vs dense-only on 20 queries.

**Checkpoint question:** HyDE improves recall but can hallucinate the hypothesis.
What happens to retrieval quality when the hypothesis is wrong?

---

### Week 9 — Reranking & RAG Evaluation

**Concepts:**
- Cross-encoder reranking: why it's more accurate than bi-encoder similarity
- Cohere Rerank API vs HuggingFace cross-encoder — latency, accuracy, cost
- RAGAS metrics: faithfulness, answer relevance, context precision, context recall
- LLM-as-judge: strengths, biases, calibration against human labels

**Codebase targets:**
- `backend/rag_pipeline/engine.py` — reranker section (Cohere, HuggingFace, Passthrough)

**Exercise:**
1. Run the full RAG pipeline on 20 question-answer pairs. Score with RAGAS.
   Baseline: no reranker. Compare vs Cohere reranker. Quantify the delta.
2. Implement LLM-as-judge: for each answer, ask Claude to score faithfulness 1–5.
   Correlate with RAGAS faithfulness score.
3. Build an evaluation harness: `tests/integration/test_rag_quality.py`. Run on every
   code change to detect regression.

**Checkpoint question:** Context precision is 0.4 but faithfulness is 0.9.
What does this tell you about the pipeline?

---

## Phase 3: Agents & LLMOps — Weeks 10–16

**Goal:** Build production-grade multi-agent systems and operate them reliably.
Know how to detect quality degradation, handle failures, and run advanced retrieval patterns.

### Week 10 — Agent Patterns

**Concepts:**
- ReAct loop: Reason → Act → Observe → Reason cycle
- Plan-and-Execute: upfront planning agent + execution agents
- Tool selection: how the model decides which tool to call, prompt engineering for tool choice
- Stopping conditions: max_steps, goal detection, confidence threshold

**Codebase targets:**
- `backend/agents/orchestrator.py`
- `examples/03_agentic_pipeline.py`

**Exercise:**
1. Trace the full ReAct loop in `orchestrator.py`. Draw the state machine.
2. Implement a `max_steps` guard with a graceful degradation response (not an error).
3. Add a Plan-and-Execute variant: a planner LLM call produces a JSON plan,
   executor runs each step in order, reporter synthesizes the final answer.

---

### Week 11 — Multi-Agent Orchestration

**Concepts:**
- LangGraph: nodes (agents/tools), edges (conditional routing), state (shared dict)
- Supervisor pattern: one orchestrator routes to specialist agents
- Peer-to-peer handoff: agents pass tasks directly between themselves
- Shared memory vs isolated state: when each is correct

**Exercise:**
1. Build a 2-agent system: `ResearchAgent` (retrieves from RAG) + `WriterAgent` (synthesizes).
   Supervisor routes query → researcher → writer → user.
2. Add a critic loop: `CriticAgent` scores the writer's output; if below threshold, sends back
   for revision (max 2 revision cycles).
3. Benchmark: how does multi-agent latency compare to single-agent on the same task?

---

### Week 12 — Production Reliability

**Concepts:**
- Exponential backoff with jitter (full jitter formula)
- Circuit breaker: closed → open → half-open state machine
- Fallback chain: primary provider → secondary → cached response → error
- Rate limiting: token bucket algorithm, per-minute and per-day budget guards

**Exercise:**
1. Implement a circuit breaker around the LLM client. Test: inject 5 consecutive failures,
   confirm the circuit opens and stops forwarding calls.
2. Build a provider fallback chain: OpenAI → DeepSeek → Ollama. Verify automatic failover.
3. Add a budget guard: `max_tokens_per_minute` enforced with a leaky bucket. Confirm it
   queues and releases at the correct rate.

---

### Week 13 — Observability & LLMOps

**Concepts:**
- Distributed tracing: trace spans, parent-child relationships, baggage propagation
- LLM-specific metrics: latency p50/p95/p99, token throughput, cost/request
- Embedding drift detection: monitor cosine similarity distribution of new vs baseline queries
- Output quality drift: LLM-as-judge score over rolling window; alert on degradation

**Exercise:**
1. Add OpenTelemetry spans to the RAG pipeline: one span per stage
   (query_transform, retrieve, rerank, generate). Export to Jaeger locally.
2. Build a drift detector: every 100 queries, compare embedding distribution to baseline
   using MMD or a simple percentile shift. Log alert when drift exceeds threshold.
3. Wire Prometheus metrics: `rag_query_latency_seconds`, `rag_tokens_total`,
   `rag_quality_score`. Build a Grafana dashboard.

---

### Week 14 — Agentic RAG & Advanced Patterns

**Concepts:**
- Agentic RAG: agent decides whether to retrieve, how many times, with what query
- Self-RAG: model generates retrieval tokens inline; post-process to decide retrieve vs skip
- CRAG (Corrective RAG): evaluate retrieved docs; if low quality, fall back to web search
- GraphRAG: entities + relationships as a graph; multi-hop reasoning across nodes

**Exercise:**
1. Implement CRAG: after retrieval, score document relevance with a cross-encoder;
   if max score < 0.5, fall back to a web search tool.
2. Build a simple GraphRAG: extract entities from documents with NER, store relationships,
   run multi-hop traversal for queries that span multiple entities.
3. Compare CRAG vs standard RAG on 20 adversarial queries (queries whose answer requires
   info not in the corpus). Measure answer quality improvement.

---

### Week 15–16 — Capstone

**Build a production-ready AI system that demonstrates all 3 phases:**

Requirements:
- Provider-agnostic LLM layer (Phase 1)
- Hybrid RAG with HyDE, RRF fusion, cross-encoder reranking (Phase 2)
- Multi-agent orchestration with supervisor + specialist agents (Phase 3)
- RAGAS evaluation harness with quality baseline
- Structured logging + tracing + cost accounting
- Circuit breaker + fallback chain
- Full test suite (unit + integration)

Deliverables:
- `examples/04_capstone.py` — end-to-end demo
- `tests/integration/test_capstone.py` — full pipeline correctness
- Entry in `docs/LEARNING_HANDBOOK.md` — what you built and what you learned

---

## Mastery Progression

| Level | Signal |
|-------|--------|
| **Foundation** | Can run all examples, explain each module's role |
| **Builder** | Can modify modules safely; all tests pass |
| **Architect** | Can design cross-layer improvements and articulate trade-offs |
| **Operator** | Can detect failures, evaluate quality, run in production |

---

## Teaching Agents

| Agent file | When to invoke |
|------------|----------------|
| `.ai/agents/phase1_at_tutor.md` | Phase 1 sessions — LLM APIs, tool use |
| `.ai/agents/phase2_rag_tutor.md` | Phase 2 sessions — embeddings, retrieval |
| `.ai/agents/phase3_ops_tutor.md` | Phase 3 sessions — agents, LLMOps |
| `.ai/agents/eval_agent.md` | End of any week — quiz, checkpoint, grade progress |

Start each session with: *"I'm on Week N. Use the [phase] tutor."*
