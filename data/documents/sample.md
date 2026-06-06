# Enterprise Knowledge Base — Sample Documents

## pgvector: Production Best Practices

pgvector is a PostgreSQL extension that enables storing and querying high-dimensional
vector embeddings directly inside your relational database. This eliminates the need
for a separate vector database tier for many use cases.

### HNSW vs IVFFlat Index Selection

**HNSW (Hierarchical Navigable Small World)** is the preferred index type for most
production workloads:
- No training phase required — builds incrementally as data is inserted.
- Sub-linear query time: O(log N) average for retrieval.
- Parameters: `m` (max connections per layer, default 16) and
  `ef_construction` (candidate list size during build, default 64).
- Higher `m` → better recall at cost of more memory.

**IVFFlat** is appropriate only when:
- Dataset size is > 10 million vectors.
- You can afford a training step (run `VACUUM ANALYZE` after bulk load).
- Query latency requirements allow for slightly lower recall.

### Connection Pool Configuration

Always configure asyncpg pools with explicit bounds:

```python
pool = await asyncpg.create_pool(
    dsn=dsn,
    min_size=5,           # pre-warm this many connections
    max_size=20,          # ceiling matches PostgreSQL max_connections / service count
    max_queries=50_000,   # recycle connections to prevent memory leaks
    max_inactive_connection_lifetime=300.0,
)
```

Setting `min_size` too low causes cold-start latency spikes under burst traffic.
Setting `max_size` too high starves other services of connections.

---

## Advanced RAG Patterns

### Hypothetical Document Embedding (HyDE)

HyDE addresses the query-document mismatch problem: user queries are short,
document passages are long. The cosine distance between them is often large
even when semantically relevant.

**Solution:** Use an LLM to generate a "hypothetical answer" — a plausible
passage that would answer the query. Embed this passage instead of (or in
addition to) the raw query. The hypothetical answer lives in the same
embedding space as actual document passages.

```
User query:    "What are the risks of using IVFFlat?"
Hypothetical:  "IVFFlat indexes require a training step on representative data.
               If the data distribution shifts, recall degrades until the index
               is rebuilt. This makes it unsuitable for datasets with high churn."
```

The hypothetical passage is then used as the query embedding for vector search.

### Reciprocal Rank Fusion (RRF)

When merging multiple ranked result lists (e.g., from several query variants
or from vector + keyword search), simple score averaging is fragile because
score scales differ across sources. RRF normalises this:

```
RRF_score(d) = Σ_i [ 1 / (k + rank_i(d)) ]   where k = 60 (smoothing constant)
```

A document ranked 1st in any list gets a boost of 1/61 ≈ 0.016.
A document ranked 20th gets 1/80 ≈ 0.012.
Documents absent from a list contribute 0.

The `k=60` constant was empirically determined to be robust across information
retrieval benchmarks. Tune it only if your retrieval task has unusual rank distributions.

### Lost-in-the-Middle Mitigation

Research shows LLMs perform better when relevant context appears at the beginning
or end of the prompt, not in the middle. Counter-measures:

1. **Limit context to 5 chunks maximum** — the re-ranking step should prune aggressively.
2. **Order by relevance score descending** — highest-confidence chunks first.
3. **Use chunk IDs in citations** — lets you verify which chunks actually contributed.

---

## Multi-Agent System Design

### ReAct (Reason + Act) Loop

The ReAct pattern interleaves reasoning steps with tool invocations:

```
Thought: I need to find the current HNSW ef_search parameter docs.
Action: rag_search({"query": "pgvector HNSW ef_search parameter"})
Observation: ef_search controls the search beam width at query time...
Thought: Now I have the information. I can answer directly.
Answer: {"final_answer": "..."}
```

Each step is serialised as a structured JSON object so it can be parsed
reliably without regex hacks.

### Agent Loop Safety Invariants

Every production agent must enforce:
- **max_steps**: hard ceiling; raise `AgentLoopDetectedError` to surface the issue.
- **wall-clock timeout**: wrap with `asyncio.wait_for` — LLM latency is unbounded.
- **tool whitelisting**: only registered tools can be invoked; unknown names raise immediately.
- **append-only trace**: every tool call is logged; essential for debugging loops.

---

## LLMOps: Production Monitoring Checklist

| Metric                     | Alert Threshold     | Tool              |
|----------------------------|---------------------|-------------------|
| P95 RAG latency            | > 5 seconds         | Prometheus/Grafana|
| Embedding API error rate   | > 1%                | Datadog           |
| Re-ranker null result rate | > 10%               | Custom gauge      |
| Token budget overruns      | Any occurrence      | structlog warning |
| Agent loop detections      | Any occurrence      | PagerDuty alert   |
| Vector index staleness     | > 24 hours old      | Scheduled job     |

### Prompt Drift Detection

Prompt drift occurs when the LLM's interpretation of the same prompt changes
over time due to model updates or fine-tuning on new data. Detect it by:

1. Maintaining a **golden eval set** of 50–100 query/expected-answer pairs.
2. Running the eval set on every model version change.
3. Flagging if F1 score drops > 5% from baseline.
4. Use `langsmith` or `promptfoo` for automated regression tracking.
