# RAG Architecture — Production-Grade Guide

## Overview

This document details the architectural decisions, trade-offs, and production patterns
that underpin the Advanced RAG pipeline in this repository. It serves as a reference
for AI engineers building retrieval-augmented generation systems at scale.

---

## 1. Chunking Strategies

Chunking is the single highest-leverage decision in a RAG pipeline. Poor chunks poison
retrieval regardless of how good your embedding model or vector index is.

### 1.1 Fixed-Size Recursive Chunking with Token Overlap

**How it works:** Split documents into `N`-token windows with a configurable overlap
between consecutive chunks. The overlap ensures that sentences spanning a chunk boundary
are represented fully in at least one chunk.

```
Document:  [A B C D E F G H I J]
Chunk size = 4, Overlap = 2
Chunk 1:   [A B C D]
Chunk 2:       [C D E F]
Chunk 3:           [E F G H]
Chunk 4:               [G H I J]
```

**When to use:**
- General-purpose indexing where document structure is unknown.
- Low-latency ingestion pipelines where speed matters more than precision.
- When using a token-aware splitter (e.g., `tiktoken` for OpenAI models).

**Production parameters:**
| Parameter | Recommended | Rationale |
|-----------|-----------|-----------|
| Chunk size | 256–512 tokens | Fits within most embedding model context windows (512 tokens for `text-embedding-3-small`) |
| Overlap | 10–20% of chunk size | Preserves semantic continuity without excessive duplication |
| Separators | `["\n\n", "\n", ". ", " "]` | Respects natural text boundaries (paragraph → sentence → word) |

**Pitfalls:**
- Fixed-size ignores document semantics: a chunk might split a concept mid-sentence.
- Excessive overlap bloats the index and creates near-duplicate retrieval results.
- Long documents with repetitive headers/footers waste embedding budget on boilerplate.

### 1.2 Semantic Chunking

**How it works:** Use an embedding model or a sentence-similarity scorer to detect
semantic boundaries. Adjacent sentences are compared; a chunk boundary is inserted
when the similarity drops below a threshold.

**When to use:**
- Domain-specific knowledge bases where semantic coherence matters (legal, medical, research).
- Long-form content with natural section breaks (books, whitepapers, academic papers).
- When retrieval precision (IR metrics) is more important than ingestion speed.

**Production parameters:**
| Parameter | Recommended | Rationale |
|-----------|-----------|-----------|
| Similarity threshold | 0.5–0.7 cosine | Lower = more aggressive splitting |
| Min chunk size | 100 tokens | Prevents overly granular fragments |
| Max chunk size | 1,000 tokens | Prevents runaway chunks on homogeneous text |
| Breakpoint percentile | 90th | Split at the N-th percentile of local minima |

**Pitfalls:**
- 4–10× slower than fixed-size chunking due to per-sentence embedding cost.
- Sensitive to embedding model quality — model drift changes chunk boundaries.
- Non-deterministic across embedding model versions.

### 1.3 Hierarchical / Parent-Child Chunking

A hybrid approach: store small "child" chunks for precise retrieval, but augment
each result with a larger "parent" chunk for LLM context.

**Production pattern:**
```text
Parent (1,024 tokens): "The complete section on ..."
├── Child 1 (256 tokens): "First paragraph..."
├── Child 2 (256 tokens): "Second paragraph..."
└── Child 3 (256 tokens): "Third paragraph..."
```

Retrieve child chunks → map to parent → send parent to LLM. This gives the LLM
the surrounding context that was lost during chunking.

---

## 2. Vector Indexing Topologies

### 2.1 HNSW (Hierarchical Navigable Small World)

**Algorithm:** Builds a multi-layer graph where each node connects to its nearest
neighbors. Search traverses from the top (coarse) layer to the bottom (fine) layer,
narrowing the candidate set at each step.

**pgvector configuration:**
```sql
CREATE INDEX ON document_chunks
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);
```

| Parameter | Meaning | Tuning Guidance |
|-----------|---------|-----------------|
| `m` | Max connections per node per layer | Higher = better recall, slower build. Range: 4–64. Default: 16 |
| `ef_construction` | Size of dynamic candidate list during build | Higher = better recall, slower build. Range: 32–400. Default: 64 |
| `ef_search` | Size of dynamic candidate list during search | Higher = better recall, slower search. Set per-query |

**When to use HNSW:**
- **Scale:** Up to ~10M vectors.
- **Latency budget:** < 10ms per query.
- **Recall target:** > 0.95 @ top-10.
- **Write pattern:** Batch-heavy (index builds are expensive; in-place updates are amortized).

**Production pitfalls:**
- HNSW index memory = raw vectors × 1.2–1.5× for graph edges. Budget accordingly.
- Index builds are CPU-intensive; offload to a replica during peak traffic.
- `ef_search` must be ≥ `top_k`; setting it too low silently degrades recall.
- Vacuum/REINDEX after large deletions; HNSW graphs don't auto-heal.

### 2.2 IVFFlat (Inverted File with Flat Compression)

**Algorithm:** Pre-clusters vectors using k-means. At query time, searches only
the `nprobe` nearest clusters instead of the full dataset.

**pgvector configuration:**
```sql
CREATE INDEX ON document_chunks
USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);
```

| Parameter | Meaning | Tuning Guidance |
|-----------|---------|-----------------|
| `lists` | Number of clusters (k-means centroids) | Rule of thumb: `lists = sqrt(N)` for N vectors. Range: 1–32,768 |
| `nprobe` | Number of clusters to search at query time | Higher = better recall, slower. Set per-query |

**When to use IVFFlat:**
- **Scale:** 1M–100M vectors (scales better than HNSW at extreme sizes).
- **Latency budget:** < 50ms per query.
- **Recall target:** > 0.90 @ top-10 (lower than HNSW).
- **Write pattern:** Steady insert stream (k-means centroid drift requires periodic `REINDEX`).

**Production pitfalls:**
- Requires a training step (random sample or full scan) before index creation.
- Recall degrades over time as new vectors shift the data distribution — schedule REINDEX.
- `lists` too high → each cluster is tiny, search probes many; `lists` too low → cluster scan is expensive.

### 2.3 Decision Matrix

| Requirement | HNSW | IVFFlat |
|-------------|------|---------|
| Dataset < 1M vectors | **Preferred** | Overhead not worth it |
| Dataset 1M–10M | **Preferred** | Viable |
| Dataset 10M–100M | Memory-constrained | **Preferred** |
| Recall > 0.95 | **Preferred** | Hard to achieve |
| High insert rate (>1K/s) | Build-time heavy | **Preferred** |
| Memory budget tight | 1.5× raw vectors | 1.0× raw vectors |

---

## 3. Metadata Filtering — Hard Filters Before Vector Search

### 3.1 Why Pre-Filtering Matters

A vector similarity search with `top_k=20` over 10M vectors computes 10M distance
calculations, sorts them, and returns 20. Adding a metadata constraint (e.g.,
`source=arXiv AND year >= 2024`) that eliminates 80% of candidates means only 2M
distance calculations — a 5× throughput improvement.

### 3.2 Implementation Pattern: JSONB Containment

```sql
-- Full table scan (bad):
SELECT * FROM document_chunks
ORDER BY embedding <=> $1
LIMIT 20;

-- Pre-filtered (good): GIN index on metadata
SELECT * FROM document_chunks
WHERE metadata @> '{"source": "arxiv", "year": 2024}'::jsonb
ORDER BY embedding <=> $1
LIMIT 20;
```

**pgvector execution plan:**
1. GIN index scan on `metadata` → narrows to candidate rows.
2. HNSW/IVFFlat index scan on `embedding` → ANN over candidates only.
3. Sort + limit → final top-K.

### 3.3 Filter Design Rules

1. **Always index filter columns.** A GIN index on the JSONB `metadata` column costs
   ~10% of the table size and enables sub-millisecond filter resolution.

2. **Push filters to the lowest layer.** If your data has a natural partitioning key
   (tenant, datasource, date range), apply it in the vector store query, not in
   post-processing. The vector store can eliminate 90%+ of candidates before distance
   computation.

3. **Avoid `OR` filters on high-cardinality columns.** `WHERE tenant='A' OR tenant='B'`
   is fine for 2–3 values; `WHERE tenant IN (1,2,3,...,100)` degrades to a seq scan.

4. **Use partial indexes for fixed filters:**
   ```sql
   CREATE INDEX ON document_chunks
   USING hnsw (embedding vector_cosine_ops)
   WHERE source = 'production';
   ```

5. **Composite metadata keys:** Flatten nested metadata into top-level keys for
   efficient JSONB containment:
   ```json
   // Inefficient (nested)
   {"source": {"name": "arxiv", "year": 2024}}
   // Efficient (flat)
   {"source_name": "arxiv", "source_year": 2024}
   ```

---

## 4. Hybrid Search & Reciprocal Rank Fusion (RRF)

### 4.1 Why Hybrid?

Vector search is strong on semantic similarity but weak on:
- Domain-specific terminology and acronyms.
- Rare entities not well-represented in embedding training data.
- Exact match requirements (IDs, codes, version strings).

Keyword search (BM25 / tsvector) is strong on exact matching but weak on:
- Paraphrased queries.
- Cross-lingual or synonym-heavy documents.
- Conceptual relevance beyond term overlap.

Hybrid search combines both; RRF merges their ranked results without requiring
score normalization across different score distributions.

### 4.2 RRF Formula

```
RRF_score(d) = Σ [ 1 / (k + rank_i(d)) ]
```

Where:
- `k = 60` (default smoothing constant, empirically validated).
- `rank_i(d)` = document `d`'s position in result list `i` (1-indexed).
- Summation over all result lists (vector batch 1, vector batch 2, ..., keyword).

**Why k=60?**
- Small `k` (e.g., 1): top ranks dominate; a document ranked #1 in one list crushes documents ranked #2 in all lists.
- Large `k` (e.g., 1,000): all ranks converge to similar scores; ranking signal is lost.
- `k=60` is the empirically determined sweet spot from Cormack et al. (SIGIR 2009).

---

## 5. Cross-Encoder Re-Ranking

### 5.1 The "Lost in the Middle" Problem

LLMs attend disproportionately to the beginning and end of the context window.
Documents placed in the middle of a long context are effectively invisible.

**Mitigation:** Re-rank to top-5. With only 5 chunks in the context, every chunk
is at a "privileged" position near the start or end of the context.

### 5.2 Bi-Encoder vs. Cross-Encoder

| Property | Bi-Encoder (Embedding Model) | Cross-Encoder |
|----------|------------------------------|---------------|
| Encoding | Query and document encoded separately | Query and document encoded jointly |
| Speed | ~10K docs/sec | ~10 docs/sec |
| Relevance accuracy | Moderate | High |
| Use case | Candidate retrieval (top-100 from 1M) | Re-ranking (top-5 from top-20) |

### 5.3 Re-Ranking Pipeline Pattern

```
1M documents
    ↓ Bi-encoder ANN (fast, approximate)
  top-20 candidates
    ↓ Cross-encoder (slow, precise)
  top-5 documents
    ↓ Context assembly
  LLM generation
```

This two-stage pattern gives you the speed of ANN for broad recall and the accuracy
of cross-attention for final precision.

---

## 6. Query Transformation Techniques

### 6.1 Query Rewriting

Generate `N` semantically diverse reformulations of the user query. Each variant
uses different vocabulary, granularity, and framing to maximize the probability
that at least one variant matches the embedding space of relevant documents.

**Example:**
```
Original: "How do I scale pgvector?"
Variant 1: "PostgreSQL pgvector horizontal scaling strategies"
Variant 2: "Performance tuning for large vector databases in Postgres"
Variant 3: "Best practices for production pgvector deployments at 10M+ vectors"
```

### 6.2 Hypothetical Document Embedding (HyDE)

Generate a plausible answer passage, embed it, and use THAT embedding for retrieval
instead of the query embedding. This bridges the gap between short user queries and
dense, informative document embeddings.

**Example:**
```
Query: "What causes HNSW index degradation?"
HyDE passage: "HNSW index recall degrades over time due to data distribution drift
  from continuous inserts. The graph structure becomes suboptimal as new vectors
  populate regions not well-covered by the original k-means centroids. Regular
  REINDEX operations restore recall to expected levels."
```

The HyDE passage's embedding is far closer to relevant technical documents than the
short query embedding.

---

## 7. Production Pipeline Metrics

### 7.1 Latency Budget (Target)

| Stage | P50 Latency | P99 Latency |
|-------|-------------|-------------|
| Query transformation (LLM) | 800ms | 2,000ms |
| Embedding generation | 100ms | 300ms |
| Vector retrieval (ANN) | 5ms | 20ms |
| Keyword retrieval (tsvector) | 10ms | 50ms |
| RRF fusion | <1ms | 2ms |
| Cross-encoder re-rank | 200ms | 800ms |
| Context assembly + generation | 1,500ms | 4,000ms |
| **Total** | **~2.6s** | **~7.2s** |

### 7.2 Key Metrics to Monitor

1. **Recall@K:** Fraction of relevant documents retrieved in top-K. Measure against
   a golden dataset of query→relevant_doc_id mappings.
2. **MRR (Mean Reciprocal Rank):** Inverse of the rank of the first relevant document.
   Directly measures "did the right document appear first?"
3. **Embedding drift:** Cosine distance between embeddings of the same document
   across model versions. Track to trigger re-indexing.
4. **Token utilization:** Percentage of context tokens actually cited by the LLM in
   its response. Low utilization → poor chunk selection or overly long chunks.
5. **Re-rank delta:** Score difference between top retrieved chunk and 5th re-ranked
   chunk. Measures how much the cross-encoder is reordering the bi-encoder's output.

---

## 8. Security Hardening

### 8.1 Prompt Injection Defense

- Never interpolate raw user input into system prompts. Use strict template
  substitution with clearly delineated `{context}` and `{query}` markers.
- Validate that user queries match expected patterns before transformation.
- Strip markdown code fences and JSON-like structures from user queries that
  could be interpreted as prompt continuation.

### 8.2 Embedding Model Drift

- Version-tag all embeddings in metadata: `"embedding_model": "text-embedding-3-large-v2"`.
- Run a background re-embedding job on a schedule or when the model version changes.
- Maintain a staging index for the new model; cut over atomically after validation.

### 8.3 Token Cost Optimization

- Enforce a hard `max_context_tokens` budget (default: 8,000).
- Count tokens with `tiktoken` for accurate accounting, not word-count heuristics.
- Truncate chunks intelligently: keep the first and last sentences of each chunk
  when budget forces truncation.

---

## References

- Cormack, G. V., Clarke, C. L., & Buettcher, S. (2009). "Reciprocal rank fusion
  outperforms condorcet and individual rank learning methods." SIGIR.
- Malkov, Y. A., & Yashunin, D. A. (2018). "Efficient and robust approximate nearest
  neighbor search using Hierarchical Navigable Small World graphs." TPAMI.
- Gao, L., et al. (2023). "Precise Zero-Shot Dense Retrieval without Relevance Labels."
  (HyDE paper).
- pgvector documentation: https://github.com/pgvector/pgvector
