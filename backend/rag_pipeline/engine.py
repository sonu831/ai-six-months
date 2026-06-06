"""
Advanced RAG execution engine.

Pipeline:
    1. Query Transformation  — LLM rewrites + Hypothetical Document Embedding (HyDE)
    2. Hybrid Retrieval      — concurrent vector ANN + full-text search across all variants
    3. RRF Fusion            — Reciprocal Rank Fusion merges and deduplicates candidates
    4. Cross-Encoder Rerank  — Cohere API or local HuggingFace model prunes to top-K
    5. Generation            — Context assembly + LLM call

Concrete reranker classes are defined here so callers import a single module.
Swap reranker at injection time; the engine is decoupled from the choice.
"""

from __future__ import annotations

import asyncio
import json
from abc import ABC, abstractmethod
from typing import Any

import structlog
from pydantic import BaseModel, Field

from backend.core.exceptions import (
    ContextAssemblyError,
    QueryTransformationError,
    RAGPipelineError,
    ReRankingError,
    RetrievalError,
)
from backend.llm_client.base import (
    BaseLLMClient,
    ChatCompletionRequest,
    ChatMessage,
    EmbeddingRequest,
)
from backend.vector_store.client import BaseVectorStore, SearchResult

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------


class RAGConfig(BaseModel):
    retrieval_top_k: int = 20           # candidates before re-ranking
    rerank_top_k: int = 5               # final chunks sent to LLM (max 5)
    hybrid_alpha: float = 0.7           # 1.0 = pure vector, 0.0 = pure keyword
    max_context_tokens: int = 8_000     # hard budget for assembled context
    query_transform_enabled: bool = True
    transform_variants: int = 3          # how many LLM-rewritten variants
    enable_hyde: bool = True            # Hypothetical Document Embedding


class RAGRequest(BaseModel):
    query: str
    document_filter: dict[str, Any] | None = None
    config_override: RAGConfig | None = None
    conversation_history: list[dict[str, str]] = Field(default_factory=list)


class RAGResponse(BaseModel):
    answer: str
    source_chunks: list[SearchResult]
    query_variants: list[str]
    retrieval_candidate_count: int
    final_context_count: int
    model_used: str


# ---------------------------------------------------------------------------
# Reranker hierarchy
# ---------------------------------------------------------------------------


class BaseReranker(ABC):
    """Interface contract for all cross-encoder re-ranking strategies."""

    @abstractmethod
    async def rerank(
        self,
        query: str,
        candidates: list[SearchResult],
        top_k: int,
    ) -> list[SearchResult]:
        """Return ``top_k`` SearchResults sorted by descending relevance."""


class CohereReranker(BaseReranker):
    """
    Cloud re-ranking via the Cohere Rerank v3 API.

    Requires ``cohere`` package:  pip install cohere
    """

    def __init__(self, api_key: str, model: str = "rerank-english-v3.0") -> None:
        import cohere  # type: ignore[import]

        self._client = cohere.AsyncClientV2(api_key=api_key)
        self._model = model
        self._log = structlog.get_logger(self.__class__.__name__)

    async def rerank(
        self,
        query: str,
        candidates: list[SearchResult],
        top_k: int,
    ) -> list[SearchResult]:
        if not candidates:
            return []
        self._log.debug("cohere.rerank.start", count=len(candidates), model=self._model)
        try:
            response = await self._client.rerank(
                query=query,
                documents=[c.content for c in candidates],
                model=self._model,
                top_n=min(top_k, len(candidates)),
                return_documents=False,
            )
        except Exception as exc:
            raise ReRankingError(
                "Cohere re-ranking API call failed",
                {"model": self._model, "candidates": len(candidates), "error": str(exc)},
            ) from exc

        reranked = [
            SearchResult(
                **{**candidates[r.index].model_dump(), "score": r.relevance_score}
            )
            for r in response.results
        ]
        self._log.info(
            "cohere.rerank.done",
            input_count=len(candidates),
            output_count=len(reranked),
        )
        return reranked


class HuggingFaceCrossEncoderReranker(BaseReranker):
    """
    Local cross-encoder using sentence-transformers.

    Runs in a thread pool via asyncio.to_thread so the event loop stays clean.
    Requires: pip install sentence-transformers
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> None:
        self._model_name = model_name
        self._encoder: Any = None
        self._log = structlog.get_logger(self.__class__.__name__)

    async def _ensure_loaded(self) -> None:
        if self._encoder is not None:
            return
        try:
            from sentence_transformers import CrossEncoder  # type: ignore[import]

            self._encoder = await asyncio.to_thread(
                CrossEncoder, self._model_name, max_length=512
            )
            self._log.info("cross_encoder.loaded", model=self._model_name)
        except ImportError as exc:
            raise ReRankingError(
                "sentence-transformers not installed. Run: pip install sentence-transformers",
                {"model": self._model_name},
            ) from exc

    async def rerank(
        self,
        query: str,
        candidates: list[SearchResult],
        top_k: int,
    ) -> list[SearchResult]:
        if not candidates:
            return []
        await self._ensure_loaded()

        pairs = [[query, c.content] for c in candidates]
        try:
            raw_scores: list[float] = await asyncio.to_thread(
                self._encoder.predict, pairs, convert_to_numpy=False
            )
        except Exception as exc:
            raise ReRankingError(
                "Cross-encoder inference failed",
                {"model": self._model_name, "pairs": len(pairs), "error": str(exc)},
            ) from exc

        scored = sorted(
            zip(candidates, raw_scores),
            key=lambda t: t[1],
            reverse=True,
        )
        result = [
            SearchResult(**{**chunk.model_dump(), "score": float(score)})
            for chunk, score in scored[:top_k]
        ]
        self._log.debug(
            "cross_encoder.rerank.done",
            input_count=len(candidates),
            output_count=len(result),
        )
        return result


class ScorePassthroughReranker(BaseReranker):
    """
    No-op re-ranker: returns top-k by the existing fusion score.
    Useful in unit tests and when no re-ranking service is configured.
    """

    async def rerank(
        self,
        query: str,
        candidates: list[SearchResult],
        top_k: int,
    ) -> list[SearchResult]:
        return sorted(candidates, key=lambda c: c.score, reverse=True)[:top_k]


# ---------------------------------------------------------------------------
# Query transformation
# ---------------------------------------------------------------------------

_REWRITE_SYSTEM = (
    "You are an expert information-retrieval specialist. "
    "Given a user question, generate exactly {n} semantically diverse "
    "reformulations that maximise recall in a hybrid search system. "
    "Rephrase for different vocabulary, granularity, and framing. "
    "Return ONLY a valid JSON array of strings — no prose, no markdown, no commentary."
)

_HYDE_SYSTEM = (
    "You are a domain expert. Write a concise, factually plausible passage "
    "(2–4 sentences) that would DIRECTLY ANSWER the following question. "
    "Return only the passage text — no preamble, no citations."
)


class QueryTransformer:
    """Produces an augmented set of queries from a single user question."""

    def __init__(self, llm_client: BaseLLMClient, config: RAGConfig) -> None:
        self._llm = llm_client
        self._config = config
        self._log = structlog.get_logger(self.__class__.__name__)

    async def transform(self, original_query: str) -> list[str]:
        """
        Returns [original_query, *llm_rewrites, hyde_passage?].

        Failures in individual sub-tasks are logged as warnings and
        gracefully skipped so the pipeline always has at least the
        original query to work with.
        """
        if not self._config.query_transform_enabled:
            return [original_query]

        tasks: list[Any] = [self._rewrite(original_query)]
        if self._config.enable_hyde:
            tasks.append(self._generate_hyde(original_query))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        variants: list[str] = [original_query]

        rewrites_result = results[0]
        if isinstance(rewrites_result, Exception):
            self._log.warning(
                "query_transform.rewrite.failed",
                error=str(rewrites_result),
                query=original_query[:80],
            )
        else:
            variants.extend(rewrites_result)

        if self._config.enable_hyde and len(results) > 1:
            hyde_result = results[1]
            if isinstance(hyde_result, Exception):
                self._log.warning(
                    "query_transform.hyde.failed",
                    error=str(hyde_result),
                )
            else:
                variants.append(hyde_result)

        self._log.debug(
            "query_transform.complete",
            original=original_query[:80],
            total_variants=len(variants),
        )
        return variants

    async def _rewrite(self, query: str) -> list[str]:
        system = _REWRITE_SYSTEM.format(n=self._config.transform_variants)
        request = ChatCompletionRequest(
            messages=[
                ChatMessage(role="system", content=system),
                ChatMessage(role="user", content=query),
            ],
            temperature=0.7,
            max_tokens=512,
        )
        try:
            response = await self._llm.chat_complete(request)
            raw = response.content.strip()
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
            parsed: list[str] = json.loads(raw)
            if not isinstance(parsed, list):
                raise ValueError(f"Expected JSON array, got: {type(parsed)}")
            return [s for s in parsed if isinstance(s, str)]
        except Exception as exc:
            raise QueryTransformationError(
                "LLM query rewriting failed",
                {"query": query[:80], "error": str(exc)},
            ) from exc

    async def _generate_hyde(self, query: str) -> str:
        request = ChatCompletionRequest(
            messages=[
                ChatMessage(role="system", content=_HYDE_SYSTEM),
                ChatMessage(role="user", content=query),
            ],
            temperature=0.3,
            max_tokens=256,
        )
        try:
            response = await self._llm.chat_complete(request)
            return response.content.strip()
        except Exception as exc:
            raise QueryTransformationError(
                "HyDE generation failed",
                {"query": query[:80], "error": str(exc)},
            ) from exc


# ---------------------------------------------------------------------------
# Reciprocal Rank Fusion
# ---------------------------------------------------------------------------


def reciprocal_rank_fusion(
    result_lists: list[list[SearchResult]],
    k: int = 60,
) -> list[SearchResult]:
    """
    Merge multiple ranked result lists into one via RRF.

    RRF score for document d:
        score(d) = Σ_i [ 1 / (k + rank_i(d)) ]

    Deduplicates by chunk_id; ties broken arbitrarily.
    """
    rrf_scores: dict[str, float] = {}
    best_chunks: dict[str, SearchResult] = {}

    for ranked_list in result_lists:
        for rank, result in enumerate(ranked_list, start=1):
            cid = result.chunk_id
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (k + rank)
            if cid not in best_chunks:
                best_chunks[cid] = result

    return sorted(
        [
            SearchResult(**{**best_chunks[cid].model_dump(), "score": rrf_scores[cid]})
            for cid in best_chunks
        ],
        key=lambda r: r.score,
        reverse=True,
    )


# ---------------------------------------------------------------------------
# RAG Engine
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a precise enterprise knowledge assistant.
Answer the user's question using ONLY the provided context passages.
If the context does not contain enough information to answer confidently, say so explicitly.

Formatting rules:
- Cite inline as [Chunk-N] when referencing a specific passage.
- Use markdown bullet lists for enumerated answers.
- Keep the answer focused; do not pad with filler.
- Do not speculate beyond what the context states.

Context:
{context}
"""


class RAGEngine:
    """
    Orchestrates the full advanced RAG pipeline.

    Injection-based design: swap the LLM client, embedder, vector store,
    and reranker at construction time without modifying this class.

    Usage:
        engine = RAGEngine(
            llm_client=AnthropicClient.create(),
            embedder=OpenAIClient.create(),
            vector_store=await PgVectorStore.create(embedding_dim=1536),
            reranker=CohereReranker(api_key="..."),
        )
        response = await engine.execute(RAGRequest(query="What is pgvector?"))
        await engine.close()
    """

    def __init__(
        self,
        llm_client: BaseLLMClient,
        embedder: BaseLLMClient,
        vector_store: BaseVectorStore,
        reranker: BaseReranker,
        config: RAGConfig | None = None,
    ) -> None:
        self._llm = llm_client
        self._embedder = embedder
        self._store = vector_store
        self._reranker = reranker
        self._config = config or RAGConfig()
        self._transformer = QueryTransformer(llm_client=llm_client, config=self._config)
        self._log = structlog.get_logger(self.__class__.__name__)

    async def execute(self, request: RAGRequest) -> RAGResponse:
        """Run the full pipeline and return a grounded answer with source attribution."""
        cfg = request.config_override or self._config
        log = self._log.bind(query=request.query[:80])
        log.info("rag.execute.start")

        # ── Step 1: Query transformation ──────────────────────────────────
        try:
            query_variants = await self._transformer.transform(request.query)
        except QueryTransformationError:
            log.warning("rag.transform.failed_graceful", fallback="original_only")
            query_variants = [request.query]

        # ── Step 2: Hybrid retrieval across all variants ───────────────────
        log.info("rag.retrieval.start", variant_count=len(query_variants))
        try:
            candidates = await self._hybrid_retrieve(
                query_variants=query_variants,
                top_k=cfg.retrieval_top_k,
                document_filter=request.document_filter,
            )
        except Exception as exc:
            raise RetrievalError(
                "Hybrid retrieval produced no usable candidates",
                {"query": request.query[:80], "error": str(exc)},
            ) from exc
        log.info("rag.retrieval.done", candidates=len(candidates))

        # ── Step 3: Re-ranking ────────────────────────────────────────────
        try:
            reranked = await self._reranker.rerank(
                query=request.query,
                candidates=candidates,
                top_k=cfg.rerank_top_k,
            )
        except ReRankingError:
            log.warning("rag.rerank.failed_graceful", fallback="top_k_by_score")
            reranked = candidates[: cfg.rerank_top_k]
        log.info("rag.rerank.done", final_chunks=len(reranked))

        # ── Step 4: Context assembly + generation ─────────────────────────
        answer, model_used = await self._generate(
            query=request.query,
            context_chunks=reranked,
            conversation_history=request.conversation_history,
            max_context_tokens=cfg.max_context_tokens,
        )
        log.info("rag.execute.complete")

        return RAGResponse(
            answer=answer,
            source_chunks=reranked,
            query_variants=query_variants,
            retrieval_candidate_count=len(candidates),
            final_context_count=len(reranked),
            model_used=model_used,
        )

    # ------------------------------------------------------------------
    # Hybrid retrieval
    # ------------------------------------------------------------------

    async def _hybrid_retrieve(
        self,
        query_variants: list[str],
        top_k: int,
        document_filter: dict[str, Any] | None,
    ) -> list[SearchResult]:
        """
        Fan out all variant queries across both vector and keyword search,
        run all tasks concurrently, then fuse with RRF.
        """
        # Embed all variants in one batched API call
        embed_response = await self._embedder.embed(
            EmbeddingRequest(texts=query_variants)
        )
        embeddings = embed_response.embeddings

        tasks: list[Any] = []
        for emb in embeddings:
            tasks.append(
                self._store.similarity_search(
                    query_embedding=emb,
                    top_k=top_k,
                    filter=document_filter,
                )
            )
        for variant in query_variants:
            tasks.append(
                self._store.keyword_search(
                    query=variant,
                    top_k=top_k,
                    filter=document_filter,
                )
            )

        raw_results: list[Any] = await asyncio.gather(*tasks, return_exceptions=True)

        valid_lists: list[list[SearchResult]] = []
        for i, result in enumerate(raw_results):
            if isinstance(result, Exception):
                self._log.warning(
                    "rag.search_task.failed",
                    task_index=i,
                    error=str(result),
                )
            else:
                valid_lists.append(result)

        if not valid_lists:
            raise RetrievalError(
                "All search tasks failed — zero candidates available",
                {"variant_count": len(query_variants)},
            )

        return reciprocal_rank_fusion(valid_lists)

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    async def _generate(
        self,
        query: str,
        context_chunks: list[SearchResult],
        conversation_history: list[dict[str, str]],
        max_context_tokens: int,
    ) -> tuple[str, str]:
        context_str = self._assemble_context(context_chunks, max_context_tokens)
        system_prompt = _SYSTEM_PROMPT.format(context=context_str)

        messages: list[ChatMessage] = [
            ChatMessage(role="system", content=system_prompt)
        ]
        for turn in conversation_history:
            role = turn.get("role", "user")
            if role in ("user", "assistant"):
                messages.append(ChatMessage(role=role, content=turn["content"]))  # type: ignore[arg-type]
        messages.append(ChatMessage(role="user", content=query))

        request = ChatCompletionRequest(
            messages=messages,
            temperature=0.1,
            max_tokens=1024,
        )
        try:
            response = await self._llm.chat_complete(request)
            return response.content, response.model
        except Exception as exc:
            raise RAGPipelineError(
                "LLM generation step failed",
                {"query": query[:80], "error": str(exc)},
            ) from exc

    @staticmethod
    def _assemble_context(chunks: list[SearchResult], max_tokens: int) -> str:
        """
        Build a context string within the token budget.

        Token counting uses the 0.75 words-per-token approximation to avoid
        a tokenizer dependency in the hot path. Accurate token budgeting can
        be layered in by replacing the estimator with tiktoken.
        """
        token_budget = max_tokens
        parts: list[str] = []
        for i, chunk in enumerate(chunks, start=1):
            passage = (
                f"[Chunk-{i}] document_id={chunk.document_id!r} "
                f"score={chunk.score:.3f}\n{chunk.content}"
            )
            estimated_tokens = int(len(passage.split()) / 0.75)
            if token_budget - estimated_tokens < 0:
                logger.warning(
                    "rag.context.token_budget_exceeded",
                    included_chunks=i - 1,
                    total_chunks=len(chunks),
                )
                break
            token_budget -= estimated_tokens
            parts.append(passage)

        if not parts:
            raise ContextAssemblyError(
                "No chunks fit within the token budget",
                {"budget": max_tokens, "first_chunk_size": len(chunks[0].content.split()) if chunks else 0},
            )

        return "\n\n---\n\n".join(parts)

    async def close(self) -> None:
        await self._store.close()
        await self._llm.close()
        await self._embedder.close()
        self._log.info("rag_engine.closed")


Embedding: TypeAlias = list[float]


@dataclass
class RetrievedDocument:
    """A single document retrieved from any search source."""

    document: str
    metadata: dict = field(default_factory=dict)
    vector_score: float = 0.0
    keyword_score: float = 0.0
    hybrid_score: float = 0.0
    rerank_score: float = 0.0
    source: str = "vector"


@dataclass
class PipelineMetrics:
    """Latency and performance metrics for each RAG pipeline stage."""

    query_transformation_ms: float = 0.0
    embedding_generation_ms: float = 0.0
    vector_retrieval_ms: float = 0.0
    keyword_retrieval_ms: float = 0.0
    hybrid_fusion_ms: float = 0.0
    reranking_ms: float = 0.0
    total_ms: float = 0.0
    total_documents_retrieved: int = 0
    documents_after_pruning: int = 0


@dataclass
class RAGResult:
    """Complete RAG pipeline result."""

    query: str
    rewritten_queries: list[str] = field(default_factory=list)
    documents: list[RetrievedDocument] = field(default_factory=list)
    metrics: PipelineMetrics = field(default_factory=PipelineMetrics)
    correlation_id: str = field(default_factory=lambda: str(uuid.uuid4()))


class AdvancedRAGEngine:
    """End-to-end RAG engine with hybrid search and cross-encoder re-ranking.

    Pipeline stages:
      1. QueryProcessor transforms the query (rewriting + HyDE)
      2. Embedding service computes embeddings for all query variants
      3. Parallel vector retrieval across all variants
      4. Keyword/BM25 lexical retrieval (in-memory for sandbox)
      5. Hybrid fusion via Reciprocal Rank Fusion (RRF)
      6. Cross-encoder re-ranking to top-5 most relevant documents

    All stages are instrumented with detailed latency metrics for
    production observability and continuous optimization.
    """

    def __init__(
        self,
        vector_store: BaseVectorStore,
        query_processor: Optional[QueryProcessor] = None,
        hybrid_settings: Optional[HybridSearchSettings] = None,
        reranker_settings: Optional[RerankerSettings] = None,
    ) -> None:
        self._vector_store = vector_store
        self._query_processor = query_processor or QueryProcessor()
        self._hybrid_settings = hybrid_settings or get_settings().hybrid_search
        self._reranker_settings = reranker_settings or get_settings().reranker
        self._embedding_function = self._mock_embedding_function

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def query(self, raw_query: str) -> RAGResult:
        """Execute the full RAG pipeline for a raw user query.

        Args:
            raw_query: The user's natural language query.

        Returns:
            RAGResult containing re-ranked documents and per-stage metrics.
        """
        correlation_id = str(uuid.uuid4())
        total_start = time.perf_counter()
        metrics = PipelineMetrics()

        logger.info(
            "RAG pipeline started.",
            extra={
                "context": {
                    "correlation_id": correlation_id,
                    "query": raw_query[:200],
                    "query_length": len(raw_query),
                }
            },
        )

        # ── Stage 1: Query Transformation ──────────────────────────
        stage_start = time.perf_counter()
        try:
            transformation = await self._query_processor.transform_query(
                raw_query, include_hyde=True
            )
            metrics.query_transformation_ms = (
                time.perf_counter() - stage_start
            ) * 1000
        except QueryTransformationException:
            raise
        except Exception as exc:
            logger.error(
                "Query transformation failed.",
                extra={
                    "context": {
                        "correlation_id": correlation_id,
                        "error_type": type(exc).__name__,
                    }
                },
                exc_info=True,
            )
            raise RAGEngineException(
                f"Query transformation failed: {raw_query[:100]}...", correlation_id
            ) from exc

        # ── Stage 2: Embedding Generation ──────────────────────────
        stage_start = time.perf_counter()

        all_queries_to_embed: list[str] = list(transformation.rewritten_queries)
        if transformation.hypothetical_document:
            all_queries_to_embed.append(transformation.hypothetical_document)

        query_embeddings_map: dict[str, Embedding] = {}
        for query_text in all_queries_to_embed:
            query_embeddings_map[query_text] = self._embedding_function(query_text)

        metrics.embedding_generation_ms = (time.perf_counter() - stage_start) * 1000

        # ── Stage 3: Parallel Vector Retrieval ─────────────────────
        stage_start = time.perf_counter()

        vector_search_tasks = []
        for query_text, embedding in query_embeddings_map.items():
            vector_search_tasks.append(
                self._vector_store.similarity_search(
                    query_embedding=embedding,
                    top_k=self._reranker_settings.initial_retrieval_k,
                )
            )

        vector_results_batches = await asyncio.gather(*vector_search_tasks)

        metrics.vector_retrieval_ms = (time.perf_counter() - stage_start) * 1000

        # ── Stage 4: Keyword/BM25 Retrieval ────────────────────────
        stage_start = time.perf_counter()

        keyword_results: list[dict] = await self._keyword_search(
            raw_query,
            top_k=self._hybrid_settings.keyword_search_top_k,
        )

        metrics.keyword_retrieval_ms = (time.perf_counter() - stage_start) * 1000

        # ── Stage 5: Hybrid Fusion ─────────────────────────────────
        stage_start = time.perf_counter()

        fused_docs = self._hybrid_fusion(
            vector_batches=vector_results_batches,
            keyword_results=keyword_results,
            query_texts=all_queries_to_embed,
            correlation_id=correlation_id,
        )

        metrics.hybrid_fusion_ms = (time.perf_counter() - stage_start) * 1000
        metrics.total_documents_retrieved = len(fused_docs)

        # ── Stage 6: Cross-Encoder Re-ranking ──────────────────────
        stage_start = time.perf_counter()

        ranked_docs = await self._rerank_documents(
            query=raw_query,
            documents=fused_docs,
            correlation_id=correlation_id,
        )

        metrics.reranking_ms = (time.perf_counter() - stage_start) * 1000
        metrics.documents_after_pruning = len(ranked_docs)
        metrics.total_ms = (time.perf_counter() - total_start) * 1000

        result = RAGResult(
            query=raw_query,
            rewritten_queries=transformation.rewritten_queries,
            documents=ranked_docs,
            metrics=metrics,
            correlation_id=correlation_id,
        )

        logger.info(
            "RAG pipeline completed.",
            extra={
                "context": {
                    "correlation_id": correlation_id,
                    "query": raw_query[:200],
                    "documents_retrieved": metrics.total_documents_retrieved,
                    "documents_after_pruning": metrics.documents_after_pruning,
                    "total_latency_ms": f"{metrics.total_ms:.2f}",
                    "stage_breakdown": {
                        "query_transformation_ms": f"{metrics.query_transformation_ms:.2f}",
                        "embedding_generation_ms": f"{metrics.embedding_generation_ms:.2f}",
                        "vector_retrieval_ms": f"{metrics.vector_retrieval_ms:.2f}",
                        "keyword_retrieval_ms": f"{metrics.keyword_retrieval_ms:.2f}",
                        "hybrid_fusion_ms": f"{metrics.hybrid_fusion_ms:.2f}",
                        "reranking_ms": f"{metrics.reranking_ms:.2f}",
                    },
                }
            },
        )
        return result

    # ------------------------------------------------------------------
    # Keyword / BM25 Retrieval
    # ------------------------------------------------------------------

    async def _keyword_search(
        self,
        query: str,
        top_k: int,
    ) -> list[dict]:
        """Perform keyword/BM25-style lexical retrieval.

        In production, this would use Elasticsearch, PostgreSQL full-text
        search (tsvector/tsquery), or Tantivy. For this sandbox, we
        implement a lightweight in-memory TF-IDF-style scorer over the
        documents already indexed in the vector store.

        The keyword search compensates for vector search weaknesses with
        domain-specific terminology, acronyms, and rare tokens that
        embedding models may not capture well.
        """
        if top_k <= 0:
            return []

        try:
            vector_results = await self._vector_store.similarity_search(
                query_embedding=self._embedding_function(query),
                top_k=max(top_k, 20),
            )

            scored_results: list[tuple[float, dict]] = []
            query_terms = self._tokenize(query)

            for doc in vector_results:
                document_text = doc.get("document", "")
                lexical_score = self._compute_lexical_score(
                    query_terms, self._tokenize(document_text)
                )
                scored_results.append((lexical_score, doc))

            scored_results.sort(key=lambda x: x[0], reverse=True)
            top_results = [doc for _, doc in scored_results[:top_k]]

            logger.info(
                "Keyword search completed.",
                extra={
                    "context": {
                        "query_terms": list(query_terms),
                        "top_k": top_k,
                        "result_count": len(top_results),
                    }
                },
            )
            return top_results
        except Exception as exc:
            logger.warning(
                "Keyword search failed — returning empty results.",
                extra={"context": {"error_type": type(exc).__name__}},
                exc_info=True,
            )
            return []

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        """Simple tokenizer: lowercase, split on non-alphanumeric.

        In production, this would use a proper tokenizer (spaCy, NLTK,
        or a language-specific analyzer). This is adequate for the
        sandbox demonstration.
        """
        import re

        tokens = re.findall(r"\b[a-zA-Z0-9]{2,}\b", text.lower())
        return set(tokens)

    @staticmethod
    def _compute_lexical_score(
        query_terms: set[str],
        document_terms: set[str],
    ) -> float:
        """Compute a simple Jaccard-like overlap score between query and document terms.

        score = |Q ∩ D| / (|Q| + ε)

        This is a simplified proxy for BM25. Production systems should
        use a full implementation with IDF weighting and document length
        normalization.
        """
        if not query_terms:
            return 0.0

        intersection = query_terms & document_terms
        score = len(intersection) / (len(query_terms) + 1e-9)

        return score

    # ------------------------------------------------------------------
    # Hybrid Fusion
    # ------------------------------------------------------------------

    def _hybrid_fusion(
        self,
        vector_batches: list[list[dict]],
        keyword_results: list[dict],
        query_texts: list[str],
        correlation_id: str,
    ) -> list[RetrievedDocument]:
        """Fuse vector and keyword search results into a single ranked list.

        Implements Reciprocal Rank Fusion (RRF) by default:
          RRF_score(d) = Σ (1 / (k + rank_i(d)))

        Where rank_i(d) is document d's rank in result list i, and k is
        the smoothing parameter (default 60).

        Alternative: Weighted Sum (controlled via config).
        """
        fusion_method = self._hybrid_settings.fusion_method

        doc_map: dict[str, RetrievedDocument] = {}
        vector_rankings: dict[str, list[tuple[str, int, float]]] = defaultdict(list)

        for batch_idx, batch in enumerate(vector_batches):
            for rank, doc_dict in enumerate(batch):
                doc_text = doc_dict.get("document", "")
                doc_id = f"vec_{batch_idx}_{rank}_{hash(doc_text) & 0xFFFFF:05x}"

                if doc_id not in doc_map:
                    doc_map[doc_id] = RetrievedDocument(
                        document=doc_text,
                        metadata=doc_dict.get("metadata", {}),
                        vector_score=doc_dict.get("score", 0.0),
                        source="vector",
                    )

        for doc_id, doc in doc_map.items():
            doc.hybrid_score = doc.vector_score * self._hybrid_settings.vector_weight

        for rank, kw_doc in enumerate(keyword_results):
            kw_text = kw_doc.get("document", "")
            matched = False
            for doc_id, existing_doc in doc_map.items():
                if existing_doc.document == kw_text:
                    existing_doc.keyword_score = 1.0 / (rank + 1)
                    existing_doc.hybrid_score += (
                        existing_doc.keyword_score
                        * self._hybrid_settings.keyword_weight
                    )
                    matched = True
                    break

            if not matched:
                doc_id = f"kw_{rank}_{hash(kw_text) & 0xFFFFF:05x}"
                new_doc = RetrievedDocument(
                    document=kw_text,
                    metadata=kw_doc.get("metadata", {}),
                    keyword_score=1.0 / (rank + 1),
                    source="keyword",
                )
                new_doc.hybrid_score = (
                    new_doc.keyword_score * self._hybrid_settings.keyword_weight
                )
                doc_map[doc_id] = new_doc

        fused_docs = sorted(
            doc_map.values(),
            key=lambda d: d.hybrid_score,
            reverse=True,
        )

        top_k_before_rerank = min(
            len(fused_docs),
            self._reranker_settings.initial_retrieval_k * 2,
        )
        fused_docs = fused_docs[:top_k_before_rerank]

        logger.info(
            "Hybrid fusion completed.",
            extra={
                "context": {
                    "correlation_id": correlation_id,
                    "fusion_method": fusion_method,
                    "vector_batches": len(vector_batches),
                    "keyword_results": len(keyword_results),
                    "fused_count": len(fused_docs),
                }
            },
        )
        return fused_docs

    # ------------------------------------------------------------------
    # Cross-Encoder Re-ranking
    # ------------------------------------------------------------------

    async def _rerank_documents(
        self,
        query: str,
        documents: list[RetrievedDocument],
        correlation_id: str,
    ) -> list[RetrievedDocument]:
        """Re-rank documents using a cross-encoder model.

        Cross-encoders compute a relevance score by jointly encoding the
        query and each document, capturing fine-grained semantic interaction
        that bi-encoder (separate encoding) models miss.

        The pipeline re-ranks the top-N candidates (from hybrid fusion)
        and prunes to the top-K most relevant (default 5) to minimize
        context window pollution ("lost in the middle").

        Supports:
          - Local HuggingFace cross-encoder (via transformers pipeline)
          - Cohere Rerank API
          - Mock deterministic scorer (sandbox default)
        """
        if not documents:
            logger.info(
                "No documents to re-rank — skipping.",
                extra={"context": {"correlation_id": correlation_id}},
            )
            return []

        provider = self._reranker_settings.provider

        if provider == "local":
            scores = await self._local_rerank(query, documents, correlation_id)
        elif provider == "cohere":
            scores = await self._cohere_rerank(query, documents, correlation_id)
        else:
            scores = await self._mock_rerank(query, documents, correlation_id)

        for doc, score in zip(documents, scores):
            doc.rerank_score = float(score)

        documents.sort(key=lambda d: d.rerank_score, reverse=True)

        threshold = self._reranker_settings.min_relevance_score
        top_k = self._reranker_settings.top_k_after_rerank

        pruned: list[RetrievedDocument] = []
        for doc in documents:
            if doc.rerank_score >= threshold:
                pruned.append(doc)
            if len(pruned) >= top_k:
                break

        logger.info(
            "Cross-encoder re-ranking completed.",
            extra={
                "context": {
                    "correlation_id": correlation_id,
                    "provider": provider,
                    "input_count": len(documents),
                    "output_count": len(pruned),
                    "min_relevance_threshold": threshold,
                    "top_scores": [
                        f"{d.rerank_score:.4f}" for d in pruned[:3]
                    ],
                }
            },
        )
        return pruned

    async def _mock_rerank(
        self,
        query: str,
        documents: list[RetrievedDocument],
        correlation_id: str,
    ) -> list[float]:
        """Deterministic mock re-ranking for development/testing.

        Scores documents based on:
          - Lexical overlap with the query (Jaccard similarity)
          - Position in the original retrieval (rank bias favoring earlier docs)
          - A small deterministic perturbation seeded by document content

        This mimics cross-encoder behavior without requiring a GPU or
        external API. Production systems should use a real model.
        """
        query_terms = self._tokenize(query)
        scores: list[float] = []

        for idx, doc in enumerate(documents):
            doc_terms = self._tokenize(doc.document)

            lexical_score = self._compute_lexical_score(query_terms, doc_terms)

            position_bias = 1.0 / (1.0 + math.log(1 + idx))

            content_hash = hash(doc.document[:100]) % 100
            perturbation = (content_hash * 0.001)

            score = 0.5 * lexical_score + 0.3 * position_bias + 0.2 * perturbation
            score = max(0.0, min(1.0, score))
            scores.append(score)

        return scores

    async def _local_rerank(
        self,
        query: str,
        documents: list[RetrievedDocument],
        correlation_id: str,
    ) -> list[float]:
        """Local cross-encoder re-ranking via HuggingFace transformers.

        Uses sentence-transformers CrossEncoder or a Transformers pipeline.
        This is memory-intensive and CPU-bound; in production, it should
        be offloaded to a dedicated inference service.

        Falls back to mock if the model cannot be loaded.
        """
        try:
            from sentence_transformers import CrossEncoder

            model = CrossEncoder(self._reranker_settings.local_model_name)

            pairs = [(query, doc.document) for doc in documents]
            raw_scores = await asyncio.to_thread(model.predict, pairs)

            scores: list[float] = [
                float(s) for s in (raw_scores if isinstance(raw_scores, list) else [raw_scores])
            ]

            logger.info(
                "Local cross-encoder re-ranking completed.",
                extra={
                    "context": {
                        "correlation_id": correlation_id,
                        "model": self._reranker_settings.local_model_name,
                        "document_count": len(documents),
                    }
                },
            )
            return scores
        except ImportError:
            logger.warning(
                "sentence_transformers not installed — falling back to mock re-ranker.",
                extra={"context": {"correlation_id": correlation_id}},
            )
            return await self._mock_rerank(query, documents, correlation_id)
        except Exception as exc:
            logger.error(
                "Local cross-encoder failed — falling back to mock.",
                extra={
                    "context": {
                        "correlation_id": correlation_id,
                        "error_type": type(exc).__name__,
                    }
                },
                exc_info=True,
            )
            return await self._mock_rerank(query, documents, correlation_id)

    async def _cohere_rerank(
        self,
        query: str,
        documents: list[RetrievedDocument],
        correlation_id: str,
    ) -> list[float]:
        """Cohere Rerank API re-ranking.

        Calls the Cohere Rerank endpoint with the query and all candidate
        documents. Returns relevance scores in [0, 1].

        Falls back to mock if the API key is not configured or the
        request fails.
        """
        api_key = self._reranker_settings.cohere_api_key.get_secret_value()
        if not api_key:
            logger.warning(
                "Cohere API key not configured — falling back to mock re-ranker.",
                extra={"context": {"correlation_id": correlation_id}},
            )
            return await self._mock_rerank(query, documents, correlation_id)

        try:
            import httpx

            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    "https://api.cohere.ai/v1/rerank",
                    json={
                        "model": self._reranker_settings.cohere_model,
                        "query": query,
                        "documents": [doc.document for doc in documents],
                        "top_n": len(documents),
                    },
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    },
                )
                response.raise_for_status()
                data = response.json()

            results = data.get("results", [])
            scores_map: dict[int, float] = {
                r.get("index", 0): r.get("relevance_score", 0.0)
                for r in results
            }
            scores = [
                scores_map.get(i, 0.0) for i in range(len(documents))
            ]

            logger.info(
                "Cohere re-ranking completed.",
                extra={
                    "context": {
                        "correlation_id": correlation_id,
                        "model": self._reranker_settings.cohere_model,
                        "document_count": len(documents),
                    }
                },
            )
            return scores
        except Exception as exc:
            logger.error(
                "Cohere re-ranking failed — falling back to mock.",
                extra={
                    "context": {
                        "correlation_id": correlation_id,
                        "error_type": type(exc).__name__,
                    }
                },
                exc_info=True,
            )
            return await self._mock_rerank(query, documents, correlation_id)

    # ------------------------------------------------------------------
    # Embedding Function
    # ------------------------------------------------------------------

    def _mock_embedding_function(self, text: str) -> Embedding:
        """Deterministic mock embedding generator.

        Generates embeddings seeded by text content so the same text
        always produces the same embedding. Uses character-level hashing
        with sinusoidal positional encoding to create pseudo-semantic
        vector representations.

        In production, replace with sentence-transformers or an embedding API.
        """
        import hashlib

        settings = get_settings().embeddings
        dim = getattr(settings, "target_dimension", 384)

        seed = int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], 16)
        rng_state = seed

        embedding: list[float] = []
        for i in range(dim):
            rng_state = (rng_state * 1103515245 + 12345) & 0x7FFFFFFF
            raw = (rng_state / 0x7FFFFFFF) * 2.0 - 1.0
            phase = math.sin(i * 0.001 + seed * 1e-9)
            embedding.append(raw * 0.3 + phase * 0.05)

        norm = math.sqrt(sum(v * v for v in embedding))
        if norm > 0:
            embedding = [v / norm for v in embedding]

        return embedding

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def shutdown(self) -> None:
        """Release any resources held by the RAG engine."""
        correlation_id = str(uuid.uuid4())
        logger.info(
            "RAG engine shutting down.",
            extra={"context": {"correlation_id": correlation_id}},
        )
        await self._vector_store.shutdown()
