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
from backend.rag_pipeline.query_processor import QueryTransformer
from backend.vector_store.base import BaseVectorStore, SearchResult

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
