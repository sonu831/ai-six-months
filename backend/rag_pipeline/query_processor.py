"""
Query transformation layer — LLM rewriting and Hypothetical Document Embedding (HyDE).

Implements two complementary retrieval-augmentation techniques:

    1. Query Rewriting: generates N semantically diverse reformulations
       of the user query to maximise recall across different vocabulary
       and framing patterns in the vector index.

    2. Hypothetical Document Embedding (HyDE): generates a plausible
       answer passage for the query, then uses its embedding as the
       search vector. This bridges the semantic gap between short
       user queries and dense document embeddings.

Both strategies are invoked in parallel and any individual failure is
gracefully degraded — the pipeline always has at least the original
query to work with.
"""

from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING, Any

import structlog

from backend.core.exceptions import QueryTransformationError
from backend.llm_client.base import (
    BaseLLMClient,
    ChatCompletionRequest,
    ChatMessage,
)

if TYPE_CHECKING:
    from backend.rag_pipeline.engine import RAGConfig

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

    def __init__(self, llm_client: BaseLLMClient, config: "RAGConfig") -> None:
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
