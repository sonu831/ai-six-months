"""
Typed exception hierarchy for enterprise-ai-sandbox.

Hierarchy:
    EnterpriseAIError                   ← root
        ConfigurationError
        VectorStoreError
            EmbeddingError
            UpsertError
            SearchError
            MigrationError
        LLMClientError
            RateLimitError
            AuthenticationError
            ContextLengthError
            ProviderUnavailableError
            ModelNotFoundError
        RAGPipelineError
            QueryTransformationError
            RetrievalError
            ReRankingError
            ContextAssemblyError
        AgentError
            AgentLoopDetectedError
            AgentTimeoutError
            ToolExecutionError
"""

from __future__ import annotations

from typing import Any


class EnterpriseAIError(Exception):
    """Root exception. All subsystem errors inherit from this."""

    def __init__(self, message: str, context: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.context: dict[str, Any] = context or {}

    def __str__(self) -> str:
        if self.context:
            ctx_str = ", ".join(f"{k}={v!r}" for k, v in self.context.items())
            return f"{super().__str__()} [{ctx_str}]"
        return super().__str__()


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


class ConfigurationError(EnterpriseAIError):
    """Raised when environment config is missing or invalid at startup."""


# ---------------------------------------------------------------------------
# Vector Store
# ---------------------------------------------------------------------------


class VectorStoreError(EnterpriseAIError):
    """Base for all vector store failures."""


class EmbeddingError(VectorStoreError):
    """Embedding generation or dimension mismatch."""


class UpsertError(VectorStoreError):
    """Batch write to the vector store failed."""


class SearchError(VectorStoreError):
    """Similarity or keyword search failed."""


class MigrationError(VectorStoreError):
    """Schema migration / DDL execution failed."""


# ---------------------------------------------------------------------------
# LLM Client
# ---------------------------------------------------------------------------


class LLMClientError(EnterpriseAIError):
    """Base for all LLM provider failures."""


class RateLimitError(LLMClientError):
    """Provider rate limit hit (HTTP 429)."""


class AuthenticationError(LLMClientError):
    """API key invalid or expired (HTTP 401/403)."""


class ContextLengthError(LLMClientError):
    """Prompt exceeds the model's context window."""


class ProviderUnavailableError(LLMClientError):
    """Provider is unreachable or returned a 5xx."""


class ModelNotFoundError(LLMClientError):
    """Requested model does not exist on the provider."""


# ---------------------------------------------------------------------------
# RAG Pipeline
# ---------------------------------------------------------------------------


class RAGPipelineError(EnterpriseAIError):
    """Base for RAG workflow failures."""


class QueryTransformationError(RAGPipelineError):
    """LLM-based query rewriting or HyDE generation failed."""


class RetrievalError(RAGPipelineError):
    """Hybrid retrieval produced zero usable candidates."""


class ReRankingError(RAGPipelineError):
    """Cross-encoder or Cohere re-ranking failed."""


class ContextAssemblyError(RAGPipelineError):
    """Context string could not be assembled within token budget."""


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


class AgentError(EnterpriseAIError):
    """Base for agentic loop failures."""


class AgentLoopDetectedError(AgentError):
    """Agent exceeded max_steps without reaching a terminal state."""


class AgentTimeoutError(AgentError):
    """Agent wall-clock timeout exceeded."""


class ToolExecutionError(AgentError):
    """A tool invoked by the agent raised an unhandled exception."""
