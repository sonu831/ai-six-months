"""
Centralised configuration via pydantic-settings v2.

Each sub-config reads its own env-prefix group so .env files remain legible.
get_settings() returns a process-wide singleton; reset it in tests with
    monkeypatch.setattr("backend.core.config._settings", None)
"""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class DatabaseSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="DB_", extra="ignore")

    host: str = "localhost"
    port: int = 5432
    name: str = "enterprise_ai"
    user: str = "postgres"
    password: SecretStr = SecretStr("changeme")
    pool_min_size: int = 5
    pool_max_size: int = 20
    pool_max_queries: int = 50_000
    pool_max_inactive_connection_lifetime: float = 300.0

    @property
    def dsn(self) -> str:
        return (
            f"postgresql://{self.user}:{self.password.get_secret_value()}"
            f"@{self.host}:{self.port}/{self.name}"
        )


class OpenAISettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="OPENAI_", extra="ignore")

    api_key: SecretStr = SecretStr("")
    default_model: str = "gpt-4o"
    embedding_model: str = "text-embedding-3-large"
    max_retries: int = 3
    timeout_seconds: float = 30.0


class AnthropicSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="ANTHROPIC_", extra="ignore")

    api_key: SecretStr = SecretStr("")
    default_model: str = "claude-sonnet-4-6"
    max_retries: int = 3
    timeout_seconds: float = 30.0


class OllamaSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="OLLAMA_", extra="ignore")

    base_url: str = "http://localhost:11434"
    default_model: str = "llama3.2"
    embedding_model: str = "nomic-embed-text"


class CohereSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="COHERE_", extra="ignore")

    api_key: SecretStr | None = None
    rerank_model: str = "rerank-english-v3.0"


class ChromaSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="CHROMA_", extra="ignore")

    persist_directory: str = "./data/vector_db/chroma"
    collection_name: str = "enterprise_docs"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    environment: str = "development"
    log_level: str = "INFO"

    db: DatabaseSettings = Field(default_factory=DatabaseSettings)
    openai: OpenAISettings = Field(default_factory=OpenAISettings)
    anthropic: AnthropicSettings = Field(default_factory=AnthropicSettings)
    ollama: OllamaSettings = Field(default_factory=OllamaSettings)
    cohere: CohereSettings = Field(default_factory=CohereSettings)
    chroma: ChromaSettings = Field(default_factory=ChromaSettings)

    @property
    def is_development(self) -> bool:
        return self.environment == "development"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
