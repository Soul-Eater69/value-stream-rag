"""
Centralised application settings via pydantic-settings.

All values can be overridden by environment variables or an .env file.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class AzureSearchSettings(BaseSettings):
    """Azure AI Search configuration."""

    model_config = SettingsConfigDict(env_prefix="AZURE_SEARCH_", extra="ignore")

    endpoint: str = Field(default="", description="Azure AI Search endpoint URL")
    api_key: str = Field(default="", description="Azure AI Search admin key")
    value_stream_index: str = Field(default="value-streams")
    historical_ppts_index: str = Field(default="historical-idea-cards")
    api_version: str = Field(default="2024-05-01-preview")

    # Semantic configuration name (if configured in Azure portal)
    semantic_config_name: str = Field(default="default")


class AzureOpenAISettings(BaseSettings):
    """Azure OpenAI or plain OpenAI configuration."""

    model_config = SettingsConfigDict(env_prefix="AZURE_OPENAI_", extra="ignore")

    endpoint: str = Field(default="")
    api_key: str = Field(default="")
    api_version: str = Field(default="2024-02-01")
    embedding_deployment: str = Field(default="text-embedding-3-large")
    chat_deployment: str = Field(default="gpt-4o")

    # Fallback: plain OpenAI
    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")
    openai_embedding_model: str = Field(
        default="text-embedding-3-large", alias="OPENAI_EMBEDDING_MODEL"
    )
    openai_chat_model: str = Field(default="gpt-4o", alias="OPENAI_CHAT_MODEL")

    @property
    def use_azure(self) -> bool:
        return bool(self.endpoint and self.api_key)


class ChromaSettings(BaseSettings):
    """Local ChromaDB vector store configuration."""

    model_config = SettingsConfigDict(env_prefix="CHROMA_", extra="ignore")

    persist_dir: Path = Field(default=Path("./data/indexes/chroma"))
    historical_collection: str = Field(default="historical_idea_cards")
    value_stream_collection: str = Field(default="value_streams_local")


class ChunkingSettings(BaseSettings):
    """Chunking behaviour."""

    model_config = SettingsConfigDict(env_prefix="CHUNK_", extra="ignore")

    max_tokens: int = Field(default=512, ge=128, le=2048)
    overlap_tokens: int = Field(default=64, ge=0)
    slide_level_chunking: bool = Field(default=True)
    table_aware_chunking: bool = Field(default=True)


class RetrievalSettings(BaseSettings):
    """Retrieval and ranking tunables."""

    model_config = SettingsConfigDict(extra="ignore")

    top_k_value_streams: int = Field(default=10, alias="TOP_K_VALUE_STREAMS")
    top_k_historical: int = Field(default=20, alias="TOP_K_HISTORICAL")
    rerank_top_n: int = Field(default=5, alias="RERANK_TOP_N")
    similarity_threshold: float = Field(default=0.6, alias="SIMILARITY_THRESHOLD")
    hybrid_search_alpha: float = Field(
        default=0.5,
        alias="HYBRID_SEARCH_ALPHA",
        description="0=pure keyword, 1=pure vector",
    )

    @field_validator("hybrid_search_alpha")
    @classmethod
    def validate_alpha(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError("hybrid_search_alpha must be between 0 and 1")
        return v


class Settings(BaseSettings):
    """Root application settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # Application
    app_env: str = Field(default="development")
    log_level: str = Field(default="INFO")
    max_upload_size_mb: int = Field(default=50)
    upload_dir: Path = Field(default=Path("./data/uploads"))

    # Database
    database_url: str = Field(default="sqlite:///./data/value_stream_rag.db")

    # Nested settings (instantiated lazily)
    @property
    def azure_search(self) -> AzureSearchSettings:
        return AzureSearchSettings()

    @property
    def azure_openai(self) -> AzureOpenAISettings:
        return AzureOpenAISettings()

    @property
    def chroma(self) -> ChromaSettings:
        return ChromaSettings()

    @property
    def chunking(self) -> ChunkingSettings:
        return ChunkingSettings()

    @property
    def retrieval(self) -> RetrievalSettings:
        return RetrievalSettings()

    @property
    def is_production(self) -> bool:
        return self.app_env.lower() == "production"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached singleton Settings instance."""
    return Settings()
