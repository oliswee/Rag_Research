"""Application configuration using Pydantic Settings."""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ModelID:
    """Anthropic model identifiers.

    These are validated against all_models() to catch typos/deprecated models.
    Update both when adding new models.
    """

    # Claude 3 models
    HAIKU_3 = "claude-3-haiku-20240307"

    # Claude 3.5 models
    HAIKU_3_5 = "claude-3-5-haiku-20241022"

    # Claude 4 models
    SONNET_4 = "claude-sonnet-4-20250514"

    @classmethod
    def all_models(cls) -> set[str]:
        """Return all valid model IDs."""
        return {cls.HAIKU_3, cls.HAIKU_3_5, cls.SONNET_4}

    @classmethod
    def validate(cls, model_id: str) -> None:
        """Validate a model ID is known.

        Args:
            model_id: Model identifier to validate.

        Raises:
            ValueError: If model_id is not in the known list.
        """
        if model_id not in cls.all_models():
            valid = ", ".join(sorted(cls.all_models()))
            raise ValueError(f"Unknown model ID: {model_id}. Valid models: {valid}")


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # API Keys
    anthropic_api_key: str = ""
    openai_api_key: str = ""
    semantic_scholar_api_key: str = ""
    ncbi_api_key: str = ""
    ncbi_email: str = ""
    unpaywall_email: str = ""

    # LLM Settings
    anthropic_model: str = ModelID.SONNET_4
    llm_provider: str = "ollama"
    ollama_base_url: str = "http://localhost:11434/v1/"
    ollama_model: str = "qwen3:4b-instruct"
    deepseek_api_key: str = ""
    deepseek_model: str = "deepseek-flash"

    # Local embeddings are the default; OpenAI remains an explicit option.
    embedding_provider: str = "fastembed"
    fastembed_model: str = "BAAI/bge-small-en-v1.5"

    # Database
    database_url: str = "sqlite:///./data/research.db"
    retrieval_trace_path: str = "data/traces/retrieval_trace.jsonl"
    query_variant_limit: int = Field(3, ge=1, le=3)
    query_rewrite_timeout_seconds: float = Field(20, gt=0, le=60)
    retrieval_branch_timeout_seconds: float = Field(15, gt=0, le=60)
    retrieval_concurrency: int = Field(3, ge=1, le=6)
    retrieval_per_query_k: int = Field(30, ge=1, le=100)
    retrieval_rrf_k: int = Field(60, ge=1, le=200)

    # Qdrant
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333

    # Ingestion Settings
    max_pdf_pages: int = 200
    max_file_size_mb: int = 10
    extraction_timeout_seconds: int = 120
    extraction_max_retries: int = 3

    # LangSmith (optional)
    langchain_tracing_v2: bool = False
    langchain_api_key: str = ""
    langchain_project: str = "foliotrace"


settings = Settings()
