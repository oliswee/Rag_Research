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
    fastembed_cache_dir: str = "data/models/embedding"

    # Database
    database_url: str = "sqlite:///./data/research.db"
    retrieval_trace_path: str = "data/traces/retrieval_trace.jsonl"
    query_variant_limit: int = Field(3, ge=1, le=3)
    query_rewrite_timeout_seconds: float = Field(20, gt=0, le=20)
    retrieval_branch_timeout_seconds: float = Field(15, gt=0, le=15)
    retrieval_concurrency: int = Field(3, ge=1, le=3)
    retrieval_per_query_k: int = Field(30, ge=1, le=30)
    retrieval_rrf_k: int = Field(60, ge=1, le=60)
    reranker_model: str = "Xenova/ms-marco-MiniLM-L-6-v2"
    reranker_cache_dir: str = "data/models/reranker"
    reranker_candidate_limit: int = Field(30, ge=10, le=50)
    reranker_selected_limit: int = Field(10, ge=5, le=10)
    reranker_timeout_seconds: float = Field(15, gt=0, le=15)
    context_token_budget: int = Field(8192, ge=128, le=16384)
    context_parent_budget: int = Field(2048, ge=128, le=4096)
    web_max_attempts: int = Field(8, ge=1, le=8)
    web_timeout_seconds: float = Field(10, gt=0, le=10)
    web_deadline_seconds: float = Field(45, gt=0, le=45)
    web_metadata_bytes: int = Field(524288, ge=1024, le=524288)
    web_candidates_per_provider: int = Field(8, ge=1, le=8)
    web_pdf_bytes: int = Field(3145728, ge=1024, le=3145728)
    web_pdf_pages: int = Field(50, ge=1, le=50)
    web_parser_timeout_seconds: float = Field(15, gt=0, le=15)
    verification_max_calls: int = Field(10, ge=1, le=10)
    verification_output_tokens: int = Field(8192, ge=128, le=8192)
    verification_input_bytes: int = Field(131072, ge=128, le=131072)
    verification_deadline_seconds: float = Field(180, gt=0, le=180)
    verification_call_timeout_seconds: float = Field(30, gt=0, le=30)
    request_deadline_seconds: float = Field(240, gt=0, le=300)
    request_max_model_attempts: int = Field(12, ge=1, le=20)
    request_max_input_bytes: int = Field(262144, ge=128, le=1048576)
    request_max_output_tokens: int = Field(16384, ge=128, le=32768)
    request_max_external_attempts: int = Field(12, ge=0, le=24)
    request_max_currency: float | None = Field(None, ge=0)
    request_input_price_per_million: float | None = Field(None, ge=0)
    request_output_price_per_million: float | None = Field(None, ge=0)
    request_embedding_price_per_million: float | None = Field(None, ge=0)

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
