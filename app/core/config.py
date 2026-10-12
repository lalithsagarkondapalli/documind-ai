"""Central settings, loaded from environment variables (see .env.example)."""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_name: str = "documind-ai"
    environment: str = "local"

    # Auth
    jwt_secret: str = "change-me-in-production"
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 30
    # Demo users for local development only. Production would use an IdP (OAuth2/OIDC).
    demo_password: str = "documind-demo"

    # Storage
    database_url: str = "sqlite:///./documind.db"
    redis_url: str | None = None
    cache_ttl_seconds: int = 3600
    vector_backend: str = "memory"  # "memory" | "qdrant"
    qdrant_url: str | None = None  # None + backend=qdrant -> embedded in-process Qdrant

    # Agent behaviour
    review_confidence_threshold: float = 0.85
    retrieval_top_k: int = 5
    index_sync_interval_seconds: float = 2.0  # how often a replica checks the DB for new chunks
    llm_provider: str = "mock"  # "mock" | "anthropic"
    small_model: str = "claude-haiku-4-5"
    frontier_model: str = "claude-sonnet-4-5"

    # Observability
    otlp_endpoint: str | None = None


@lru_cache
def get_settings() -> Settings:
    return Settings()
