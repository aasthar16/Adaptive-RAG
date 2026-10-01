"""
Centralized application configuration.

Every setting is loaded from environment variables (via `.env` in local dev).
Nothing here should ever be hardcoded with a real secret. `Settings` is a
singleton accessed through `get_settings()` so the rest of the app never
touches `os.environ` directly.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

# Load .env once, as early as possible. Real environment variables (e.g. set
# by Docker/K8s) always take precedence over values in the .env file.
_ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=_ENV_PATH, override=False)


def _bool(name: str, default: bool = False) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "on"}


def _float(name: str, default: float) -> float:
    val = os.getenv(name)
    try:
        return float(val) if val is not None else default
    except ValueError:
        return default


def _int(name: str, default: int) -> int:
    val = os.getenv(name)
    try:
        return int(val) if val is not None else default
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    # --- Database -----------------------------------------------------
    database_url: str = field(
        default_factory=lambda: os.getenv(
            "DATABASE_URL",
            "postgresql://adaptive_rag:adaptive_rag@localhost:5432/adaptive_rag",
        )
    )

    @property
    def database_url_psycopg(self) -> str:
        """Plain psycopg3 DSN (used by psycopg pool + LangGraph checkpointer)."""
        url = self.database_url
        # Normalize a SQLAlchemy-style URL down to a plain psycopg URL.
        return url.replace("postgresql+psycopg://", "postgresql://")

    @property
    def database_url_sqlalchemy(self) -> str:
        """SQLAlchemy-async-style DSN required by langchain-postgres (PGEngine)."""
        url = self.database_url
        if url.startswith("postgresql+psycopg://"):
            return url
        if url.startswith("postgresql://"):
            return url.replace("postgresql://", "postgresql+psycopg://", 1)
        return url

    # --- LLM provider ----------------------------------------------------
    llm_provider: str = field(default_factory=lambda: os.getenv("LLM_PROVIDER", "openai"))
    openai_api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    chat_model: str = field(default_factory=lambda: os.getenv("CHAT_MODEL", "gpt-4o-mini"))
    fast_model: str = field(default_factory=lambda: os.getenv("FAST_MODEL", "gpt-4o-mini"))
    embedding_model: str = field(
        default_factory=lambda: os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    )
    embedding_dimensions: int = field(
        default_factory=lambda: _int("EMBEDDING_DIMENSIONS", 1536)
    )

    # --- LangSmith ---------------------------------------------------------
    langsmith_tracing: bool = field(default_factory=lambda: _bool("LANGSMITH_TRACING", False))
    langsmith_api_key: str = field(default_factory=lambda: os.getenv("LANGSMITH_API_KEY", ""))
    langsmith_project: str = field(
        default_factory=lambda: os.getenv("LANGSMITH_PROJECT", "AdaptiveRAG")
    )
    langsmith_endpoint: str = field(
        default_factory=lambda: os.getenv(
            "LANGSMITH_ENDPOINT", "https://api.smith.langchain.com"
        )
    )

    # --- Web / external search -------------------------------------------
    enable_web_search: bool = field(default_factory=lambda: _bool("ENABLE_WEB_SEARCH", False))
    web_search_api_key: str = field(default_factory=lambda: os.getenv("WEB_SEARCH_API_KEY", ""))

    # --- Auth / sessions -----------------------------------------------
    secret_key: str = field(default_factory=lambda: os.getenv("SECRET_KEY", "dev-insecure-secret-change-me"))
    session_ttl_hours: int = field(default_factory=lambda: _int("SESSION_TTL_HOURS", 24 * 7))

    # --- File uploads ------------------------------------------------------
    upload_dir: Path = field(
        default_factory=lambda: Path(os.getenv("UPLOAD_DIR", "./data/uploads")).resolve()
    )
    max_upload_mb: int = field(default_factory=lambda: _int("MAX_UPLOAD_MB", 25))
    allowed_extensions: tuple = (".pdf", ".txt", ".docx", ".md")

    # --- Adaptive retrieval thresholds -------------------------------------
    sufficient_threshold: float = field(
        default_factory=lambda: _float("SUFFICIENT_THRESHOLD", 0.75)
    )
    partial_threshold: float = field(default_factory=lambda: _float("PARTIAL_THRESHOLD", 0.45))
    max_retrieval_iterations: int = field(
        default_factory=lambda: _int("MAX_RETRIEVAL_ITERATIONS", 3)
    )
    max_verification_retries: int = field(
        default_factory=lambda: _int("MAX_VERIFICATION_RETRIES", 1)
    )
    default_top_k: int = field(default_factory=lambda: _int("DEFAULT_TOP_K", 6))
    rerank_top_n: int = field(default_factory=lambda: _int("RERANK_TOP_N", 5))

    # --- Misc ------------------------------------------------------------
    app_env: str = field(default_factory=lambda: os.getenv("APP_ENV", "development"))
    log_level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))

    def validate(self) -> list[str]:
        """Return a list of human-readable configuration problems (empty = OK)."""
        problems: list[str] = []
        if not self.database_url:
            problems.append("DATABASE_URL is not set.")
        if self.llm_provider == "openai" and not self.openai_api_key:
            problems.append(
                "OPENAI_API_KEY is not set. LLM/embedding calls will fail until it is."
            )
        if self.secret_key == "dev-insecure-secret-change-me" and self.app_env == "production":
            problems.append("SECRET_KEY must be overridden in production.")
        if self.enable_web_search and not self.web_search_api_key:
            problems.append("ENABLE_WEB_SEARCH is true but WEB_SEARCH_API_KEY is empty.")
        return problems


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
