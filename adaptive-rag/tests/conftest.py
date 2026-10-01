"""
Shared pytest fixtures.

Tests that touch the database run against a dedicated `adaptive_rag_test`
database (see `TEST_DATABASE_URL` below) — never the dev database — and
every table is truncated before each test for isolation. Tests that need
an LLM are given a stubbed structured-output model via `patch_llm` rather
than calling a real API, so the whole suite (except a couple of
explicitly-marked integration tests) runs with no external network access
and no API key required.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

TEST_DATABASE_URL = os.environ.get(
    "TEST_DATABASE_URL",
    "postgresql://adaptive_rag:adaptive_rag@localhost:5432/adaptive_rag_test",
)
# Must happen before the first `from config.settings import get_settings`
# anywhere in the process, since Settings() reads the environment once and
# is then cached.
os.environ["DATABASE_URL"] = TEST_DATABASE_URL
os.environ.setdefault("SECRET_KEY", "test-secret-key")
os.environ.setdefault("OPENAI_API_KEY", "sk-test-not-a-real-key")

import pytest  # noqa: E402

from config.settings import get_settings  # noqa: E402
from database.connection import close_pool, get_connection  # noqa: E402

APP_TABLES = ["document_chunks", "messages", "documents", "chats", "users"]


@pytest.fixture(scope="session", autouse=True)
def _setup_test_database():
    get_settings.cache_clear()
    settings = get_settings()
    assert "adaptive_rag_test" in settings.database_url, (
        "Refusing to run tests against a non-test database: " + settings.database_url
    )

    from scripts.migrate import run_sql_migrations, setup_langgraph_checkpointer

    run_sql_migrations()
    setup_langgraph_checkpointer()
    yield
    close_pool()


@pytest.fixture(autouse=True)
def _clean_tables():
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(f"TRUNCATE {', '.join(APP_TABLES)} RESTART IDENTITY CASCADE;")
    yield


@pytest.fixture
def make_user():
    from auth.password import hash_password
    from database.repositories.user_repository import UserRepository

    repo = UserRepository()

    def _make(email: str = "test@example.com", password: str = "Password123") -> object:
        return repo.create(email=email, password_hash=hash_password(password))

    return _make
