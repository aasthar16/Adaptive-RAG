"""
PostgreSQL connection management using psycopg3.

A single process-wide connection pool is created lazily and reused by every
repository. Streamlit reruns the script top-to-bottom on every interaction,
so the pool is cached with `st.cache_resource` where Streamlit is available,
and falls back to a simple module-level singleton otherwise (e.g. in tests
and scripts).
"""
from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Iterator

import psycopg
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from config.settings import get_settings

logger = logging.getLogger(__name__)

_pool: ConnectionPool | None = None


def _build_pool() -> ConnectionPool:
    settings = get_settings()
    pool = ConnectionPool(
        conninfo=settings.database_url_psycopg,
        min_size=1,
        max_size=10,
        kwargs={"row_factory": dict_row, "autocommit": False},
        open=True,
    )
    logger.info("PostgreSQL connection pool created.")
    return pool


def get_pool() -> ConnectionPool:
    global _pool
    if _pool is None:
        _pool = _build_pool()
    return _pool


def close_pool() -> None:
    global _pool
    if _pool is not None:
        _pool.close()
        _pool = None


@contextmanager
def get_connection() -> Iterator[psycopg.Connection]:
    """Yield a pooled connection; commits on success, rolls back on error."""
    pool = get_pool()
    with pool.connection() as conn:
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise


@contextmanager
def get_cursor() -> Iterator[psycopg.Cursor]:
    with get_connection() as conn:
        with conn.cursor() as cur:
            yield cur


def healthcheck() -> bool:
    try:
        with get_cursor() as cur:
            cur.execute("SELECT 1;")
            cur.fetchone()
        return True
    except Exception as exc:  # pragma: no cover - defensive
        logger.error("Database healthcheck failed: %s", exc)
        return False
