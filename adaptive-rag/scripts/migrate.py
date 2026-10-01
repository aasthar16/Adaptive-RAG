"""
Run all SQL migrations in `database/migrations/` (in filename order) and
initialize the LangGraph PostgreSQL checkpointer tables.

Usage:
    python -m scripts.migrate
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import get_settings  # noqa: E402
from database.connection import get_connection  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("migrate")

MIGRATIONS_DIR = Path(__file__).resolve().parent.parent / "database" / "migrations"


def run_sql_migrations() -> None:
    sql_files = sorted(MIGRATIONS_DIR.glob("*.sql"))
    if not sql_files:
        logger.warning("No SQL migration files found in %s", MIGRATIONS_DIR)
        return
    with get_connection() as conn:
        for path in sql_files:
            logger.info("Applying migration: %s", path.name)
            sql = path.read_text(encoding="utf-8")
            with conn.cursor() as cur:
                cur.execute(sql)
    logger.info("All SQL migrations applied.")


def setup_langgraph_checkpointer() -> None:
    from langgraph.checkpoint.postgres import PostgresSaver

    settings = get_settings()
    logger.info("Initializing LangGraph checkpoint tables...")
    with PostgresSaver.from_conn_string(settings.database_url_psycopg) as saver:
        saver.setup()
    logger.info("LangGraph checkpoint tables ready.")


def main() -> None:
    settings = get_settings()
    problems = settings.validate()
    for p in problems:
        logger.warning("Config warning: %s", p)

    run_sql_migrations()
    setup_langgraph_checkpointer()
    logger.info("Migration complete.")


if __name__ == "__main__":
    main()
