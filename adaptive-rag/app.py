"""
AdaptiveRAG — Streamlit entrypoint.

Run with:
    streamlit run app.py

This file only handles top-level routing (logged out -> login page;
logged in -> sidebar + active chat). All state that matters (users, chats,
messages, documents, vectors, LangGraph checkpoints) lives in PostgreSQL —
see `config/settings.py` for `DATABASE_URL` and `auth/session.py` for the
(explicit) distinction between that persistent data and this file's
UI-only use of `st.session_state`.
"""
from __future__ import annotations

import logging

import streamlit as st

from auth.session import get_active_chat, is_authenticated
from config.settings import get_settings
from database.connection import healthcheck
from observability.langsmith import init_langsmith
from ui.chat import render_chat_page, render_empty_state
from ui.login import render_login_page
from ui.sidebar import render_sidebar

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

st.set_page_config(page_title="AdaptiveRAG", page_icon="🧠", layout="wide")


@st.cache_resource
def _startup() -> dict:
    """Runs once per process (Streamlit caches `cache_resource` across
    reruns/sessions). Validates config and warms the LangSmith env vars."""
    settings = get_settings()
    init_langsmith()
    return {"problems": settings.validate(), "db_ok": healthcheck()}


def main() -> None:
    startup = _startup()

    for problem in startup["problems"]:
        st.sidebar.warning(problem, icon="⚠️")

    if not startup["db_ok"]:
        st.error(
            "Could not connect to PostgreSQL. Check `DATABASE_URL` in your `.env` file and make "
            "sure the database is running (`docker-compose up -d`), then reload this page."
        )
        st.stop()

    if not is_authenticated():
        render_login_page()
        return

    active_chat_id = get_active_chat()
    render_sidebar(active_chat_id)

    if active_chat_id:
        render_chat_page(active_chat_id)
    else:
        render_empty_state()


if __name__ == "__main__":
    main()
