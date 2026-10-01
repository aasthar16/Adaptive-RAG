"""
Thin session layer.

Important distinction from the rest of the app: `st.session_state` is used
here ONLY to remember which user is currently logged in in this browser
tab, and which chat is open. It is UI state, not application data — every
chat, message, and document is persisted in PostgreSQL (see
`database/repositories`) and reloaded from there on every rerun. If the
Streamlit process restarts, the user simply has to log in again; no data
is lost because none of it lived in session state.
"""
from __future__ import annotations

from typing import Optional

import streamlit as st

from database.models import User

_SESSION_USER_KEY = "auth_user"
_ACTIVE_CHAT_KEY = "active_chat_id"


def login_user(user: User) -> None:
    st.session_state[_SESSION_USER_KEY] = {
        "id": user.id,
        "email": user.email,
    }


def logout_user() -> None:
    st.session_state.pop(_SESSION_USER_KEY, None)
    st.session_state.pop(_ACTIVE_CHAT_KEY, None)


def get_current_user() -> Optional[dict]:
    return st.session_state.get(_SESSION_USER_KEY)


def is_authenticated() -> bool:
    return _SESSION_USER_KEY in st.session_state


def set_active_chat(chat_id: Optional[str]) -> None:
    st.session_state[_ACTIVE_CHAT_KEY] = chat_id


def get_active_chat() -> Optional[str]:
    return st.session_state.get(_ACTIVE_CHAT_KEY)
