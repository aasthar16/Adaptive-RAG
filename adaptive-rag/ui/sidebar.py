from __future__ import annotations

from datetime import datetime, timezone

import streamlit as st

from auth.session import get_current_user, logout_user, set_active_chat
from services.chat_service import ChatService
from ui.upload import new_chat_dialog

_chats = ChatService()


def _group_label(updated_at: datetime) -> str:
    now = datetime.now(timezone.utc)
    if updated_at.tzinfo is None:
        updated_at = updated_at.replace(tzinfo=timezone.utc)
    delta_days = (now.date() - updated_at.date()).days
    if delta_days <= 0:
        return "Today"
    if delta_days == 1:
        return "Yesterday"
    if delta_days <= 7:
        return "This week"
    return "Older"


def render_sidebar(active_chat_id: str | None) -> None:
    user = get_current_user()

    with st.sidebar:
        st.markdown("## 🧠 AdaptiveRAG")

        if st.button("＋ New Chat", use_container_width=True, type="primary"):
            new_chat_dialog()

        st.markdown("### Recent Chats")
        chats = _chats.list_chats(user["id"])

        if not chats:
            st.caption("No chats yet — create one to get started.")

        groups: dict[str, list] = {}
        for chat in chats:
            groups.setdefault(_group_label(chat.updated_at), []).append(chat)

        for label in ["Today", "Yesterday", "This week", "Older"]:
            chat_list = groups.get(label)
            if not chat_list:
                continue
            st.caption(label)
            for chat in chat_list:
                is_active = chat.id == active_chat_id
                if st.button(
                    f"{'📄 ' if not is_active else '👉 '}{chat.title}",
                    key=f"chat_btn_{chat.id}",
                    use_container_width=True,
                    type="secondary" if not is_active else "primary",
                ):
                    set_active_chat(chat.id)
                    st.rerun()

        st.markdown("---")
        st.caption(f"👤 {user['email']}")
        if st.button("Logout", use_container_width=True):
            logout_user()
            st.rerun()
