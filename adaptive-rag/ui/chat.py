from __future__ import annotations

import streamlit as st

from auth.session import get_current_user, set_active_chat
from services.chat_service import ChatNotFoundError, ChatService
from ui.components import render_document_status_badge, render_retrieval_analysis, render_sources
from ui.upload import render_add_documents_expander

_chats = ChatService()


def _render_header(chat, user_id: str) -> None:
    docs = _chats.document_service.list_for_chat(chat.id)
    indexed = sum(1 for d in docs if d.status == "indexed")

    col_title, col_actions = st.columns([5, 1])
    with col_title:
        st.markdown(f"### {chat.title}")
        st.caption(f"📚 Knowledge base: {indexed}/{len(docs)} document(s) indexed")

    with col_actions:
        with st.popover("⋮", use_container_width=True):
            new_title = st.text_input("Rename chat", value=chat.title, key=f"rename_{chat.id}")
            if st.button("Save name", key=f"save_rename_{chat.id}", use_container_width=True):
                _chats.rename_chat(chat.id, user_id, new_title)
                st.rerun()
            st.markdown("---")
            confirm = st.checkbox("Confirm permanent deletion", key=f"confirm_delete_{chat.id}")
            if st.button(
                "🗑️ Delete chat", key=f"delete_{chat.id}", use_container_width=True, disabled=not confirm
            ):
                _chats.delete_chat(chat.id, user_id)
                set_active_chat(None)
                st.rerun()

    if docs:
        with st.expander("Documents in this chat"):
            for d in docs:
                st.markdown(f"- {d.filename} — {render_document_status_badge(d.status)}")
                if d.status == "failed" and getattr(d, "error_message", None):
                    st.caption(d.error_message)

    render_add_documents_expander(chat.id)


def _render_message_history(chat_id: str) -> None:
    messages = _chats.message_service.list_for_chat(chat_id)
    for m in messages:
        with st.chat_message("user" if m.role == "user" else "assistant"):
            st.markdown(m.content)
            if m.role == "assistant":
                render_sources(m.sources)
                render_retrieval_analysis(m.debug_info)


def render_chat_page(chat_id: str) -> None:
    user = get_current_user()
    try:
        chat = _chats.get_chat(chat_id, user["id"])
    except ChatNotFoundError:
        st.error("This chat could not be found.")
        set_active_chat(None)
        return

    _render_header(chat, user["id"])
    st.markdown("---")
    _render_message_history(chat.id)

    query = st.chat_input("Ask a question about this chat's documents…")
    if query:
        with st.chat_message("user"):
            st.markdown(query)
        with st.chat_message("assistant"):
            with st.spinner("Thinking — analyzing, retrieving, and verifying…"):
                try:
                    result = _chats.ask(chat_id=chat.id, user_id=user["id"], query=query)
                except Exception as exc:  # keep the UI alive even on a hard backend failure
                    st.error(f"Something went wrong answering that: {exc}")
                    return
            st.markdown(result.assistant_message.content)
            render_sources(result.assistant_message.sources)
            render_retrieval_analysis(result.debug_info)
        st.rerun()


def render_empty_state() -> None:
    st.markdown(
        "<div style='text-align:center;padding-top:80px;'>"
        "<h2>👋 Welcome to AdaptiveRAG</h2>"
        "<p style='color:gray;'>Create a new chat and upload some documents to get started.</p>"
        "</div>",
        unsafe_allow_html=True,
    )
