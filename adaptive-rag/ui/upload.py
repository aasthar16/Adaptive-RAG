from __future__ import annotations

import streamlit as st

from auth.session import get_current_user, set_active_chat
from services.chat_service import ChatService

_chats = ChatService()


def _read_uploads(files) -> list[tuple[str, bytes]]:
    return [(f.name, f.getvalue()) for f in files] if files else []


def _report_outcomes(outcomes) -> None:
    for outcome in outcomes:
        if outcome.ok:
            st.success(f"✅ {outcome.document.filename} — indexed ({outcome.chunk_count} chunks)")
        else:
            st.error(f"❌ {outcome.document.filename} — {outcome.error}")


@st.dialog("Create New Chat")
def new_chat_dialog() -> None:
    user = get_current_user()
    st.markdown("**Upload documents** (PDF, TXT, DOCX, MD)")
    files = st.file_uploader(
        "Upload documents",
        type=["pdf", "txt", "docx", "md"],
        accept_multiple_files=True,
        label_visibility="collapsed",
        key="new_chat_files",
    )

    default_title = ""
    if files:
        default_title = ", ".join(f.name.rsplit(".", 1)[0] for f in files[:2])
    title = st.text_input(
        "Chat name (leave blank to auto-suggest from your documents)",
        value=default_title,
        key="new_chat_title",
    )

    col_create, col_cancel = st.columns(2)
    create_clicked = col_create.button("Create Chat", type="primary", use_container_width=True)
    cancel_clicked = col_cancel.button("Cancel", use_container_width=True)

    if cancel_clicked:
        st.rerun()

    if create_clicked:
        uploads = _read_uploads(files)
        with st.spinner("Indexing documents…"):
            result = _chats.create_chat(user_id=user["id"], uploads=uploads, title=title or None)
        _report_outcomes(result.upload_outcomes)
        set_active_chat(result.chat.id)
        st.rerun()


def render_add_documents_expander(chat_id: str) -> None:
    """Section 36: allow adding more documents to an existing chat."""
    user = get_current_user()
    with st.expander("➕ Add documents to this chat"):
        files = st.file_uploader(
            "Upload additional documents",
            type=["pdf", "txt", "docx", "md"],
            accept_multiple_files=True,
            key=f"add_docs_{chat_id}",
        )
        if st.button("Upload & Index", key=f"add_docs_btn_{chat_id}"):
            uploads = _read_uploads(files)
            if not uploads:
                st.warning("Choose at least one file first.")
            else:
                with st.spinner("Indexing…"):
                    outcomes = _chats.add_documents(chat_id, user["id"], uploads)
                _report_outcomes(outcomes)
                st.rerun()
