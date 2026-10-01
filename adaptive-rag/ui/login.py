from __future__ import annotations

import streamlit as st

from auth.auth_service import AuthService
from auth.session import login_user

_auth = AuthService()


def render_login_page() -> None:
    st.markdown(
        "<h1 style='text-align:center;'>🧠 AdaptiveRAG</h1>"
        "<p style='text-align:center;color:gray;'>Self-correcting retrieval for reliable answers "
        "from your own documents.</p>",
        unsafe_allow_html=True,
    )

    _, center, _ = st.columns([1, 1.2, 1])
    with center:
        tab_login, tab_signup = st.tabs(["Log In", "Sign Up"])

        with tab_login:
            _render_login_form()

        with tab_signup:
            _render_signup_form()


def _render_login_form() -> None:
    with st.form("login_form", border=True):
        email = st.text_input("Email", key="login_email")
        password = st.text_input("Password", type="password", key="login_password")
        submitted = st.form_submit_button("Log In", use_container_width=True, type="primary")

    if submitted:
        result = _auth.log_in(email, password)
        if result.ok:
            login_user(result.user)
            st.rerun()
        else:
            st.error(result.error)


def _render_signup_form() -> None:
    with st.form("signup_form", border=True):
        email = st.text_input("Email", key="signup_email")
        password = st.text_input("Password", type="password", key="signup_password")
        confirm = st.text_input("Confirm Password", type="password", key="signup_confirm")
        st.caption("At least 8 characters, with an uppercase letter, a lowercase letter, and a digit.")
        submitted = st.form_submit_button("Create Account", use_container_width=True, type="primary")

    if submitted:
        result = _auth.sign_up(email, password, confirm)
        if result.ok:
            login_user(result.user)
            st.success("Account created!")
            st.rerun()
        else:
            for err in result.errors or []:
                st.error(err)
