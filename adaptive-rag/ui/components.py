"""Small, reusable Streamlit rendering helpers shared across pages."""
from __future__ import annotations

import streamlit as st

STATUS_ICONS = {
    "pending": "⏳",
    "processing": "⚙️",
    "indexed": "✅",
    "failed": "❌",
}

CLASSIFICATION_COLORS = {
    "sufficient": "🟢",
    "partial": "🟡",
    "poor": "🔴",
}


def render_sources(sources: list[dict]) -> None:
    if not sources:
        return
    with st.container():
        st.caption("Sources")
        for s in sources:
            page = f" — page {s['page_number']}" if s.get("page_number") else ""
            st.markdown(f"**[{s['index']}]** {s['filename']}{page}")


def render_retrieval_analysis(debug_info: dict) -> None:
    """The '🔍 Retrieval Analysis' expander called for in section 26 —
    shows exactly why the adaptive workflow took the path it took."""
    if not debug_info:
        return
    with st.expander("🔍 Retrieval Analysis", expanded=False):
        cols = st.columns(4)
        cols[0].metric("Iterations", debug_info.get("iterations", 0))
        cols[1].metric("LLM calls", debug_info.get("llm_call_count", 0))
        cols[2].metric("Retrieval calls", debug_info.get("retrieval_call_count", 0))
        elapsed = debug_info.get("elapsed_seconds")
        cols[3].metric("Latency", f"{elapsed}s" if elapsed is not None else "—")

        st.markdown(
            f"**Query type:** `{debug_info.get('query_type', '—')}` &nbsp;·&nbsp; "
            f"**Complexity:** `{debug_info.get('complexity', '—')}` &nbsp;·&nbsp; "
            f"**Initial strategy:** `{debug_info.get('initial_strategy', '—')}`"
        )

        retrieval_history = debug_info.get("retrieval_history", [])
        evaluation_history = debug_info.get("evaluation_history", [])
        correction_history = debug_info.get("correction_history", [])

        st.markdown("---")
        st.markdown("**Retrieval / correction timeline**")
        max_len = max(len(retrieval_history), len(evaluation_history))
        for i in range(max_len):
            retrieval = retrieval_history[i] if i < len(retrieval_history) else None
            evaluation = evaluation_history[i] if i < len(evaluation_history) else None
            if retrieval:
                st.markdown(
                    f"- **Retrieval {i + 1}** — strategy `{retrieval['strategy']}`, "
                    f"{retrieval['num_results']} chunks, {retrieval['latency_seconds']}s "
                    f"(query: _{retrieval['query_used'][:80]}_)"
                )
            if evaluation:
                icon = CLASSIFICATION_COLORS.get(evaluation["classification"], "⚪")
                st.markdown(
                    f"  - {icon} Evaluation: relevance={evaluation['relevance_score']:.2f}, "
                    f"completeness={evaluation['completeness_score']:.2f}, "
                    f"evidence={evaluation['evidence_score']:.2f}, "
                    f"**overall={evaluation['overall_score']:.2f} ({evaluation['classification']})**"
                )
                st.caption(evaluation["reasoning"])
            correction = next((c for c in correction_history if c["iteration"] == i + 1), None)
            if correction:
                st.markdown(f"  - ↪️ Correction: **{correction['action']}** — {correction['reason']}")

        verification = debug_info.get("verification_result")
        if verification:
            st.markdown("---")
            supported = "✅ Supported" if verification.get("supported") else "⚠️ Unsupported claims found"
            st.markdown(
                f"**Answer verification:** {supported} "
                f"(faithfulness={verification.get('faithfulness_score', 0):.2f})"
            )
            for claim in verification.get("unsupported_claims", []):
                st.caption(f"⚠️ {claim.get('claim')} — {claim.get('reason')}")

        if debug_info.get("gave_up"):
            st.warning("The system gave up after exhausting corrective retrieval options.")


def render_document_status_badge(status: str) -> str:
    return f"{STATUS_ICONS.get(status, '•')} {status}"
