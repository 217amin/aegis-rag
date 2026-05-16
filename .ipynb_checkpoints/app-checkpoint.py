"""
AEGIS-RAG · Regulatory Compliance Demo
======================================
Interactive demo for the AEGIS-RAG hybrid retrieval + reranking pipeline.
Adapted for OSFI and FINTRAC financial compliance guidelines.
"""

from __future__ import annotations

import os
import time
import traceback
from typing import Any

import streamlit as st

st.set_page_config(
    page_title="AEGIS-RAG · Compliance Engine",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

from scripts import config
from scripts.config import SAFE_FALLBACK_ANSWER
from scripts.retriever import load_retrieval_assets
from scripts.run_rag import answer_question

PROFILE_META: dict[str, dict[str, str]] = {
    "hybrid_rerank": {
        "label": "🥇 Hybrid + Rerank (Production)",
        "description": "BM25 + Dense + RRF fusion + Cross-Encoder reranking.",
    },
    "hybrid": {
        "label": "⚡ Hybrid (BM25 + Dense)",
        "description": "BM25 + Dense + RRF fusion, no reranking.",
    },
    "dense_only": {
        "label": "🧠 Dense only",
        "description": "E5 embeddings + Chroma.",
    },
    "bm25_only": {
        "label": "🔤 BM25 only",
        "description": "Lexical retrieval only.",
    },
}

EXAMPLE_QUESTIONS: list[dict[str, str]] = [
    {
        "label": "✅ Transaction Rules",
        "question": "What is the 24-hour rule for large cash transactions?",
        "hint": "Retrieves FINTRAC aggregation rules.",
    },
    {
        "label": "🛡️ Incident Reporting",
        "question": "Under OSFI B13, what is the timeline to report a major cyber incident?",
        "hint": "Tests precise timeline extraction from Cyber Risk guidelines.",
    },
    {
        "label": "🤝 Third-Party Risk",
        "question": "What are the expectations for managing sub-contractor risks?",
        "hint": "Tests routing to OSFI Third-Party Risk management.",
    },
    {
        "label": "🚫 Abstention Test",
        "question": "What are the tax implications of offshore holding accounts?",
        "hint": "Should trigger abstention — out of scope of OSFI/FINTRAC docs.",
    },
]

@st.cache_resource(show_spinner="Loading AEGIS-RAG regulatory indices…")
def get_assets():
    """Loads the functional retrieval assets (Chroma, BM25, CrossEncoder) once into memory."""
    return load_retrieval_assets()

class _ConfigOverride:
    def __init__(self, rerank_thresh: float, fused_thresh: float):
        self._rerank_thresh = rerank_thresh
        self._fused_thresh = fused_thresh
        self._original: dict[str, float] = {}

    def __enter__(self) -> "_ConfigOverride":
        self._original = {
            "rerank": config.NO_ANSWER_MIN_RERANK_SCORE,
            "fused": config.NO_ANSWER_MIN_FUSED_SCORE,
        }
        config.NO_ANSWER_MIN_RERANK_SCORE = self._rerank_thresh
        config.NO_ANSWER_MIN_FUSED_SCORE = self._fused_thresh
        import scripts.run_rag as _run_rag
        _run_rag.NO_ANSWER_MIN_RERANK_SCORE = self._rerank_thresh
        _run_rag.NO_ANSWER_MIN_FUSED_SCORE = self._fused_thresh
        return self

    def __exit__(self, *args: Any) -> None:
        config.NO_ANSWER_MIN_RERANK_SCORE = self._original["rerank"]
        config.NO_ANSWER_MIN_FUSED_SCORE = self._original["fused"]
        import scripts.run_rag as _run_rag
        _run_rag.NO_ANSWER_MIN_RERANK_SCORE = self._original["rerank"]
        _run_rag.NO_ANSWER_MIN_FUSED_SCORE = self._original["fused"]

def run_query(question: str, profile_name: str, rerank_threshold: float, fused_threshold: float) -> dict[str, Any]:
    assets = get_assets()
    start = time.perf_counter()
    
    # Thread-safe pipeline execution
    result = answer_question(
        question=question, 
        assets=assets, 
        profile_name=profile_name,
        rerank_threshold=rerank_threshold,
        fused_threshold=fused_threshold
    )
    
    result["latency_ms"] = (time.perf_counter() - start) * 1000.0
    
    # Process Confidence metrics
    if not result.get("retrieved_chunks"):
        result["confidence_raw"] = 0.0
        result["confidence"] = 0.0
        result["confidence_kind"] = "None"
    else:
        top_chunk = result["retrieved_chunks"][0]
        if profile_name == "hybrid_rerank":
            raw = top_chunk.get("rerank_score", 0.0)
            import math
            result["confidence"] = 1.0 / (1.0 + math.exp(-float(raw)))
            result["confidence_kind"] = "rerank"
        else:
            raw = top_chunk.get("weighted_rrf_score", 0.0)
            result["confidence"] = min(1.0, float(raw) / 0.05)
            result["confidence_kind"] = "fused_rrf"
            
        result["confidence_raw"] = raw

    result["abstained"] = result["answer"] == SAFE_FALLBACK_ANSWER
    return result

def render_sidebar() -> dict[str, Any]:
    st.sidebar.header("⚙️ Configuration")
    profile_name = st.sidebar.selectbox(
        "Retrieval profile",
        options=list(PROFILE_META.keys()),
        format_func=lambda p: PROFILE_META[p]["label"],
        index=0,
    )
    st.sidebar.divider()
    st.sidebar.subheader("🛡️ Abstention thresholds")
    rerank_threshold = st.sidebar.slider(
        "Min rerank score", min_value=-5.0, max_value=5.0,
        value=float(config.NO_ANSWER_MIN_RERANK_SCORE), step=0.05,
    )
    fused_threshold = st.sidebar.slider(
        "Min weighted-RRF score", min_value=0.0, max_value=0.05,
        value=float(config.NO_ANSWER_MIN_FUSED_SCORE), step=0.001, format="%.3f",
    )
    return {
        "profile_name": profile_name,
        "rerank_threshold": rerank_threshold,
        "fused_threshold": fused_threshold,
    }

def main() -> None:
    st.title("⚖️ AEGIS-RAG: Regulatory Compliance")
    st.caption("Financial compliance engine enforcing OSFI and FINTRAC guidelines.")
    settings = render_sidebar()

    if "history" not in st.session_state:
        st.session_state.history = []

    st.subheader("Try an example")
    cols = st.columns(len(EXAMPLE_QUESTIONS))
    clicked: str | None = None
    for i, ex in enumerate(EXAMPLE_QUESTIONS):
        with cols[i]:
            if st.button(ex["label"], use_container_width=True):
                clicked = ex["question"]
            st.caption(ex["hint"])
            
    st.divider()

    for q, res in st.session_state.history:
        with st.chat_message("user"):
            st.write(q)
        with st.chat_message("assistant"):
            if res["abstained"]:
                st.warning(f"🛡️ **Abstained** — {res['answer']}")
            else:
                st.success(f"💬 **Answer:** {res['answer']}")

    user_q = st.chat_input("Query financial compliance frameworks...")
    question = clicked or user_q

    if not question:
        return

    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Processing regulatory frameworks..."):
            try:
                result = run_query(
                    question=question,
                    profile_name=settings["profile_name"],
                    rerank_threshold=settings["rerank_threshold"],
                    fused_threshold=settings["fused_threshold"],
                )
            except Exception as exc:
                st.error(f"Pipeline error: {exc}")
                st.code(traceback.format_exc())
                return

        if result["abstained"]:
            st.warning(f"🛡️ **Abstained** — {result['answer']}")
        else:
            st.success(f"💬 **Answer:** {result['answer']}")
            
        cols = st.columns(3)
        cols[0].metric(f"Top {result['confidence_kind']} score", f"{result.get('confidence_raw', 0):.3f}")
        cols[1].metric("Profile", settings["profile_name"])
        cols[2].metric("Latency", f"{result['latency_ms']:.0f} ms")
        
        with st.expander(f"📚 Retrieved Regulatory Evidence ({len(result['retrieved_chunks'])} chunks)"):
            for i, ch in enumerate(result["retrieved_chunks"], start=1):
                st.markdown(f"**#{i}** · `{ch['source']}` · score `{ch['score']:.4f}`")
                st.markdown(f"> {ch['text'].strip()}")
                st.divider()

    st.session_state.history.append((question, result))

if __name__ == "__main__":
    main()