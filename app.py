"""
AEGIS-RAG · Streamlit Demo
==========================

Interactive demo for the AEGIS-RAG hybrid retrieval + reranking + abstention pipeline.

Layout:
- Sidebar: profile selector, abstention threshold slider, query rewrite toggle,
  example questions, and pipeline info.
- Main: chat conversation. Each answer ships with a confidence indicator,
  abstention badge (when triggered), retrieved-chunks expander showing per-chunk
  scores and which retrievers contributed (BM25 / dense / rerank), and latency.

Wiring:
- Imports your existing scripts.retriever.HybridRetriever and scripts.run_rag.answer_question.
- Loads the retriever (and reranker, if profile uses one) ONCE via @st.cache_resource.
- Reloads when the user switches profile (different reranker requirements).

Generation backend (LLM):
- Default = Ollama (matches your CLI): set GEN_BACKEND=ollama (or leave unset).
- For Hugging Face Spaces / Cloud: set GEN_BACKEND=hf_inference and HF_TOKEN.
- Or use any OpenAI-compatible endpoint: set GEN_BACKEND=openai_compat,
  OPENAI_BASE_URL, OPENAI_API_KEY, OPENAI_MODEL.

Abstention threshold slider:
- Live-overrides config.NO_ANSWER_MIN_RERANK_SCORE and NO_ANSWER_MIN_FUSED_SCORE
  for the duration of each query, so the slider has immediate effect without
  editing scripts/config.py.

Run locally:
    streamlit run app.py

Deploy to HF Spaces:
    1. Push this repo (with built indexes under data/embeddings/) to a Streamlit Space.
    2. Set GEN_BACKEND=hf_inference and HF_TOKEN in Space secrets.
    3. Note: e5-large-v2 + Chroma + cross-encoder fits the 16GB free tier;
       LLM generation MUST be remote (Ollama unavailable on Spaces).
"""

from __future__ import annotations

import os
import time
import traceback
from typing import Any

import streamlit as st

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="AEGIS-RAG · Reliable RAG with Abstention",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# Imports from scripts
# ---------------------------------------------------------------------------
from scripts import config  # noqa: E402  — used for live threshold overrides
from scripts.config import SAFE_FALLBACK_ANSWER  # noqa: E402
from scripts.retriever import HybridRetriever  # noqa: E402
from scripts.run_rag import answer_question  # noqa: E402


# ---------------------------------------------------------------------------
# Constants for the UI
# ---------------------------------------------------------------------------
PROFILE_META: dict[str, dict[str, str]] = {
    "hybrid_rerank": {
        "label": "Hybrid + Rerank (best quality)",
        "description": (
            "BM25 + Dense + RRF fusion + Cross-Encoder reranking. "
            "Best nDCG@5 (0.80). ~7.6× slower than retrieval-only."
        ),
    },
    "hybrid": {
        "label": "Hybrid (BM25 + Dense)",
        "description": (
            "BM25 + Dense + RRF fusion, no reranking. "
            "Strong recall (≈0.85), low latency (~95 ms retrieval)."
        ),
    },
    "dense_only": {
        "label": "Dense only",
        "description": (
            "E5 embeddings + Chroma. Strong on paraphrase, weaker on exact wording."
        ),
    },
    "bm25_only": {
        "label": "BM25 only",
        "description": (
            "Lexical retrieval only. Strong on exact wording, weaker on paraphrase."
        ),
    },
}

EXAMPLE_QUESTIONS: list[dict[str, str]] = [
    {
        "label": "Standard policy lookup",
        "question": "What time is standard hotel check-in?",
        "hint": "Should produce a confident, grounded answer.",
    },
    {
        "label": "Specific fee question",
        "question": "How much does early check-in cost before the standard time?",
        "hint": "Should retrieve the early-arrival fee policy.",
    },
    {
        "label": "Parking authorization",
        "question": "When is parking covered by the corporate sponsor?",
        "hint": "Tests routing to the parking_transport policy group.",
    },
    {
        "label": "Abstention test (out of scope)",
        "question": "What is the company's stock price today?",
        "hint": "Should trigger the abstention pathway — not in source documents.",
    },
    {
        "label": "Abstention test (procedural ambiguity)",
        "question": "What happens if I don't have a receipt for a meal?",
        "hint": "Edge case from the failure analysis — may abstain on low rerank score.",
    },
    {
        "label": "Wi-Fi / internet",
        "question": "Which Wi-Fi network should hotel guests connect to?",
        "hint": "Tests query rewriting (wifi → guest network).",
    },
]


# ---------------------------------------------------------------------------
# Pipeline loading (cached)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading AEGIS-RAG retriever (E5 + Chroma + BM25)…")
def load_retriever(profile_name: str) -> HybridRetriever:
    """
    Load a HybridRetriever for the given profile. Cached per-profile because
    different profiles need different sub-modules loaded (e.g. reranker only
    for hybrid_rerank).
    """
    return HybridRetriever(profile_name=profile_name)


# ---------------------------------------------------------------------------
# Generation backend selection
# ---------------------------------------------------------------------------
def get_generation_kwargs() -> dict[str, Any]:
    """
    Resolve which LLM backend to use based on env vars.
    Returns kwargs to pass to answer_question; falls back to Ollama defaults
    if nothing is set (matches your CLI behaviour).
    """
    backend = os.getenv("GEN_BACKEND", "ollama").lower()

    if backend == "ollama":
        # Default path — answer_question already uses ChatOllama via run_rag.
        # Just expose model + timeout if user wants to override.
        return {
            "model_name": os.getenv("OLLAMA_MODEL", config.LLM_MODEL),
            "request_timeout": int(os.getenv("OLLAMA_TIMEOUT", "120")),
        }

    # For non-Ollama backends, we need to monkey-patch run_rag.build_answer_chain.
    # We do that lazily inside run_with_overrides() to keep this function pure.
    return {"model_name": os.getenv("REMOTE_MODEL", "remote"), "request_timeout": 120}


# ---------------------------------------------------------------------------
# Override helpers — abstention thresholds and LLM backend
# ---------------------------------------------------------------------------
class _ConfigOverride:
    """Context manager to temporarily override config thresholds for one query."""

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
        # run_rag.py imported these by name at import time, so patch there too.
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


def _maybe_install_remote_llm() -> None:
    """
    If GEN_BACKEND is hf_inference or openai_compat, monkey-patch
    run_rag.build_answer_chain to return a chain backed by that endpoint.
    Idempotent — installs once per process.
    """
    backend = os.getenv("GEN_BACKEND", "ollama").lower()
    if backend == "ollama":
        return

    import scripts.run_rag as _run_rag

    if getattr(_run_rag, "_aegis_remote_installed", False):
        return

    from langchain_core.prompts import ChatPromptTemplate
    from langchain_classic.chains.combine_documents import create_stuff_documents_chain

    if backend == "hf_inference":
        from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

        hf_token = os.getenv("HF_TOKEN")
        if not hf_token:
            st.error("GEN_BACKEND=hf_inference but HF_TOKEN env var is not set.")
            st.stop()

        model_id = os.getenv(
            "HF_MODEL_ID", "meta-llama/Llama-3.1-8B-Instruct"
        )

        def build_answer_chain(model_name=None, request_timeout: int = 120):
            llm = ChatHuggingFace(
                llm=HuggingFaceEndpoint(
                    repo_id=model_id,
                    huggingfacehub_api_token=hf_token,
                    temperature=0.0,
                    max_new_tokens=120,
                    timeout=request_timeout,
                )
            )
            prompt = ChatPromptTemplate.from_messages(
                [("system", config.SYSTEM_PROMPT), ("human", config.HUMAN_PROMPT)]
            )
            return create_stuff_documents_chain(llm, prompt)

    elif backend == "openai_compat":
        from langchain_openai import ChatOpenAI

        api_key = os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("OPENAI_BASE_URL")
        model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        if not api_key:
            st.error("GEN_BACKEND=openai_compat but OPENAI_API_KEY is not set.")
            st.stop()

        def build_answer_chain(model_name=None, request_timeout: int = 120):
            llm = ChatOpenAI(
                model=model,
                api_key=api_key,
                base_url=base_url,
                temperature=0.0,
                max_tokens=120,
                timeout=request_timeout,
            )
            prompt = ChatPromptTemplate.from_messages(
                [("system", config.SYSTEM_PROMPT), ("human", config.HUMAN_PROMPT)]
            )
            return create_stuff_documents_chain(llm, prompt)

    else:
        st.error(f"Unknown GEN_BACKEND: {backend}")
        st.stop()

    _run_rag.build_answer_chain = build_answer_chain
    _run_rag._aegis_remote_installed = True


def run_query(
    question: str,
    profile_name: str,
    rerank_threshold: float,
    fused_threshold: float,
) -> dict[str, Any]:
    """End-to-end: load retriever (cached), apply threshold overrides, run."""
    _maybe_install_remote_llm()
    retriever = load_retriever(profile_name)

    gen_kwargs = get_generation_kwargs()

    start = time.perf_counter()
    with _ConfigOverride(rerank_threshold, fused_threshold):
        result = answer_question(
            question=question,
            retriever=retriever,
            **gen_kwargs,
        )
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    result["latency_ms"] = elapsed_ms

    # Compute a normalized confidence for display.
    diag = result.get("retrieval_diagnostics", {})
    if profile_name == "hybrid_rerank":
        raw = diag.get("top_rerank_score")
        # CrossEncoder ms-marco scores are typically in roughly [-10, +10];
        # squash to [0,1] for a friendly bar.
        result["confidence"] = _squash_rerank(raw)
        result["confidence_kind"] = "rerank"
    else:
        raw = diag.get("top_weighted_rrf_score")
        # Weighted RRF scores are small positives (typical max ≈ 0.05).
        result["confidence"] = min(1.0, (raw or 0.0) / 0.05)
        result["confidence_kind"] = "fused_rrf"
    result["confidence_raw"] = raw
    result["abstained"] = result["answer"] == SAFE_FALLBACK_ANSWER

    return result


def _squash_rerank(score: float | None) -> float:
    """Sigmoid-squash a CrossEncoder logit into [0, 1] for a confidence bar."""
    if score is None:
        return 0.0
    import math

    return 1.0 / (1.0 + math.exp(-float(score)))


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
def render_header() -> None:
    st.title("🛡️ AEGIS-RAG")
    st.caption(
        "Reliable Retrieval-Augmented Generation for policy documents — with hybrid "
        "retrieval, cross-encoder reranking, and **safe abstention** when evidence is weak."
    )


def render_sidebar() -> dict[str, Any]:
    st.sidebar.header("⚙️ Pipeline configuration")

    profile_name = st.sidebar.selectbox(
        "Retrieval profile",
        options=list(PROFILE_META.keys()),
        format_func=lambda p: PROFILE_META[p]["label"],
        index=0,
        help="A/B test the four retrieval strategies. The 'hybrid_rerank' profile "
        "is the production default.",
    )
    st.sidebar.caption(PROFILE_META[profile_name]["description"])

    st.sidebar.divider()
    st.sidebar.subheader("🛡️ Abstention thresholds")
    st.sidebar.caption(
        "Below these scores, AEGIS abstains instead of generating an answer. "
        "Lower = more answers (less safe). Higher = more abstentions (safer)."
    )

    rerank_threshold = st.sidebar.slider(
        "Min rerank score (hybrid_rerank)",
        min_value=-5.0,
        max_value=5.0,
        value=float(config.NO_ANSWER_MIN_RERANK_SCORE),
        step=0.05,
        help="Cross-encoder rerank score threshold. Default from config: "
        f"{config.NO_ANSWER_MIN_RERANK_SCORE}.",
    )
    fused_threshold = st.sidebar.slider(
        "Min weighted-RRF score (hybrid)",
        min_value=0.0,
        max_value=0.05,
        value=float(config.NO_ANSWER_MIN_FUSED_SCORE),
        step=0.001,
        format="%.3f",
        help="Weighted RRF fusion score threshold. Default from config: "
        f"{config.NO_ANSWER_MIN_FUSED_SCORE}.",
    )

    st.sidebar.divider()

    with st.sidebar.expander("ℹ️ About this demo"):
        st.markdown(
            """
            **AEGIS-RAG** prioritizes retrieval quality and safe abstention over
            flashy generation.

            - **Hybrid retrieval**: BM25 (lexical) + E5 (dense) via Reciprocal Rank Fusion
            - **Cross-encoder reranking**: ms-marco-MiniLM-L-6-v2
            - **Abstention gating**: refuses to answer when evidence is weak
            - **Failure-aware**: every category of error is documented and tested

            **Best metrics (hybrid_rerank):**
            Recall@5 = 0.85 · nDCG@5 = 0.80 · Context Precision = 0.83
            """
        )

    backend = os.getenv("GEN_BACKEND", "ollama")
    st.sidebar.caption(f"🔌 LLM backend: `{backend}`")

    return {
        "profile_name": profile_name,
        "rerank_threshold": rerank_threshold,
        "fused_threshold": fused_threshold,
    }


def render_examples_panel() -> str | None:
    """Render the example-questions row. Returns the question if one was clicked."""
    st.subheader("Try an example")
    cols = st.columns(3)
    clicked: str | None = None
    for i, ex in enumerate(EXAMPLE_QUESTIONS):
        with cols[i % 3]:
            if st.button(ex["label"], use_container_width=True, key=f"ex_{i}"):
                clicked = ex["question"]
            st.caption(ex["hint"])
    return clicked


def render_confidence_bar(confidence: float, kind: str, raw: float | None) -> None:
    """Render a confidence bar with a label."""
    pct = int(round(confidence * 100))
    raw_str = f"{raw:.3f}" if raw is not None else "n/a"
    label_kind = "rerank score" if kind == "rerank" else "fused RRF score"
    st.progress(confidence, text=f"Confidence ≈ {pct}%  ·  raw {label_kind}: {raw_str}")


def render_chunk(chunk: dict[str, Any], idx: int) -> None:
    """Render one retrieved chunk with metadata + signals."""
    src = chunk.get("source") or "unknown"
    page = chunk.get("page")
    title = chunk.get("title") or ""
    score = chunk.get("score")
    retrievers = chunk.get("retrievers") or []

    page_str = f" · p.{page}" if page is not None else ""
    score_str = f"{score:.4f}" if isinstance(score, (int, float)) else str(score)
    retr_str = ", ".join(retrievers) if retrievers else "—"

    header = f"**#{idx}** · `{src}`{page_str} · score `{score_str}` · retrievers: `{retr_str}`"
    st.markdown(header)
    if title:
        st.caption(title)
    st.markdown(f"> {chunk.get('text', '').strip()}")
    st.divider()


def render_result(result: dict[str, Any]) -> None:
    """Render an answer result block."""
    abstained = result["abstained"]
    answer = result["answer"]
    diagnostics = result.get("retrieval_diagnostics", {})
    chunks = result.get("retrieved_chunks", [])
    latency_ms = result.get("latency_ms", 0.0)

    if abstained:
        st.warning(f"🛡️ **Abstained** — {answer}")
    else:
        st.success(f"💬 **Answer:** {answer}")

    cols = st.columns(3)
    with cols[0]:
        render_confidence_bar(
            result["confidence"], result["confidence_kind"], result["confidence_raw"]
        )
    with cols[1]:
        st.metric("Profile", result.get("profile_name", "—"))
    with cols[2]:
        st.metric("Latency", f"{latency_ms:.0f} ms")

    rewritten = diagnostics.get("rewritten_query")
    routed = diagnostics.get("routed_group")
    if (rewritten and rewritten != diagnostics.get("original_query")) or routed:
        info_bits = []
        if rewritten and rewritten != diagnostics.get("original_query"):
            info_bits.append(f"**Rewritten query:** _{rewritten}_")
        if routed:
            info_bits.append(f"**Routed to group:** `{routed}`")
        st.info(" · ".join(info_bits))

    with st.expander(f"📚 Retrieved evidence ({len(chunks)} chunks)", expanded=False):
        if not chunks:
            st.write("No chunks retrieved.")
        else:
            for i, ch in enumerate(chunks, start=1):
                render_chunk(ch, i)

    with st.expander("🔬 Raw diagnostics (for debugging)", expanded=False):
        st.json(diagnostics)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    render_header()
    settings = render_sidebar()

    # Session state for chat history
    if "history" not in st.session_state:
        st.session_state.history = []  # list[tuple[str, dict]]

    # Examples
    example_q = render_examples_panel()

    st.divider()

    # Render history
    for q, res in st.session_state.history:
        with st.chat_message("user"):
            st.write(q)
        with st.chat_message("assistant"):
            render_result(res)

    # Input
    user_q = st.chat_input("Ask a question about the policy documents…")

    question = example_q or user_q
    if not question:
        return

    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving + reranking + generating…"):
            try:
                result = run_query(
                    question=question,
                    profile_name=settings["profile_name"],
                    rerank_threshold=settings["rerank_threshold"],
                    fused_threshold=settings["fused_threshold"],
                )
            except Exception as exc:  # noqa: BLE001 — surface to UI
                st.error(f"Pipeline error: {exc}")
                with st.expander("Stack trace"):
                    st.code(traceback.format_exc())
                return

        render_result(result)

    st.session_state.history.append((question, result))


if __name__ == "__main__":
    main()
