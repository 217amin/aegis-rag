from __future__ import annotations

import json
import re
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PDF_DIR = PROJECT_ROOT / "data" / "pdfs"
TEXT_DIR = PROJECT_ROOT / "data" / "texts"
EMBED_DIR = PROJECT_ROOT / "data" / "embeddings"
EVAL_DIR = PROJECT_ROOT / "data" / "eval"
PLOTS_DIR = EVAL_DIR / "plots"
REPORTS_DIR = EVAL_DIR / "reports"

CHUNKS_FILE = TEXT_DIR / "chunks.jsonl"
CHROMA_DIR = EMBED_DIR / "chroma"
BM25_FILE = EMBED_DIR / "bm25_index.pkl"
META_FILE = EMBED_DIR / "chunks_metadata.json"

RESULTS_RETRIEVAL_FILE = EVAL_DIR / "results_retrieval.json"
RESULTS_GENERATION_FILE = EVAL_DIR / "results_generation.json"
RESULTS_SUMMARY_FILE = EVAL_DIR / "results_summary.csv"
RESULTS_FAILURES_FILE = EVAL_DIR / "results_failures.json"
RESULTS_ABLATION_FILE = EVAL_DIR / "results_ablation.json"
RESULTS_ABSTENTION_FILE = EVAL_DIR / "results_abstention.json"
EXPERIMENT_MANIFEST_FILE = EVAL_DIR / "experiment_manifest.json"

# ---------------------------------------------------------------------------
# Model names
# ---------------------------------------------------------------------------

EMBEDDING_MODEL = "intfloat/e5-large-v2"
SEMANTIC_MATCH_MODEL = "sentence-transformers/all-MiniLM-L12-v2"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
CHROMA_COLLECTION = "compliance_fraud_chunks"

LLM_MODEL = "llama3.1:8b"
JUDGE_MODEL = "llama3.1:8b"

# ---------------------------------------------------------------------------
# Chunking & Retrieval Parameters
# ---------------------------------------------------------------------------

CHUNKING_METHOD = "semantic"
CHUNK_SIZE = 900
CHUNK_OVERLAP = 100

TOP_K_BM25 = 10
TOP_K_DENSE = 10
TOP_K_FUSED = 10
TOP_K_RERANK = 5
TOP_K_FINAL = 5

RRF_K = 60
BM25_WEIGHT = 0.6
DENSE_WEIGHT = 1.0
MIN_WEIGHTED_RRF_SCORE = 0.005
NO_ANSWER_MIN_RERANK_SCORE = 0.25
NO_ANSWER_MIN_FUSED_SCORE = 0.005

ENABLE_QUERY_REWRITE = True
ENABLE_ROUTING = True
ENABLE_RERANK = True
ENABLE_NO_ANSWER_GATING = True

RETRIEVAL_PROFILES: dict[str, dict[str, Any]] = {
    "hybrid_rerank": {
        "top_k_bm25": TOP_K_BM25,
        "top_k_dense": TOP_K_DENSE,
        "top_k_fused": TOP_K_FUSED,
        "top_k_final": TOP_K_FINAL,
        "enable_reranker": True,
    },
    "hybrid": {
        "top_k_bm25": TOP_K_BM25,
        "top_k_dense": TOP_K_DENSE,
        "top_k_fused": TOP_K_FUSED,
        "top_k_final": TOP_K_FINAL,
        "enable_reranker": False,
    },
    "dense_only": {
        "top_k_bm25": 0,
        "top_k_dense": TOP_K_FINAL,
        "top_k_fused": 0,
        "top_k_final": TOP_K_FINAL,
        "enable_reranker": False,
    },
    "bm25_only": {
        "top_k_bm25": TOP_K_FINAL,
        "top_k_dense": 0,
        "top_k_fused": 0,
        "top_k_final": TOP_K_FINAL,
        "enable_reranker": False,
    },
}

# ---------------------------------------------------------------------------
# Domain Adaptation: Financial Compliance & Fraud
# ---------------------------------------------------------------------------

REWRITE_HINTS: dict[str, str] = {
    "lctr": "large cash transaction report",
    "eft": "electronic funds transfer",
    "b13": "b-13 cyber risk",
    "b-13": "b-13 cyber risk",
    "e21": "e-21 operational risk",
    "e-21": "e-21 operational risk",
    "b10": "b-10 third party risk",
    "b-10": "b-10 third party risk",
    "24 hour": "24-hour rule",
}

# 1:1 Document Routing to prevent "Friendly Fire" boosting
ROUTING_RULES: dict[str, list[str]] = {
    "fintrac_lctr": ["lctr", "large cash", "cash transaction"],
    "fintrac_24hr": ["24-hour", "24 hour", "aggregation"],
    "fintrac_eft": ["eft", "electronic funds transfer"],
    "osfi_e21": ["e21", "e-21", "operational risk", "operational resilience"],
    "osfi_b13": ["b13", "b-13", "cyber risk", "cyber security", "cyber-security"],
    "osfi_b10_third_party": ["b10", "b-10", "third party", "third-party", "vendor", "subcontractor"],
}

# Must strictly match the exact "title" metadata injected during document chunking
ROUTING_GROUP_TO_TITLES: dict[str, list[str]] = {
    "fintrac_lctr": ["01 Fintrac Lctr Guidance"],
    "fintrac_24hr": ["02 Fintrac 24hour Rule"],
    "fintrac_eft": ["03 Fintrac Eft Reporting"],
    "osfi_e21": ["04 Osfi E21 Operational Risk"],
    "osfi_b13": ["05 Osfi B13 Cyber Risk"],
    "osfi_b10_third_party": ["06 Osfi Third Party Risk"],
}

# ---------------------------------------------------------------------------
# Generation Guardrails
# ---------------------------------------------------------------------------

SAFE_FALLBACK_ANSWER = (
    "I cannot find sufficient evidence in the regulatory documents to answer this safely."
)

SYSTEM_PROMPT = (
    "You are a strict financial compliance and risk management assistant. "
    "Answer ONLY using the provided regulatory context. "
    "Use concise, factual language. Do not explain, speculate, or offer legal advice. "
    f"If the answer is not supported by the context, say exactly: '{SAFE_FALLBACK_ANSWER}'"
)

HUMAN_PROMPT = (
    "Question:\n{question}\n\n"
    "Context:\n{context}\n\n"
    "Provide a concise, factual answer based strictly on the regulations."
)

_BM25_TOKEN_RE = re.compile(r"\b\w+\b|[$%]")

def bm25_tokenize(text: str) -> list[str]:
    return _BM25_TOKEN_RE.findall(text.lower())

@dataclass(slots=True)
class ExperimentManifest:
    created_at_utc: str
    embedding_model: str
    reranker_model: str
    llm_model: str
    chunking_method: str
    chunk_size: int
    chunk_overlap: int
    routing_enabled: bool
    query_rewrite_enabled: bool
    reranker_enabled: bool
    no_answer_gating_enabled: bool

def save_experiment_manifest(path: Path = EXPERIMENT_MANIFEST_FILE) -> None:
    manifest = ExperimentManifest(
        created_at_utc=datetime.now(timezone.utc).isoformat(),
        embedding_model=EMBEDDING_MODEL,
        reranker_model=RERANKER_MODEL,
        llm_model=LLM_MODEL,
        chunking_method=CHUNKING_METHOD,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        routing_enabled=ENABLE_ROUTING,
        query_rewrite_enabled=ENABLE_QUERY_REWRITE,
        reranker_enabled=ENABLE_RERANK,
        no_answer_gating_enabled=ENABLE_NO_ANSWER_GATING,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(asdict(manifest), f, ensure_ascii=False, indent=2)