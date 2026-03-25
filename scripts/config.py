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
EXPERIMENT_MANIFEST_FILE = EVAL_DIR / "experiment_manifest.json"

# ---------------------------------------------------------------------------
# Model names
# ---------------------------------------------------------------------------

EMBEDDING_MODEL = "intfloat/e5-large-v2"
SEMANTIC_MATCH_MODEL = "sentence-transformers/all-MiniLM-L12-v2"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
CHROMA_COLLECTION = "hotel_chunks"

LLM_MODEL = "llama3.1:8b"
JUDGE_MODEL = "llama3.1:8b"

# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

CHUNKING_METHOD = "semantic"   # "recursive" | "semantic"
CHUNK_SIZE = 900
CHUNK_OVERLAP = 100

# ---------------------------------------------------------------------------
# Retrieval knobs
# ---------------------------------------------------------------------------

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

REWRITE_HINTS: dict[str, str] = {
    "come early": "early arrival noon check-in fee availability",
    "arrive early": "early arrival noon check-in fee availability",
    "leave later": "late checkout 2 pm extension fee",
    "check out late": "late checkout 2 pm extension fee",
    "parking included": "parking covered authorization sponsor corporate account",
    "wifi": "internet guest network browser sign-in",
    "wi-fi": "internet guest network browser sign-in",
    "smoke": "smoking vaping balcony room charge",
}

ROUTING_RULES: dict[str, list[str]] = {
    "accessibility": ["access", "accessible", "wheelchair", "service animal", "medication", "safety", "comfort"],
    "parking_transport": ["parking", "valet", "garage", "vehicle", "mileage", "fuel", "transport", "shuttle", "ev charging"],
    "reimbursement": ["receipt", "reimbursement", "folio", "expense", "itemized", "supporting documentation", "business meal"],
    "billing_workflow": ["chargeback", "department", "approval", "covered", "personal charges", "workflow", "finance reviewer"],
    "events": ["conference", "event", "banquet", "hosted dining", "meeting room", "attendee", "proposal"],
    "guest_stay": ["check-in", "checkout", "quiet hours", "smoking", "guest", "internet", "luggage", "balcony", "spa"],
}

ROUTING_GROUP_TO_TITLES: dict[str, list[str]] = {
    "accessibility": [
        "05 Guest Safety Accessibility And Reasonable Comfort Standard",
        "08 Corporate Stay Faq For Sponsored Guests",
        "02 Hosted Travel And Accommodation Policy",
    ],
    "parking_transport": [
        "03 Ground Transportation Parking And Personal Vehicle Policy",
        "01 Guest Stay And Incidentals Handbook",
        "08 Corporate Stay Faq For Sponsored Guests",
        "02 Hosted Travel And Accommodation Policy",
    ],
    "reimbursement": [
        "04 Receipts Reimbursements And Supporting Documentation Standard",
        "02 Hosted Travel And Accommodation Policy",
        "07 Department Approval And Chargeback Workflow",
    ],
    "billing_workflow": [
        "07 Department Approval And Chargeback Workflow",
        "02 Hosted Travel And Accommodation Policy",
        "06 Conference Hosted Dining And Event Billing Policy",
    ],
    "events": [
        "06 Conference Hosted Dining And Event Billing Policy",
        "07 Department Approval And Chargeback Workflow",
        "02 Hosted Travel And Accommodation Policy",
    ],
    "guest_stay": [
        "01 Guest Stay And Incidentals Handbook",
        "08 Corporate Stay Faq For Sponsored Guests",
        "05 Guest Safety Accessibility And Reasonable Comfort Standard",
        "02 Hosted Travel And Accommodation Policy",
    ],
}

RETRIEVAL_PROFILES: dict[str, dict[str, Any]] = {
    "bm25_only": {
        "bm25_weight": 1.0,
        "dense_weight": 0.0,
        "enable_reranker": False,
        "top_k_bm25": 10,
        "top_k_dense": 0,
        "top_k_fused": 10,
        "top_k_final": 10,
    },
    "dense_only": {
        "bm25_weight": 0.0,
        "dense_weight": 1.0,
        "enable_reranker": False,
        "top_k_bm25": 0,
        "top_k_dense": 10,
        "top_k_fused": 10,
        "top_k_final": 10,
    },
    "hybrid": {
        "bm25_weight": BM25_WEIGHT,
        "dense_weight": DENSE_WEIGHT,
        "enable_reranker": False,
        "top_k_bm25": TOP_K_BM25,
        "top_k_dense": TOP_K_DENSE,
        "top_k_fused": TOP_K_FUSED,
        "top_k_final": TOP_K_FINAL,
    },
    "hybrid_rerank": {
        "bm25_weight": BM25_WEIGHT,
        "dense_weight": DENSE_WEIGHT,
        "enable_reranker": True,
        "top_k_bm25": TOP_K_BM25,
        "top_k_dense": TOP_K_DENSE,
        "top_k_fused": TOP_K_FUSED,
        "top_k_rerank": TOP_K_RERANK,
        "top_k_final": TOP_K_FINAL,
    },
}

# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

SAFE_FALLBACK_ANSWER = (
    "I cannot find sufficient evidence in the source documents to answer this safely."
)

SYSTEM_PROMPT = (
    "You are a strict hotel policy extraction assistant. "
    "Answer ONLY using the provided context. "
    "Use one short factual sentence. "
    "Do not explain, speculate, or add extra information. "
    f"If the answer is not supported by the context, say exactly: '{SAFE_FALLBACK_ANSWER}'"
)

HUMAN_PROMPT = (
    "Question:\n{question}\n\n"
    "Context:\n{context}\n\n"
    "Provide one short factual sentence."
)

# ---------------------------------------------------------------------------
# Canonical BM25 tokeniser — used by both build_index and retriever
# ---------------------------------------------------------------------------

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
    retrieval_profiles: dict[str, dict[str, Any]]
    routing_enabled: bool
    query_rewrite_enabled: bool
    reranker_enabled: bool
    no_answer_gating_enabled: bool


def build_experiment_manifest() -> dict[str, Any]:
    manifest = ExperimentManifest(
        created_at_utc=datetime.now(timezone.utc).isoformat(),
        embedding_model=EMBEDDING_MODEL,
        reranker_model=RERANKER_MODEL,
        llm_model=LLM_MODEL,
        chunking_method=CHUNKING_METHOD,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        retrieval_profiles=RETRIEVAL_PROFILES,
        routing_enabled=ENABLE_ROUTING,
        query_rewrite_enabled=ENABLE_QUERY_REWRITE,
        reranker_enabled=ENABLE_RERANK,
        no_answer_gating_enabled=ENABLE_NO_ANSWER_GATING,
    )
    return asdict(manifest)


def save_experiment_manifest(path: Path = EXPERIMENT_MANIFEST_FILE) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(build_experiment_manifest(), f, ensure_ascii=False, indent=2)
