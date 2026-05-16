---
title: AEGIS-RAG
emoji: 🛡️
colorFrom: indigo
colorTo: blue
sdk: streamlit
sdk_version: 1.36.0
app_file: app.py
pinned: false
license: mit
---

# AEGIS-RAG — Production-Style RAG for Canadian Financial Compliance

> Reliable Retrieval-Augmented Generation over FINTRAC and OSFI regulatory guidance — with hybrid retrieval, cross-encoder reranking, **score-gated abstention**, and a rigorous evaluation harness.

[![Live Demo](https://img.shields.io/badge/🤗%20Hugging%20Face-Live%20Demo-blue)](https://huggingface.co/spaces/aminlasri/aegis-rag)

![AEGIS-RAG demo: confident answer, LLM refusal abstention, and score-gated abstention](docs/aegis-demo.gif)

*Three question types in 30 seconds: a grounded answer, an LLM-refusal abstention on out-of-scope content, and a score-gated abstention triggered by adjusting the rerank threshold.*

[![Python](https://img.shields.io/badge/Python-3.10+-blue)](https://python.org)
[![LangChain](https://img.shields.io/badge/LangChain-RAG-green)](https://langchain.com)
[![Status](https://img.shields.io/badge/Status-Complete-brightgreen)](https://github.com/217amin/aegis-rag)
[![License](https://img.shields.io/badge/License-MIT-lightgrey)](https://github.com/217amin/aegis-rag)

> A retrieval-first RAG system for policy-heavy PDF corpora. Built around the parts that matter in real deployments: retrieval quality, ranking precision, **safe abstention**, evaluation rigor, and reproducible experimentation.

---

## Results at a Glance

Evaluated on a 17-question in-scope gold set + 3-question out-of-scope abstention set, drawn from 800 chunks across 6 Canadian financial-compliance documents (FINTRAC LCTR/24-hour/EFT guidance, OSFI E-21, B-13, B-10).

### Retrieval Ablation

| Profile        | Recall@5 | nDCG@5 | MRR    | Latency  | QPS    |
|----------------|----------|--------|--------|----------|--------|
| `bm25_only`    | 0.667    | 0.558  | 0.561  | 1.9 ms   | 530    |
| `dense_only`   | 0.686    | 0.590  | 0.642  | 60.8 ms  | 16.4   |
| **`hybrid`**   | **0.686**| **0.701** | **0.794** | **63 ms** | **15.8** |
| `hybrid_rerank`| 0.657    | 0.638  | 0.716  | 81.7 ms  | 12.2   |

### Abstention Precision (out-of-scope queries)

| Profile        | Precision  | Notes |
|----------------|------------|-------|
| `hybrid_rerank`| **3/3 (100%)** | Cross-encoder yields strongly negative scores on OOD content |
| `hybrid`       | 0/3        | Fused-RRF scores stay just above the 0.005 threshold |
| `dense_only`   | N/A        | No native score-gate at this profile |
| `bm25_only`    | N/A        | No native score-gate at this profile |

### End-to-End RAG (production profile `hybrid_rerank`, judged by RAGAS)

| Metric              | Value |
|---------------------|-------|
| Faithfulness        | 0.866 |
| Answer Relevancy    | 0.960 |
| Context Precision   | 0.908 |
| Semantic Match      | 0.852 |
| Avg Latency         | 2.22 s |
| Examples            | 17    |

**Key finding — pick your profile per task, not per project:** `hybrid` wins on raw retrieval quality (best nDCG, MRR, faster), but `hybrid_rerank` is required for safe abstention because the reranker produces strongly-negative scores on out-of-scope content. The system ships `hybrid_rerank` as the production default because **trust beats top-1 ranking** in regulatory domains — but the ablation findings are documented honestly rather than hidden.

---

## Why This Project

Most RAG demos overestimate performance: small corpora, informal evaluation, no failure analysis. This project treats RAG as an **information retrieval system first** and a generation system second.

The goal: answer Canadian financial-compliance questions from FINTRAC and OSFI guidance while minimizing unsupported answers and making every performance trade-off measurable. Banking and fintech operate under hard regulatory constraints — a hallucinated answer about a $10,000 reporting threshold is worse than no answer at all. Abstention is the feature, not the failure mode.

---

## Architecture

```
PDF corpus (FINTRAC + OSFI)
→ PDF extraction and cleaning
→ semantic chunking
→ BM25 index + Chroma vector index (E5-large-v2)
→ query rewriting
→ lightweight document routing
→ profile-based retrieval
   ├── bm25_only       (lexical-only baseline)
   ├── dense_only      (semantic-only baseline)
   ├── hybrid          (best retrieval — RRF fusion)
   └── hybrid_rerank   (production default — cross-encoder + abstention gate)
→ weighted reciprocal rank fusion
→ cross-encoder reranking (ms-marco-MiniLM-L-6-v2)
→ confidence gating / abstention (two layers: score-gate + LLM refusal)
→ grounded answer generation (Llama-3.1-8B via Ollama)
→ retrieval + generation + abstention evaluation
```

---

## Core Design Choices

### 1. Semantic Chunking for Policy Documents
Groups nearby paragraphs when possible, falls back to recursive splitting for longer segments. Keeps regulatory rules and their exceptions together — critical for accurate retrieval on legalese where one clause modifies the previous.

### 2. Hybrid Retrieval
- **BM25** for exact lexical wording (regulatory documents are full of high-value exact terms: "$10,000", "PCMLTFA", "FRFI")
- **Dense retrieval (E5-large-v2)** for semantic paraphrase ("Can I delegate reporting?" → finds "responsibility cannot be transferred")
- **Weighted RRF** for candidate merging without score-calibration issues across heterogeneous retrievers

### 3. Cross-Encoder Reranking (with Honest Caveats)
Applied after initial retrieval with `ms-marco-MiniLM-L-6-v2`. On this regulatory corpus, the reranker **slightly hurts in-scope retrieval quality** (nDCG@5 drops 0.701 → 0.638) but is the only profile that produces **deeply negative scores on out-of-scope content**, enabling reliable abstention. This is a defensible trade-off, not a contradiction — see [Results at a Glance](#results-at-a-glance).

### 4. Two-Layer Abstention
- **Layer A — score-gate:** Pre-LLM. If `top_rerank_score < 0.25`, return the safe fallback string without invoking the LLM. Cheap (saves a generation call) and 100% reliable on tested out-of-scope queries.
- **Layer B — LLM grounding:** Post-retrieval. The LLM is prompted to return the same fallback string if the retrieved context doesn't support an answer. Catches cases where the score-gate passes but the question isn't actually answerable.

### 5. Query Rewriting + Lightweight Routing
Domain-specific rewrite hints map colloquial phrasings to regulatory vocabulary. Soft document-family routing biases the fusion toward likely-relevant document titles.

### 6. Evaluation-First Development
The project is instrumented for:
- Retrieval metrics (recall@k, nDCG@5, MRR) across all profiles
- Abstention precision on out-of-scope queries
- Generation metrics (RAGAS: faithfulness, answer relevancy, context precision; plus semantic match)
- Latency / throughput tracking per profile
- Failure categorization
- Experiment manifest logging (every run records the full config)

---

## Failure Analysis

Errors are grouped into actionable categories for targeted debugging:

| Category | Description | Fix Direction |
|---|---|---|
| `retrieval_miss` | Relevant doc not retrieved | Chunking / retriever improvement OR gold-set label review |
| `ranking_or_answer_miss` | Correct doc retrieved, poorly ranked | Reranker or prompt tuning |
| `unsupported_question_answered` | LLM answered without evidence | Tighten abstention thresholds |
| `over_abstention` | Correct answer rejected | Loosen confidence gate |
| `generation_hallucination` | LLM fabricated unsupported facts | Stricter prompt constraints |

On the current 17-question in-scope set: **14 correct, 3 `retrieval_miss`**. Manual inspection of the 3 misses shows the answers produced are semantically correct but reached via different chunks than the labeled gold — a known limitation of strict ID-matched recall on a domain where the same regulatory definition appears across multiple chunks.

---

## Repository Structure

```
AEGIS-RAG/
├── data/
│   ├── pdfs/           # FINTRAC + OSFI source documents
│   ├── texts/          # chunks.jsonl
│   ├── embeddings/     # Chroma + BM25 + metadata
│   └── eval/           # gold sets + results
├── notebooks/
│   ├── 01_data_ingestion_and_chunking.ipynb
│   ├── 02_hybrid_retrieval_baseline.ipynb    # retrieval + abstention A/B
│   └── 03_model_comparison_and_evaluation.ipynb
├── scripts/
│   ├── ab_test.py      # retrieval + abstention ablation runner
│   ├── build_index.py  # BM25 + Chroma indexing
│   ├── config.py       # all hyperparameters + thresholds
│   ├── eval.py         # retrieval + abstention + generation metrics
│   ├── ingest.py       # PDF extraction + chunking
│   ├── retriever.py    # profile-based retrieval
│   └── run_rag.py      # single-query inference w/ abstention gate
├── models/
├── requirements.txt
├── app.py              # Streamlit demo
└── README.md
```

---

## Quickstart

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Build chunks and indexes
python -m scripts.ingest
python -m scripts.build_index

# 3. Run retrieval + abstention ablation across all profiles
python -m scripts.ab_test

# 4. Evaluate the production profile end-to-end (retrieval + abstention + generation)
python -m scripts.eval --profile hybrid_rerank

# 5. Ask a single question
python -m scripts.run_rag --query "What is the 24-hour rule for cash transactions?" --profile hybrid_rerank
```

---

# AEGIS-RAG · Streamlit app deployment notes

## Local run

```bash
# Make sure your indexes are built first
python -m scripts.ingest
python -m scripts.build_index

# Make sure Ollama is running
ollama pull llama3.1:8b
ollama serve  # in another terminal

# Install Streamlit and run
pip install -r requirements-app.txt
streamlit run app.py
```

The app loads the same retriever and generation pipeline
(`scripts.retriever.HybridRetriever` + `scripts.run_rag.answer_question`).
The abstention threshold sliders override `NO_ANSWER_MIN_RERANK_SCORE` and
`NO_ANSWER_MIN_FUSED_SCORE` live, per-query, without touching `config.py`.

## Hugging Face Spaces deployment

Ollama is **not available** on Spaces, so generation must go to a remote
LLM endpoint. Two options:

### Hugging Face Inference API (free tier)

1. Create a HF Space with the **Streamlit** SDK.
2. Push repo to it (including `data/embeddings/chroma/` and
   `data/embeddings/bm25_index.pkl` — these are small enough for git LFS).
3. In Space → Settings → Variables and secrets, add:
   - `GEN_BACKEND` = `hf_inference`
   - `HF_TOKEN` = (a token with read access)
   - `HF_MODEL_ID` = `meta-llama/Llama-3.1-8B-Instruct` (or any chat-completion model)

---

## Stack

- **Orchestration:** LangChain
- **Vector store:** Chroma
- **Sparse retrieval:** BM25 (rank-bm25)
- **Dense embeddings:** sentence-transformers (intfloat/e5-large-v2)
- **Reranking:** cross-encoder (ms-marco-MiniLM-L-6-v2)
- **Evaluation:** RAGAS (faithfulness, answer relevancy, context precision) + custom retrieval + abstention harness
- **PDF extraction:** pdfplumber
- **LLM (local):** Llama-3.1-8B via Ollama
- **LLM (deployed):** HF Inference API

---

## Key Findings

- **Hybrid retrieval (BM25 + dense + RRF) is the right base** — neither alone matches it on nDCG or MRR
- **Reranking is not universally a win.** On this regulatory corpus, the off-the-shelf `ms-marco-MiniLM-L-6-v2` reranker hurts in-scope ranking by ~0.06 nDCG, likely because MS MARCO web-passage training distribution doesn't match regulatory legalese
- **Reranking IS necessary for abstention.** It's the only profile that produces strongly-negative scores on out-of-scope queries, enabling a reliable score-gate
- **Production choice is `hybrid_rerank`** because abstention reliability outweighs the small in-scope ranking cost in a regulatory domain
- **Eval label quality matters as much as retrieval quality.** Three "failures" on the gold set turned out to be label artifacts, not system bugs

---

## Future Work

- Swap reranker for one trained on technical/regulatory text (e.g., `BAAI/bge-reranker-base` or `mxbai-rerank-large-v1`) to see if both axes improve simultaneously
- Threshold calibration curves for abstention (currently a fixed threshold; should be tuned per profile with a held-out OOD set)
- Section-aware chunking with heading preservation
- Citation alignment in final answers
- Metadata filtering in the dense retriever path
- Larger and harder benchmark set (current 17+3 is small enough that single-query noise affects nDCG by ±0.04)
