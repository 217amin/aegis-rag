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

# AEGIS-RAG — Production-Style RAG with Hybrid Retrieval, Reranking & Safe Abstention

[![Python](https://img.shields.io/badge/Python-3.10+-blue)](https://python.org)
[![LangChain](https://img.shields.io/badge/LangChain-RAG-green)](https://langchain.com)
[![Status](https://img.shields.io/badge/Status-Complete-brightgreen)](https://github.com/217amin/aegis-rag)
[![License](https://img.shields.io/badge/License-MIT-lightgrey)](https://github.com/217amin/aegis-rag)

> A retrieval-first RAG system for policy-heavy PDF corpora. Built around the parts that matter in real deployments: retrieval quality, ranking precision, safe abstention, evaluation rigor, and reproducible experimentation.

---

## Results at a Glance

### Retrieval (Hybrid + RRF)

| Metric | Value |
|---|---|
| Recall@5 | **0.85** |
| nDCG@5 | **0.80** |
| MRR | **0.83** |
| Avg Latency | 95 ms |
| QPS | 10.52 |

### End-to-End RAG (Hybrid + Rerank)

| Metric | Value |
|---|---|
| Faithfulness | 0.67 |
| Answer Relevancy | 0.67 |
| Context Precision | **0.83** |
| Semantic Match | **0.75** |
| Avg Latency | 0.73 sec |
| QPS | 1.37 |

**Key takeaway:** Retrieval and ranking improvements (not fine-tuning) drove the largest quality gains. Reranking adds +8 nDCG@5 points at a 7.6× latency cost — a measurable, defensible trade-off.

---

## Why This Project

Most RAG demos overestimate performance: small corpora, informal evaluation, no failure analysis. This project treats RAG as an **information retrieval system first** and a generation system second.

The goal: answer policy questions from hotel-style operational documents while minimizing unsupported answers and making every performance trade-off measurable.

---

## Architecture

```
PDF corpus
→ PDF extraction and cleaning
→ semantic chunking
→ BM25 index + Chroma vector index
→ query rewriting
→ lightweight document routing
→ profile-based retrieval
   ├── bm25_only
   ├── dense_only
   ├── hybrid
   └── hybrid_rerank   ← best config
→ weighted reciprocal rank fusion
→ cross-encoder reranking
→ confidence gating / abstention
→ grounded answer generation
→ retrieval + generation + failure analysis
```

---

## Core Design Choices

### 1. Semantic Chunking for Policy Documents
Groups nearby paragraphs when possible, falls back to recursive splitting for longer segments. Keeps policy rules and exceptions together — critical for accurate retrieval.

### 2. Hybrid Retrieval
- **BM25** for exact wording and lexical precision
- **Dense retrieval (E5 embeddings)** for semantic recall
- **Weighted RRF** for candidate merging without score calibration issues

### 3. Cross-Encoder Reranking
Applied after initial retrieval to reorder the candidate pool by true query-document relevance. nDCG@5 improves from 0.72 → 0.80.

### 4. Query Rewriting + Routing
Lightweight rewrite hints and soft document-family routing help a small corpus behave like a practical enterprise knowledge base.

### 5. Safe No-Answer Behavior
Abstention thresholds based on rerank and fusion scores. When evidence is weak, the system returns a fixed fallback instead of forcing the LLM to improvise.

### 6. Evaluation-First Development
The project is instrumented for retrieval metrics, generation metrics, latency/throughput tracking, failure categorization, and experiment manifest logging.

---

## Retrieval Profile Comparison

| Profile | nDCG@5 | QPS | Use Case |
|---|---|---|---|
| bm25_only | 0.58 | 42.1 | Exact match / low latency |
| dense_only | 0.65 | 18.3 | Semantic / paraphrase queries |
| hybrid | 0.72 | 10.5 | Balanced production baseline |
| hybrid_rerank | **0.80** | 1.37 | Best quality, latency-tolerant |

---

## Failure Analysis

Errors are grouped into actionable categories for targeted debugging:

| Category | Description | Fix Direction |
|---|---|---|
| `retrieval_miss` | Relevant doc not retrieved | Chunking / retriever improvement |
| `ranking_or_answer_miss` | Correct doc retrieved, poorly ranked | Reranker or prompt tuning |
| `unsupported_question_answered` | LLM answered without evidence | Tighten abstention thresholds |
| `over_abstention` | Correct answer rejected | Loosen confidence gate |
| `generation_hallucination` | LLM fabricated unsupported facts | Stricter prompt constraints |

### Example Failures

**Failure 1 — Over-conservative abstention**
- Q: *"Can I check in at noon for free?"*
- Expected: *"No — early check-in costs $35"*
- Model returned: *"I cannot find sufficient evidence…"*
- Cause: Rerank score fell below threshold despite correct retrieval (Recall = 1.0)

**Failure 2 — Ranking miss despite correct retrieval**
- Q: *"What happens if I don't have a receipt?"*
- Model returned: fallback response
- Cause: Cross-encoder underestimated procedural / indirect answers

---

## Repository Structure

```
AEGIS-RAG/
├── data/
│   ├── pdfs/
│   ├── texts/
│   ├── embeddings/
│   └── eval/
├── notebooks/
│   ├── 01_data_ingestion_and_chunking.ipynb
│   ├── 02_hybrid_retrieval_baseline.ipynb
│   └── 03_model_comparison_and_evaluation.ipynb
├── scripts/
│   ├── ab_test.py       # profile A/B testing
│   ├── build_index.py   # BM25 + Chroma indexing
│   ├── config.py        # all hyperparameters
│   ├── eval.py          # retrieval + generation metrics
│   ├── ingest.py        # PDF extraction + chunking
│   ├── retriever.py     # profile-based retrieval
│   └── run_rag.py       # single-query inference
├── models/
├── requirements.txt
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

# 3. Run retrieval ablations across all profiles
python -m scripts.ab_test

# 4. Evaluate the strongest profile end-to-end
python -m scripts.eval --profile hybrid_rerank

# 5. Ask a single question
python -m scripts.run_rag --query "What time is standard check-in?" --profile hybrid_rerank
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

### Option A — Hugging Face Inference API (free tier OK)

1. Create a HF Space with the **Streamlit** SDK.
2. Push repo to it (including `data/embeddings/chroma/` and
   `data/embeddings/bm25_index.pkl` — these are small enough for git LFS).
3. In Space → Settings → Variables and secrets, add:
   - `GEN_BACKEND` = `hf_inference`
   - `HF_TOKEN` = (a token with read access)
   - `HF_MODEL_ID` = `meta-llama/Llama-3.1-8B-Instruct` (or any chat-completion model)

### Option B — OpenAI-compatible endpoint (OpenAI, Together, Groq, etc.)

1. Same Space setup.
2. Add secrets:
   - `GEN_BACKEND` = `openai_compat`
   - `OPENAI_API_KEY` = your key
   - `OPENAI_BASE_URL` = `https://api.openai.com/v1` (or Together / Groq URL)
   - `OPENAI_MODEL` = `gpt-4o-mini` (or `meta-llama/Llama-3.3-70B-Instruct-Turbo` for Together)

The app auto-detects the backend at startup. If `GEN_BACKEND` is unset, it
defaults to Ollama (i.e. the local-dev path).

## Memory budget on HF Spaces free tier (16 GB)

Rough RAM at startup with `hybrid_rerank` profile:

| Component | RAM |
|---|---|
| Streamlit + Python | ~200 MB |
| E5-large-v2 embedding model | ~1.3 GB |
| Chroma vector store | ~100 MB (depends on corpus) |
| BM25 pickled index | ~50 MB |
| Cross-encoder ms-marco-MiniLM-L-6-v2 | ~90 MB |
| **Total** | **~1.8 GB** |

Plenty of headroom. Generation is remote, so no LLM weights loaded locally.

## Demo GIF for README

After deploying, record a 30-second walkthrough showing:

1. Click an example "Standard policy lookup" → confident answer with citations
2. Click the "Abstention test (out of scope)" example → 🛡️ **Abstained** banner
3. Lower the rerank threshold slider → ask the same question → answer now generated
4. Open the "Retrieved evidence" expander → show per-chunk scores and retrievers

Tools: [Kap](https://getkap.co) (macOS) or [ScreenToGif](https://www.screentogif.com) (Windows).
Embed at the top of `README.md` under the title.

---

## Stack

- **Orchestration:** LangChain
- **Vector store:** Chroma
- **Sparse retrieval:** BM25
- **Dense embeddings:** sentence-transformers (E5)
- **Reranking:** cross-encoder (mxbai-rerank-base-v1)
- **Evaluation:** RAGAS (faithfulness, answer relevancy, context precision)
- **PDF extraction:** pdfplumber

---

## Key Takeaways

- Retrieval improvements outperform fine-tuning on small, structured corpora
- Hybrid retrieval (BM25 + dense + RRF) is the right production baseline — neither alone is sufficient
- Reranking is worth it when accuracy matters more than throughput
- Abstention is critical for trust — but threshold calibration is non-trivial

---

## Future Work

- Threshold calibration curves for abstention
- Section-aware chunking with heading preservation
- Citation alignment in final answers
- Metadata filtering in the dense retriever path
- Larger and harder benchmark set
