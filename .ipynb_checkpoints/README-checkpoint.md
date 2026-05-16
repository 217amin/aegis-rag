# AEGIS-RAG  
Reliable RAG with Hybrid Retrieval, Reranking, and Safe Abstention

AEGIS-RAG is a production-style Retrieval-Augmented Generation system for policy-heavy PDF corpora. The project is built around the parts of RAG that matter most in real deployments: retrieval quality, ranking precision, safe abstention, evaluation rigor, and reproducible experimentation.

The current version is intentionally focused on retrieval and evaluation rather than fine-tuning. In this project, the largest gains came from better chunking, hybrid retrieval, reranking, routing, and diagnostics rather than from adapting the language model itself.

**Best configuration:** `hybrid_rerank`  
- Recall@5: 0.85  
- nDCG@5: 0.80  
- Semantic Match: 0.75  
- Latency: 7.6× higher than retrieval-only baseline

## Problem statement

Many RAG demos overestimate performance due to small corpora and informal evaluation. This project treats RAG as an information retrieval system first and a generation system second. The goal is to answer policy questions from hotel-style operational documents while minimizing unsupported answers and making performance trade-offs measurable.

## Architecture

```text
PDF corpus
→ PDF extraction and cleaning
→ semantic chunking
→ BM25 index + Chroma vector index
→ query rewriting
→ lightweight document routing
→ profile-based retrieval
   - bm25_only
   - dense_only
   - hybrid
   - hybrid_rerank
→ weighted reciprocal rank fusion
→ cross-encoder reranking
→ confidence gating / abstention
→ grounded answer generation
→ retrieval evaluation + generation evaluation + failure analysis
```

## Pipeline Overview

```text
Query
 ↓
Rewrite → Router
 ↓
BM25 + Dense
 ↓
RRF Fusion
 ↓
Reranker
 ↓
Threshold
 ↓
Answer / No-answer
```
This pipeline emphasizes a separation between retrieval, ranking, and generation, allowing each component to be evaluated and optimized independently.

## Core design choices

### 1. Semantic chunking for policy documents
The corpus is chunked with a semantic-first strategy that groups nearby paragraphs when possible and falls back to recursive splitting for longer segments. This keeps policy rules and exceptions together without creating overly large contexts.

### 2. Hybrid retrieval
The retrieval system combines:
- BM25 for exact wording and lexical precision
- dense retrieval with E5 embeddings for semantic recall
- weighted reciprocal rank fusion for candidate merging

This matters because policy questions often depend on both exact wording and paraphrased intent.

### 3. Cross-encoder reranking
A cross-encoder reranker is applied to candidate chunks in the strongest retrieval profile. This improves ranking quality after recall has already been established by the initial retrievers.

### 4. Query rewriting and routing
The pipeline improves vague questions using lightweight rewrite hints and soft document-family routing. These two additions help a small corpus behave more like a practical enterprise knowledge base.

### 5. Safe no-answer behavior
The project includes abstention thresholds based on rerank and fusion scores. When evidence is weak, the system returns a fixed fallback instead of forcing the language model to improvise.

### 6. Evaluation-first development
The project is instrumented for:
- retrieval metrics
- generation metrics
- latency / throughput tracking
- failure categorization
- experiment manifest logging

This makes the project easy to debug, compare, and discuss in a professional setting.

## Retrieval profiles

The codebase supports the following retrieval profiles:

- `bm25_only`
- `dense_only`
- `hybrid`
- `hybrid_rerank`

These profiles allow clean A/B testing between lexical retrieval, semantic retrieval, hybrid fusion, and reranking.

## Results analysis

The main retrieval pattern observed in this project is consistent with modern search systems:

- BM25 remains competitive on exact policy wording.
- Dense retrieval improves paraphrase handling.
- Hybrid retrieval improves recall by combining both signals.
- Reranking improves nDCG and MRR by reordering candidate evidence more precisely.

A representative outcome from the project showed:

- `hybrid_rerank` achieved the best overall retrieval quality.
- `hybrid` improved recall over single-retriever baselines but still benefited from reranking.
- `dense_only` and `bm25_only` remained useful baselines for latency-quality comparisons.

This is the expected pattern in a policy corpus where exact wording and semantic similarity are both important.
This confirms that improving retrieval and ranking yields larger gains than model adaptation in small, structured corpora.

### What the metrics mean here

- **Recall@5** measures whether the system retrieves relevant chunks early.
- **nDCG@5** measures ranking quality, not just retrieval coverage.
- **MRR** captures how early the first relevant chunk appears.
- **QPS / average latency** quantify the runtime cost of each profile.
- **Faithfulness / answer relevancy / context precision** help separate retrieval problems from generation problems.

---

## Results & Overall Performance

### Retrieval Performance (Hybrid + RRF)

| Metric      | Value |
| ----------- | ----- |
| Recall@5    | 0.85  |
| nDCG@5      | 0.80  |
| MRR         | 0.83  |
| QPS         | 10.52 |
| Avg Latency | 95 ms |

The hybrid retriever achieves high recall (85%), ensuring relevant documents are consistently retrieved. Ranking quality is strong (nDCG@5 = 0.80), while maintaining low latency, making it suitable for real-time applications.

---

### End-to-End RAG Performance (Hybrid + Rerank)

| Metric            | Value    |
| ----------------- | -------- |
| Faithfulness      | 0.67     |
| Answer Relevancy  | 0.67     |
| Context Precision | 0.83     |
| Exact Match       | 0.25     |
| Semantic Match    | 0.75     |
| QPS               | 1.37     |
| Avg Latency       | 0.73 sec |

Retrieval quality translates into high context precision (0.83) and solid semantic correctness (0.75). However, cross-encoder reranking introduces a significant latency cost, reducing throughput from 10.5 → 1.37 QPS.

---

## Key Observations

- Retrieval is strong: relevant evidence is consistently retrieved (Recall@5 = 0.85)
- Reranking improves accuracy but is the main latency bottleneck
- Abstention prevents hallucinations, but is sometimes overly conservative

| Profile        | nDCG@5 | QPS |
|----------------|--------|-----|
| hybrid         | 0.72   | 10.5 |
| hybrid_rerank  | 0.80   | 1.37 |

---

## Failure analysis findings

The project groups errors into practical categories:

- `retrieval_miss`
- `generation_hallucination`
- `ranking_or_answer_miss`
- `unsupported_question_answered`
- `correct_no_answer`
- `correct`

This allows targeted debugging. For example:
- high `retrieval_miss` means the retriever or chunking strategy needs work
- high `ranking_or_answer_miss` suggests reranking or prompt constraints need improvement
- high `unsupported_question_answered` means abstention thresholds may need recalibration

### Failure 1 — Over-conservative abstention

**Question:** Can I check in at noon for free?  
**Expected:** No — early check-in costs $35  
**Model:** “I cannot find sufficient evidence…”

**Diagnosis**
- Relevant documents retrieved (Recall = 1.0)
- Rerank score below threshold → answer rejected

**Root cause**
- Overly strict confidence threshold  
- Difficulty handling negation and policy nuance  

---

### Failure 2 — Ranking / Answer miss despite correct retrieval

**Question:** What happens if I don’t have a receipt?  
**Expected:** Provide explanation + secondary evidence  
**Model:** fallback response  

**Diagnosis**
- Correct document retrieved  
- Reranker assigns low score → discarded  

**Root cause**
- Cross-encoder underestimates procedural / indirect answers  

---

### Failure 3 — Known-answer missed due to low confidence

**Question:** What internet network should I use?  
**Expected:** SweetSpot_Guest  
**Model:** fallback response  

**Diagnosis**
- Relevant chunk retrieved  
- Very low rerank score  

**Root cause**
- Weak alignment for exact entities (SSID mismatch)

---

## Summary of Failure Modes

| Category           | Description                                        |
| ------------------ | -------------------------------------------------- |
| Over-abstention    | Correct answers rejected due to strict thresholds  |
| Ranking errors     | Relevant docs retrieved but poorly scored          |
| Semantic mismatch  | Queries phrased differently than documents         |
| Entity sensitivity | Weak handling of exact strings (e.g., Wi-Fi names) |

---

## Final Takeaways

- The system is retrieval-strong but ranking-sensitive
- Main trade-off:
  - Accuracy increases with reranking  
  - Latency decreases significantly  
- Highest-impact improvements:
  - Threshold calibration
  - Stronger reranker / fine-tuning
  - Better query rewriting for edge cases

## Repository structure

```text
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
│   ├── ab_test.py
│   ├── build_index.py
│   ├── config.py
│   ├── eval.py
│   ├── ingest.py
│   ├── retriever.py
│   └── run_rag.py
├── models/
├── requirements.txt
└── README.md
```

## Notebook workflow

### 1. Ingestion and indexing
Use `notebooks/01_data_ingestion_and_chunking.ipynb` to:
- extract PDF text
- create chunks
- rebuild BM25 and Chroma indexes

### 2. Retrieval benchmarking
Use `notebooks/02_hybrid_retrieval_baseline.ipynb` to:
- compare retrieval profiles
- save ablation results
- generate retrieval plots automatically

### 3. Full evaluation
Use `notebooks/03_model_comparison_and_evaluation.ipynb` to:
- run retrieval ablations
- evaluate the selected production profile
- save generation outputs
- inspect failure cases

## Command-line workflow

### 1. Build chunks and indexes
```bash
python -m scripts.ingest
python -m scripts.build_index
```

### 2. Run retrieval ablations
```bash
python -m scripts.ab_test
```

### 3. Evaluate the strongest profile
```bash
python -m scripts.eval --profile hybrid_rerank
```

### 4. Ask a single question
```bash
python -m scripts.run_rag --query "What time is standard check-in?" --profile hybrid_rerank
```

## Main configuration points

Important parameters live in `scripts/config.py`:

- chunk size and overlap
- retrieval profile definitions
- BM25 / dense weights
- reranker model
- no-answer thresholds
- generation model and judge model

This makes experiments easy to rerun and compare.

## Why no fine-tuning

Fine-tuning was deliberately removed from the final version. In this project, retrieval and evaluation improvements produced more reliable gains than supervised model adaptation on a small corpus. That decision makes the system cleaner, more interpretable, and more realistic for a retrieval-focused portfolio project.

## Dependencies

The project uses:
- LangChain integrations for orchestration and document handling
- Chroma for dense retrieval
- BM25 for lexical retrieval
- sentence-transformers for reranking and semantic matching
- RAGAS for generation-side evaluation
- matplotlib for plots
- pdfplumber for PDF extraction

## Strengths of the final project

- clean retrieval-first architecture
- strong evaluation discipline
- multiple retrieval baselines
- practical no-answer safety
- reproducible indexing and experiment manifest logging
- interpretable diagnostics for debugging

## Limitations

- the corpus is still relatively small compared with a full enterprise deployment
- routing is rule-based rather than learned
- abstention thresholds are heuristic and should be calibrated on a larger validation set
- answer generation is intentionally simple and constrained rather than stylistically rich

## Future work

High-value next steps would be:
- threshold calibration curves for abstention
- section-aware chunking with more precise heading preservation
- citation alignment in final answers
- a larger and harder benchmark set
- metadata filtering in the dense retriever path

## Key Contributions

- Designed a hybrid retrieval system (BM25 + dense + RRF)
- Implemented profile-based A/B testing framework
- Built full RAG evaluation pipeline (retrieval + generation + failure analysis)
- Added abstention mechanism to reduce hallucinations
- Demonstrated that retrieval improvements outperform fine-tuning on small corpora
