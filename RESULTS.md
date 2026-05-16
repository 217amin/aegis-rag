# AEGIS-RAG · Detailed Results

This document captures the full eval state of the AEGIS-RAG fraud/AML corpus build, including the ablation finding, per-failure analysis, and the trade-offs that informed production-profile selection.

## Corpus

- **6 PDFs**, 800 chunks across 415 pages
  - FINTRAC LCTR Guidance (167 chunks)
  - FINTRAC 24-hour Rule (78 chunks)
  - FINTRAC EFT Reporting (299 chunks)
  - OSFI E-21 Operational Risk (61 chunks)
  - OSFI B-13 Cyber Risk (85 chunks)
  - OSFI Third-Party Risk / B-10 (110 chunks)

## Gold sets

- **17 in-scope questions** (`qa_gold.json` + `retrieval_gold.json`) covering large cash transactions, 24-hour rule, EFT reporting, operational resilience, cyber risk capabilities, third-party criticality, and incident management
- **3 out-of-scope questions** (`abstention_gold.json`) — 2 hotel-domain (the old corpus topic), 1 EU GDPR (regulatory-adjacent but outside Canadian scope)

## Retrieval ablation

Run via `python -m scripts.ab_test`. Profiles defined in `scripts/config.py::RETRIEVAL_PROFILES`.

| Profile        | Recall@5 | nDCG@5 | MRR    | Latency  |
|----------------|----------|--------|--------|----------|
| bm25_only      | 0.667    | 0.558  | 0.561  | 1.9 ms   |
| dense_only     | 0.686    | 0.590  | 0.642  | 60.8 ms  |
| **hybrid**     | **0.686**| **0.701** | **0.794** | **63 ms** |
| hybrid_rerank  | 0.657    | 0.638  | 0.716  | 81.7 ms  |

### The reranker hurts in-scope retrieval

`hybrid_rerank` underperforms `hybrid` on every quality metric:
- Recall@5: -0.029 (0.686 → 0.657)
- nDCG@5: -0.063 (0.701 → 0.638)
- MRR: -0.078 (0.794 → 0.716)

This contradicts the conventional wisdom that adding a cross-encoder reranker improves quality. **Likely cause**: the off-the-shelf reranker (`ms-marco-MiniLM-L-6-v2`) was fine-tuned on MS MARCO web passages. Regulatory legalese ("A reporting entity shall report to the Centre…") has a different distribution than the conversational queries MS MARCO encodes. The reranker reorders chunks the hybrid scoring had already ranked well.

This is one of the most important findings in this project. Worth reading [this Pinecone post](https://www.pinecone.io/learn/series/rag/rerankers/) for the general phenomenon.

## Abstention precision

Run via the same `ab_test`, but scored against `abstention_gold.json`.

| Profile        | Precision  | Notes |
|----------------|------------|-------|
| hybrid_rerank  | **3/3 (100%)** | Rerank scores: -4.58, -10.68, -9.63 (all well below 0.25 threshold) |
| hybrid         | 0/3        | Fused scores: 0.026, 0.016, 0.025 (all above 0.005 threshold) |
| dense_only     | N/A        | No score-gate at this profile |
| bm25_only      | N/A        | No score-gate at this profile |

### The reranker is the only path to reliable abstention

The cross-encoder produces strongly-negative scores when query and chunk are semantically unrelated, which is exactly what an out-of-scope query produces. The weighted-RRF score, by contrast, is a sum of rank-based reciprocals — it stays positive even when the top-ranked chunk is barely relevant, because *something* always reaches the top.

This is why `hybrid_rerank` is the production default: **abstention reliability outweighs the in-scope ranking cost** in a regulatory domain. A hallucinated answer about a $10,000 reporting threshold is worse than no answer.

If the reranker were swapped for one trained on technical/regulatory text, this trade-off might disappear — both quality and abstention could improve together. That's the highest-priority future-work item.

## End-to-end generation eval

Run via `python -m scripts.eval --profile hybrid_rerank`. Generation by Llama-3.1-8B via Ollama, judged by RAGAS.

| Metric              | Value | Notes |
|---------------------|-------|-------|
| Faithfulness        | 0.866 | Answer is supported by retrieved context |
| Answer Relevancy    | 0.960 | Answer addresses the question |
| Context Precision   | 0.908 | Retrieved chunks were necessary for the answer |
| Semantic Match      | 0.852 | Answer is semantically close to ground truth |
| Exact Match         | 0.000 | Expected — gold answers are paraphrased, not literal |
| Avg Latency         | 2.22 s | Full pipeline incl. cross-encoder + LLM |

## Failure analysis (3 of 17 in-scope)

### 1. "What does OSFI Guideline E-21 cover?" — `retrieval_miss`

- Gold chunks: `04_OSFI_E21_Operational_Risk_p2_c2`, `_p3_c1`
- Retrieved (top-5): `05_OSFI_B13_Cyber_Risk_p4_c2`, `06_OSFI_Third_Party_Risk_p7_c2`, `04_OSFI_E21_Operational_Risk_p1_c1` (metadata), `06_OSFI_Third_Party_Risk_p6_c2`, `06_OSFI_Third_Party_Risk_p4_c1`
- LLM output: "OSFI Guideline E-21 covers Operational Risk Management."

The retriever found the E-21 metadata chunk (p1_c1) but missed the substantive p2_c2 and p3_c1 chunks. The LLM produced a thin but technically-correct answer using the title from p1_c1. **Real retrieval failure** — the question's semantic intent ("what does it cover?") didn't match the chunk content style.

### 2. "What is the scope of OSFI Guideline B-13?" — `retrieval_miss`

- Gold chunks: `05_OSFI_B13_Cyber_Risk_p3_c1`, `_p4_c1`
- Retrieved (top-5): mostly Third-Party Risk + E-21 chunks; only `05_OSFI_B13_Cyber_Risk_p1_c1` (metadata) from B-13
- LLM output: A polished answer about "Technology and Cyber Risk Management" — semantically correct but **constructed from the B-13 title + LLM priors**, not from grounded evidence.

This is the more concerning failure. The LLM hallucinated a plausible-sounding regulatory answer using its training data. Faithfulness should have caught this; the high aggregate (0.866) suggests RAGAS gave partial credit because the answer is technically correct. **This is the case that should be tightened with stricter prompting and a lower abstention threshold for cross-document confusion.**

### 3. "What are the cyber-security capabilities outlined in B-13?" — `retrieval_miss`

Same pattern as #2: B-13 chunks rejected by the reranker, E-21/B-10 chunks surfaced instead. LLM produced an answer citing "Guideline B-13 section 4.3" with capabilities that look plausible but weren't grounded in the retrieved context.

### Pattern

All three failures involve cross-document regulatory questions where multiple OSFI guidelines have overlapping vocabulary (cyber, operational, risk). The reranker tends to surface chunks from the more-frequent terms (E-21, B-10) rather than the less-frequent target document (B-13). This is consistent with the in-scope ranking degradation seen in the ablation.

**Fix direction**: domain-tuned reranker (BAAI/bge-reranker-base or mxbai-rerank-large-v1) trained on technical text rather than MS MARCO.

## Reproducibility

```bash
# Full pipeline from PDFs to eval results
python -m scripts.ingest
python -m scripts.build_index
python -m scripts.ab_test                      # writes results_ablation.json + results_abstention.json
python -m scripts.eval --profile hybrid_rerank # writes results_generation.json + results_failures.json
```

Every run also writes `data/eval/experiment_manifest.json` capturing the exact config used.
