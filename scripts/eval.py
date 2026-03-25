from __future__ import annotations

import argparse
import csv
import json
import math
import time
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from datasets import Dataset
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from ragas import evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import answer_relevancy, context_precision, faithfulness
from ragas.run_config import RunConfig
from sentence_transformers import SentenceTransformer, util

from scripts.config import (
    EMBEDDING_MODEL,
    EVAL_DIR,
    EXPERIMENT_MANIFEST_FILE,
    JUDGE_MODEL,
    LLM_MODEL,
    PLOTS_DIR,
    RESULTS_ABLATION_FILE,
    RESULTS_FAILURES_FILE,
    RESULTS_GENERATION_FILE,
    RESULTS_RETRIEVAL_FILE,
    RESULTS_SUMMARY_FILE,
    RETRIEVAL_PROFILES,
    SAFE_FALLBACK_ANSWER,
    SEMANTIC_MATCH_MODEL,
    save_experiment_manifest,
)
from scripts.retriever import HybridRetriever
from scripts.run_rag import answer_question


def load_json(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def save_summary_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def safe_mean(values: list[float | None]) -> float | None:
    clean = [float(v) for v in values if v is not None]
    return (sum(clean) / len(clean)) if clean else None


def coerce_optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        numeric = float(value)
    except Exception:
        return None
    return None if (math.isnan(numeric) or math.isinf(numeric)) else numeric


def recall_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    if not relevant_ids:
        return 1.0 if not retrieved_ids[:k] else 0.0
    hits = sum(1 for cid in retrieved_ids[:k] if cid in relevant_ids)
    return hits / len(relevant_ids)


def dcg_at_k(retrieved_ids: list[str], relevance_map: dict[str, float], k: int) -> float:
    dcg = 0.0
    for i, chunk_id in enumerate(retrieved_ids[:k], start=1):
        rel = relevance_map.get(chunk_id, 0.0)
        if rel > 0:
            dcg += rel / math.log2(i + 1)
    return dcg


def ndcg_at_k(retrieved_ids: list[str], relevance_map: dict[str, float], k: int) -> float:
    actual = dcg_at_k(retrieved_ids, relevance_map, k)
    ideal_ids = sorted(relevance_map, key=relevance_map.__getitem__, reverse=True)
    ideal = dcg_at_k(ideal_ids, relevance_map, k)
    return (actual / ideal) if ideal else 0.0


def mean_reciprocal_rank(retrieved_ids: list[str], relevant_ids: set[str]) -> float:
    for rank, cid in enumerate(retrieved_ids, start=1):
        if cid in relevant_ids:
            return 1.0 / rank
    return 0.0


def exact_match(pred: str, truth: str) -> float:
    return float(pred.strip().lower() == truth.strip().lower())


def semantic_match(pred: str, truth: str, embed_model: SentenceTransformer) -> float:
    embeddings = embed_model.encode([pred, truth], convert_to_tensor=True)
    return float(util.cos_sim(embeddings[0], embeddings[1]).item())


def run_ragas_with_retries(
    ragas_payload: dict[str, list[Any]],
    judge_model: str,
    request_timeout: int,
    ragas_retries: int,
) -> tuple[dict[int, dict[str, float | None]], str | None]:
    dataset = Dataset.from_dict(ragas_payload)
    ragas_embeddings = LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    )

    empty = lambda n: {
        i: {
            "faithfulness": None,
            "answer_relevancy": None,
            "context_precision": None,
        }
        for i in range(n)
    }

    last_error: str | None = None

    for attempt in range(1, ragas_retries + 1):
        try:
            judge_llm = ChatOllama(
                model=judge_model,
                temperature=0.0,
                format="json",
                num_predict=1024,
                request_timeout=request_timeout,
            )
            ragas_result = evaluate(
                dataset=dataset,
                metrics=[faithfulness, answer_relevancy, context_precision],
                llm=LangchainLLMWrapper(judge_llm),
                embeddings=ragas_embeddings,
                run_config=RunConfig(timeout=800, max_workers=1),
            )
            ragas_df = ragas_result.to_pandas()
            return {
                i: {
                    "faithfulness": coerce_optional_float(ragas_df.loc[i, "faithfulness"]),
                    "answer_relevancy": coerce_optional_float(ragas_df.loc[i, "answer_relevancy"]),
                    "context_precision": coerce_optional_float(ragas_df.loc[i, "context_precision"]),
                }
                for i in range(len(ragas_df))
            }, None
        except Exception as exc:
            last_error = f"attempt={attempt}: {type(exc).__name__}: {exc}"
            if attempt < ragas_retries:
                time.sleep(min(attempt * 2, 6))

    return empty(len(ragas_payload["question"])), last_error


def evaluate_retrieval_profile(
    retrieval_gold: list[dict[str, Any]],
    profile_name: str,
) -> dict[str, Any]:
    per_query = []

    # Build retriever once, outside timing
    routed_retriever = HybridRetriever(profile_name=profile_name)

    # Warmup once so first-query overhead does not distort QPS
    if retrieval_gold:
        first_query = retrieval_gold[0]["query"]
        _docs, _diagnostics = routed_retriever.retrieve_with_diagnostics(first_query)

    start = time.perf_counter()

    for item in retrieval_gold:
        query = item["query"]
        docs, diagnostics = routed_retriever.retrieve_with_diagnostics(query)

        # Deduplicate IDs while preserving order
        retrieved_ids = list(dict.fromkeys(d.metadata["chunk_id"] for d in docs))

        if "relevance" in item:
            relevance_map = {k: float(v) for k, v in item["relevance"].items()}
            relevant_ids = set(relevance_map.keys())
        else:
            relevant_ids = set(item.get("relevant_chunk_ids", []))
            relevance_map = {cid: 1.0 for cid in relevant_ids}

        per_query.append({
            "query": query,
            "recall@5": recall_at_k(retrieved_ids, relevant_ids, 5),
            "recall@10": recall_at_k(retrieved_ids, relevant_ids, 10),
            "ndcg@5": ndcg_at_k(retrieved_ids, relevance_map, 5),
            "mrr": mean_reciprocal_rank(retrieved_ids, relevant_ids),
            "retrieved_ids": retrieved_ids,
            "relevant_ids": sorted(relevant_ids),
            "retrieval_diagnostics": diagnostics,
        })

    elapsed = time.perf_counter() - start

    aggregate = {
        "recall@5": safe_mean([x["recall@5"] for x in per_query]),
        "recall@10": safe_mean([x["recall@10"] for x in per_query]),
        "ndcg@5": safe_mean([x["ndcg@5"] for x in per_query]),
        "mrr": safe_mean([x["mrr"] for x in per_query]),
        "qps": (len(per_query) / elapsed) if elapsed > 0 else None,
        "avg_latency_sec": (elapsed / len(per_query)) if per_query else None,
    }

    return {
        "profile_name": profile_name,
        "aggregate": aggregate,
        "per_query": per_query,
    }


def evaluate_generation(
    qa_gold: list[dict[str, Any]],
    retriever: HybridRetriever,
    model_name: str,
    judge_model: str,
    profile_name: str,
    request_timeout: int,
    ragas_retries: int,
    skip_ragas: bool,
) -> dict[str, Any]:
    semantic_embedder = SentenceTransformer(SEMANTIC_MATCH_MODEL)
    records: list[dict[str, Any]] = []
    ragas_payload: dict[str, list[Any]] = {
        "question": [],
        "answer": [],
        "contexts": [],
        "ground_truth": [],
    }

    retriever.profile_name = profile_name

    for item in qa_gold:
        question = item["question"]
        ground_truth = item["ground_truth"]
        answer_type = item.get("answer_type", "semantic")

        start = time.perf_counter()
        result = answer_question(
            question=question,
            retriever=retriever,
            model_name=model_name,
            request_timeout=request_timeout,
        )
        latency = time.perf_counter() - start

        answer = result["answer"]
        contexts = [c["text"] for c in result["retrieved_chunks"]]
        retrieved_ids = [c["chunk_id"] for c in result["retrieved_chunks"]]

        em = exact_match(answer, ground_truth)
        sem = semantic_match(answer, ground_truth, semantic_embedder)

        diagnostics = result.get("retrieval_diagnostics", {})

        records.append({
            "id": item["id"],
            "question": question,
            "model_name": model_name,
            "profile_name": profile_name,
            "answer": answer,
            "ground_truth": ground_truth,
            "exact_match": em,
            "semantic_match": sem,
            "latency_sec": latency,
            "tokens_per_answer_est": max(1, len(answer) // 4),
            "retrieved_ids": retrieved_ids,
            "retrieved_sources": [f"{c['source']}#p{c['page']}" for c in result["retrieved_chunks"]],
            "retrieval_diagnostics": diagnostics,
            "top_rerank_score": diagnostics.get("top_rerank_score"),
            "top_fused_score": diagnostics.get("top_weighted_rrf_score"),
            "primary_match_metric": em if answer_type == "exact" else sem,
        })

        ragas_payload["question"].append(question)
        ragas_payload["answer"].append(answer)
        ragas_payload["contexts"].append(contexts)
        ragas_payload["ground_truth"].append(ground_truth)

    if skip_ragas:
        ragas_metrics = {
            i: {
                "faithfulness": None,
                "answer_relevancy": None,
                "context_precision": None,
            }
            for i in range(len(records))
        }
        ragas_error = None
    else:
        ragas_metrics, ragas_error = run_ragas_with_retries(
            ragas_payload,
            judge_model,
            request_timeout,
            ragas_retries,
        )

    for i, rec in enumerate(records):
        rec.update(ragas_metrics[i])

    total_latency = sum(r["latency_sec"] for r in records if r["latency_sec"] is not None)
    aggregate = {
        "faithfulness": safe_mean([r["faithfulness"] for r in records]),
        "answer_relevancy": safe_mean([r["answer_relevancy"] for r in records]),
        "context_precision": safe_mean([r["context_precision"] for r in records]),
        "exact_match": safe_mean([r["exact_match"] for r in records]),
        "semantic_match": safe_mean([r["semantic_match"] for r in records]),
        "avg_latency_sec": safe_mean([r["latency_sec"] for r in records]),
        "avg_tokens_per_answer_est": safe_mean([r["tokens_per_answer_est"] for r in records]),
        "avg_top_rerank_score": safe_mean([r["top_rerank_score"] for r in records]),
        "avg_top_fused_score": safe_mean([r["top_fused_score"] for r in records]),
        "qps": (len(records) / total_latency) if total_latency > 0 else None,
        "num_examples": len(records),
        "num_ragas_missing": sum(
            1
            for r in records
            if any(r[k] is None for k in ("faithfulness", "answer_relevancy", "context_precision"))
        ),
    }

    return {
        "model_name": model_name,
        "judge_model": judge_model,
        "profile_name": profile_name,
        "aggregate": aggregate,
        "per_question": records,
        "ragas_error": ragas_error,
    }


def build_failure_report(
    qa_gold: list[dict[str, Any]],
    retrieval_gold: list[dict[str, Any]],
    generation_results: dict[str, Any],
) -> list[dict[str, Any]]:
    retrieval_by_question = {
        item["query"]: set(item.get("relevant_chunk_ids", []))
        for item in retrieval_gold
    }

    failures: list[dict[str, Any]] = []

    for item, rec in zip(qa_gold, generation_results["per_question"]):
        relevant_ids = retrieval_by_question.get(item["question"], set())
        retrieved_ids = set(rec["retrieved_ids"])
        is_negative = item["ground_truth"] == SAFE_FALLBACK_ANSWER
        retrieval_hit = bool(relevant_ids & retrieved_ids)
        predicted_fallback = rec["answer"] == SAFE_FALLBACK_ANSWER

        if is_negative:
            if predicted_fallback:
                error_type = "correct_no_answer"
            else:
                error_type = "unsupported_question_answered"
        
        elif not retrieval_hit:
            error_type = "retrieval_miss"
        
        elif (
            rec["faithfulness"] is not None
            and rec["faithfulness"] < 0.5
            and rec["semantic_match"] < 0.6
        ):
            error_type = "generation_hallucination"
        
        elif rec["semantic_match"] < 0.7:
            error_type = "ranking_or_answer_miss"
        
        else:
            error_type = "correct"

        failures.append({
            "id": item["id"],
            "question": item["question"],
            "error_type": error_type,
            "ground_truth": item["ground_truth"],
            "answer": rec["answer"],
            "retrieved_ids": rec["retrieved_ids"],
            "relevant_ids": sorted(relevant_ids),
            "faithfulness": rec["faithfulness"],
            "semantic_match": rec["semantic_match"],
            "retrieval_diagnostics": rec["retrieval_diagnostics"],
            "predicted_fallback": predicted_fallback,
        })

    return failures


def plot_retrieval_profiles(ablation_results: list[dict[str, Any]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    ranked = sorted(
        ablation_results,
        key=lambda x: (x["aggregate"]["ndcg@5"] or 0.0),
        reverse=True,
    )

    names = [r["profile_name"] for r in ranked]
    ndcg = [r["aggregate"]["ndcg@5"] for r in ranked]
    recall = [r["aggregate"]["recall@5"] for r in ranked]
    qps = [r["aggregate"]["qps"] for r in ranked]

    plt.figure(figsize=(9, 4.8))
    plt.bar(names, ndcg)
    plt.ylabel("nDCG@5")
    plt.title("Retrieval profile comparison")
    plt.tight_layout()
    plt.savefig(output_dir / "retrieval_profile_ndcg.png", dpi=160)
    plt.close()

    plt.figure(figsize=(9, 4.8))
    plt.bar(names, recall)
    plt.ylabel("Recall@5")
    plt.title("Retrieval profile recall comparison")
    plt.tight_layout()
    plt.savefig(output_dir / "retrieval_profile_recall.png", dpi=160)
    plt.close()

    plt.figure(figsize=(8, 4.8))
    plt.scatter(qps, ndcg)
    for x, y, label in zip(qps, ndcg, names):
        plt.annotate(label, (x, y))
    plt.xlabel("QPS")
    plt.ylabel("nDCG@5")
    plt.title("Latency-quality tradeoff")
    plt.tight_layout()
    plt.savefig(output_dir / "latency_quality_tradeoff.png", dpi=160)
    plt.close()


def build_summary_rows(
    retrieval_best: dict[str, Any],
    generation_results: dict[str, Any],
) -> list[dict[str, Any]]:
    retrieval_agg = retrieval_best["aggregate"]
    gen_agg = generation_results["aggregate"]

    return [{
        "model_name": generation_results["model_name"],
        "judge_model": generation_results["judge_model"],
        "profile_name": generation_results["profile_name"],
        "retrieval_recall@5": retrieval_agg["recall@5"],
        "retrieval_recall@10": retrieval_agg["recall@10"],
        "retrieval_ndcg@5": retrieval_agg["ndcg@5"],
        "retrieval_mrr": retrieval_agg["mrr"],
        "retrieval_qps": retrieval_agg["qps"],
        "retrieval_avg_latency_sec": retrieval_agg["avg_latency_sec"],
        "generation_faithfulness": gen_agg["faithfulness"],
        "generation_answer_relevancy": gen_agg["answer_relevancy"],
        "generation_context_precision": gen_agg["context_precision"],
        "generation_exact_match": gen_agg["exact_match"],
        "generation_semantic_match": gen_agg["semantic_match"],
        "generation_avg_latency_sec": gen_agg["avg_latency_sec"],
        "generation_avg_tokens_per_answer_est": gen_agg["avg_tokens_per_answer_est"],
        "generation_avg_top_rerank_score": gen_agg["avg_top_rerank_score"],
        "generation_avg_top_fused_score": gen_agg["avg_top_fused_score"],
        "generation_qps": gen_agg["qps"],
        "num_examples": gen_agg["num_examples"],
        "num_ragas_missing": gen_agg["num_ragas_missing"],
        "ragas_error": generation_results["ragas_error"],
    }]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--judge_model", type=str, default=JUDGE_MODEL)
    parser.add_argument("--base_model", type=str, default=LLM_MODEL)
    parser.add_argument("--request_timeout", type=int, default=300)
    parser.add_argument("--ragas_retries", type=int, default=2)
    parser.add_argument("--profile", type=str, default="hybrid_rerank")
    parser.add_argument("--skip_ragas", action="store_true")
    parser.add_argument("--output_suffix", type=str, default="")
    args = parser.parse_args()

    retrieval_gold = load_json(EVAL_DIR / "retrieval_gold.json")
    qa_gold = load_json(EVAL_DIR / "qa_gold.json")

    if args.profile not in RETRIEVAL_PROFILES:
        raise ValueError(
            f"Unknown profile: {args.profile}. "
            f"Available: {list(RETRIEVAL_PROFILES.keys())}"
        )

    retriever = HybridRetriever(profile_name=args.profile)

    save_experiment_manifest(EXPERIMENT_MANIFEST_FILE)

    ablation_results = [
        evaluate_retrieval_profile(retrieval_gold, profile_name)
        for profile_name in RETRIEVAL_PROFILES.keys()
    ]

    best_retrieval = next(
        (x for x in ablation_results if x["profile_name"] == args.profile),
        None,
    )
    if best_retrieval is None:
        raise ValueError(
            f"Could not find retrieval result for profile: {args.profile}"
        )

    generation_results = evaluate_generation(
        qa_gold=qa_gold,
        retriever=retriever,
        model_name=args.base_model,
        judge_model=args.judge_model,
        profile_name=best_retrieval["profile_name"],
        request_timeout=args.request_timeout,
        ragas_retries=args.ragas_retries,
        skip_ragas=args.skip_ragas,
    )

    failures = build_failure_report(qa_gold, retrieval_gold, generation_results)

    retrieval_file = RESULTS_RETRIEVAL_FILE
    generation_file = RESULTS_GENERATION_FILE
    summary_file = RESULTS_SUMMARY_FILE
    failures_file = RESULTS_FAILURES_FILE
    ablation_file = RESULTS_ABLATION_FILE

    if args.output_suffix:
        suffix = args.output_suffix.strip()
        retrieval_file = retrieval_file.with_name(f"{retrieval_file.stem}_{suffix}{retrieval_file.suffix}")
        generation_file = generation_file.with_name(f"{generation_file.stem}_{suffix}{generation_file.suffix}")
        summary_file = summary_file.with_name(f"{summary_file.stem}_{suffix}{summary_file.suffix}")
        failures_file = failures_file.with_name(f"{failures_file.stem}_{suffix}{failures_file.suffix}")
        ablation_file = ablation_file.with_name(f"{ablation_file.stem}_{suffix}{ablation_file.suffix}")

    save_json(best_retrieval, retrieval_file)
    save_json(generation_results, generation_file)
    save_json(failures, failures_file)
    save_json(ablation_results, ablation_file)
    save_summary_csv(build_summary_rows(best_retrieval, generation_results), summary_file)
    plot_retrieval_profiles(ablation_results, PLOTS_DIR)

    error_counts = Counter(f["error_type"] for f in failures)

    print(f"Saved retrieval results → {retrieval_file}")
    print(f"Saved generation results → {generation_file}")
    print(f"Saved failure report → {failures_file}")
    print(f"Saved retrieval ablation → {ablation_file}")
    print(f"Saved summary → {summary_file}")
    print(f"Saved experiment manifest → {EXPERIMENT_MANIFEST_FILE}")
    print(f"Saved plots → {PLOTS_DIR}")

    print("\nBest retrieval aggregate:")
    print(json.dumps(best_retrieval["aggregate"], ensure_ascii=False, indent=2))

    print("\nGeneration aggregate:")
    print(json.dumps(generation_results["aggregate"], ensure_ascii=False, indent=2))

    print("\nError breakdown:")
    for k, v in sorted(error_counts.items()):
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()