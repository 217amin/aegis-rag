from __future__ import annotations

import json
from pathlib import Path

from scripts.config import EVAL_DIR, RETRIEVAL_PROFILES, RESULTS_ABLATION_FILE, RESULTS_ABSTENTION_FILE
from scripts.eval import (
    evaluate_retrieval_profile,
    evaluate_abstention,
    load_json,
    save_json,
)
from scripts.retriever import load_retrieval_assets


def main() -> None:
    print("Loading functional retrieval assets...")
    assets = load_retrieval_assets()
    
    retrieval_gold = load_json(EVAL_DIR / "retrieval_gold.json")
    results = [
        evaluate_retrieval_profile(retrieval_gold, profile_name, assets)
        for profile_name in RETRIEVAL_PROFILES.keys()
    ]
    save_json(results, RESULTS_ABLATION_FILE)

    # Abstention eval (if gold file present)
    abstention_path = EVAL_DIR / "abstention_gold.json"
    if abstention_path.exists():
        abstention_gold = load_json(abstention_path)
        abstention_results = [
            evaluate_abstention(abstention_gold, profile_name, assets)
            for profile_name in RETRIEVAL_PROFILES.keys()
        ]
        save_json(abstention_results, RESULTS_ABSTENTION_FILE)
        print("\nAbstention precision:")
        for r in abstention_results:
            agg = r["aggregate"]
            p = agg["abstention_precision"]
            if p is not None:
                print(f"  {r['profile_name']:15s} {p:.2%} "
                      f"({agg['num_correct']}/{agg['num_applicable']})")
            else:
                print(f"  {r['profile_name']:15s} N/A")
    else:
        print(f"Skipped abstention eval — {abstention_path} not found.")

    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()