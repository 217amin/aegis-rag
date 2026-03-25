from __future__ import annotations

"""Run retrieval-only ablations across all configured profiles."""

import json

from scripts.config import EVAL_DIR, RETRIEVAL_PROFILES, RESULTS_ABLATION_FILE
from scripts.eval import evaluate_retrieval_profile, load_json, save_json


def main() -> None:
    """Evaluate every retrieval profile and save a single ablation artifact."""
    retrieval_gold = load_json(EVAL_DIR / "retrieval_gold.json")
    results = [
        evaluate_retrieval_profile(retrieval_gold, profile_name)
        for profile_name in RETRIEVAL_PROFILES.keys()
    ]
    save_json(results, RESULTS_ABLATION_FILE)
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
