"""
scripts/eval_baselines.py
==========================
Evaluate baseline result files: same metric set as eval_imt.py for fair comparison.

Usage:
    # List available baseline families
    python scripts/eval_baselines.py --list

    # Evaluate default model (azure_gpt4o) across all families
    python scripts/eval_baselines.py

    # Evaluate a specific family and model
    python scripts/eval_baselines.py --family simple_llm_judge --models azure_gpt4o

    # Evaluate ALL models in a family
    python scripts/eval_baselines.py --family simple_llm_judge --all

Output (results/eval/baselines/<family>/):
    <model>.json          per-model metrics
    summary.txt           human-readable table
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from project_paths import BASELINE_RESULTS_DIR, EVAL_RESULTS_DIR
from workflows.baseline_scores import (
    SIDES, evaluate_model, find_baseline_files,
    list_baseline_families, load_truth_index,
)

DEFAULT_MODEL = "azure_gpt4o"
OUT_BASE = EVAL_RESULTS_DIR / "baselines"

METRIC_LABELS = {
    "auroc":        "AUROC",
    "f1":           "Macro-F1",
    "auprc":        "AUPRC",
    "acc":          "Accuracy",
    "balanced_acc": "Balanced Accuracy",
    "precision":    "Precision",
    "recall":       "Recall",
    "fpr":          "False Positive Rate",
    "fnr":          "False Negative Rate",
}


# ── Summary formatting ────────────────────────────────────────────────────────

def _fmt(mean: float, std: float) -> str:
    if mean != mean:
        return "   NA   "
    return f"{mean:.4f}±{std:.4f}"


def build_summary(family: str, all_results: dict) -> str:
    lines = [f"Baseline Evaluation Summary  [{family}]", ""]
    for model, result in sorted(all_results.items()):
        if not result:
            continue
        runs = result.get("runs", 1)
        lines.append(f"Model: {model}  ({runs} run(s))")
        block = result.get("mean", result.get("single", {}))
        std_b = result.get("std", {side: {k: float("nan") for k in METRIC_LABELS} for side in SIDES})
        for side in SIDES:
            lines.append(f"  {side.capitalize()}")
            for key, label in METRIC_LABELS.items():
                m = block.get(side, {}).get(key, float("nan"))
                s = std_b.get(side, {}).get(key, float("nan"))
                lines.append(f"    {label:<20}: {_fmt(m, s)}")
        lines.append("")
    return "\n".join(lines)


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate baseline results")
    parser.add_argument("--list",   action="store_true",
                        help="List available baseline families and exit")
    parser.add_argument("--family", default=None,
                        help="Baseline family to evaluate (default: all families)")
    parser.add_argument("--models", nargs="+", default=[DEFAULT_MODEL],
                        help=f"Model preset keys (default: {DEFAULT_MODEL})")
    parser.add_argument("--all",    action="store_true",
                        help="Evaluate all models found in the family directory")
    args = parser.parse_args()

    families_available = list_baseline_families(BASELINE_RESULTS_DIR)

    if args.list:
        print("Available baseline families:")
        for f in families_available:
            print(f"  {f}")
        return

    families = [args.family] if args.family else families_available
    if not families:
        print(f"[ERROR] No baseline families found in {BASELINE_RESULTS_DIR}")
        sys.exit(1)

    truth_index = load_truth_index()

    for family in families:
        family_dir = BASELINE_RESULTS_DIR / family
        if not family_dir.is_dir():
            print(f"[SKIP] {family}: directory not found")
            continue

        model_filter = None if args.all else args.models
        model_files  = find_baseline_files(family_dir, model_filter)

        if not model_files:
            print(f"[SKIP] {family}: no matching result files")
            continue

        print(f"\nFamily: {family}  ({len(model_files)} model(s))")
        out_dir = OUT_BASE / family
        out_dir.mkdir(parents=True, exist_ok=True)

        all_results = {}
        for model, files in sorted(model_files.items()):
            print(f"  {model}  ({len(files)} run(s))")
            result = evaluate_model(files, truth_index)
            all_results[model] = result
            out = out_dir / f"{model}.json"
            with open(out, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            print(f"    -> {out.relative_to(ROOT)}")

        summary_path = out_dir / "summary.txt"
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(build_summary(family, all_results))
        print(f"  Summary: {summary_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
