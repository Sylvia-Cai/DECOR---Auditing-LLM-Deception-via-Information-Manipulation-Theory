"""
scripts/eval_imt.py
====================
Evaluate IMT audit results: AUROC + 5-fold binary metrics per model.

Usage:
    # Evaluate default model (azure_gpt4o)
    python scripts/eval_imt.py

    # Evaluate specific models
    python scripts/eval_imt.py --models azure_gpt4o azure_claude_sonnet46

    # Evaluate ALL models found in results/imt_audit/
    python scripts/eval_imt.py --all

Output (results/eval/imt/):
    <model>.json          per-model detailed metrics
    summary.txt           human-readable table across all evaluated models
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from project_paths import IMT_AUDIT_RESULTS_DIR, EVAL_RESULTS_DIR
from workflows.imt_scores import (
    DEFAULT_LABELS,
    DOMAINS, POOLING_STRATEGIES, RESPONSE_STRATEGY, SIDES, THOUGHT_STRATEGY,
    aggregate_auroc_runs, aggregate_binary_runs,
    compute_auroc_all, compute_binary_cv_both_sides,
    compute_oof_predictions,
    find_model_files, load_records,
)

DEFAULT_MODEL = "azure_gpt4o"
OUT_DIR = EVAL_RESULTS_DIR / "imt"
LABELED_OUT_DIR = ROOT / "results" / "imt_audit_with_label"

METRIC_LABELS = {
    "f1":           "Macro-F1",
    "auprc_oof":    "AUPRC (OOF)",
    "acc":          "Accuracy",
    "balanced_acc": "Balanced Accuracy",
    "precision":    "Precision",
    "recall":       "Recall",
    "fpr":          "False Positive Rate",
    "fnr":          "False Negative Rate",
}


# ── Labeled JSON output ───────────────────────────────────────────────────────

def _write_labeled_json(
    result_path: Path,
    records: list,
    predictions: dict,
    out_dir: Path,
) -> Path:
    """Augment an IMT audit result file with OOF eval labels and save."""
    with open(result_path, encoding="utf-8") as f:
        data = json.load(f)

    for item in data:
        topic    = item.get("topic", "")
        question = item.get("question", "")
        for l2_type, content in item.get("results", {}).items():
            pred = predictions.get((topic, question, l2_type))
            if pred is None:
                continue
            content["eval"] = {
                "thought":  "decept" if pred["thought"]  == 1 else "honest",
                "response": "decept" if pred["response"] == 1 else "honest",
            }

    out_path = out_dir / result_path.name
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    return out_path


# ── Evaluation ────────────────────────────────────────────────────────────────

def _count_valid_records(records: list) -> dict:
    """Count valid (non-NaN) records per side for metrics calculation."""
    import numpy as np
    counts = {"thought": 0, "response": 0}
    for r in records:
        thought_score = r["score_thought"].get(THOUGHT_STRATEGY, float("nan"))
        response_score = r["score_response"].get(RESPONSE_STRATEGY, float("nan"))
        if not np.isnan(thought_score):
            counts["thought"] += 1
        if not np.isnan(response_score):
            counts["response"] += 1
    return counts


def eval_model(
    model: str,
    files: list,
    labels_path: Path = DEFAULT_LABELS,
    labeled_out_dir: Path = LABELED_OUT_DIR,
) -> dict:
    """Run AUROC + 5-fold binary CV across all run files for a model."""
    print(f"\n  {model}  ({len(files)} run(s))")
    auroc_runs, binary_runs, data_counts = [], [], []

    for path in files:
        records = load_records(path, labels_path=labels_path)
        if not records:
            print(f"    [WARN] No valid records in {path.name}")
            continue
        valid_counts = _count_valid_records(records)
        print(f"    {path.name}: {len(records)} records  [thought={valid_counts['thought']}, response={valid_counts['response']} valid]")
        auroc_runs.append(compute_auroc_all(records))
        binary_runs.append(compute_binary_cv_both_sides(records))
        data_counts.append({"file": path.name, "total": len(records), **valid_counts})

        # Generate labeled JSON with OOF predictions
        predictions = compute_oof_predictions(records)
        labeled_path = _write_labeled_json(path, records, predictions, labeled_out_dir)
        print(f"    labeled -> {labeled_path.relative_to(ROOT)}")

    if not auroc_runs:
        return {}

    if len(auroc_runs) == 1:
        auroc_result  = {"single": auroc_runs[0]}
        binary_result = {"single": binary_runs[0]}
    else:
        auroc_result  = aggregate_auroc_runs(auroc_runs)
        binary_result = aggregate_binary_runs(binary_runs)

    return {"auroc": auroc_result, "binary": binary_result, "runs": len(auroc_runs), "data_counts": data_counts}


# ── Summary formatting ────────────────────────────────────────────────────────

def _get(block: dict, key: str) -> tuple[float, float]:
    """Return (mean, std) from a metric block."""
    if "mean" in block:
        return block["mean"].get(key, float("nan")), block["std"].get(key, float("nan"))
    if "single" in block:
        m = block["single"].get(f"{key}_mean", block["single"].get(key, float("nan")))
        s = block["single"].get(f"{key}_std", float("nan"))
        return m, s
    return float("nan"), float("nan")


def _get_auroc(auroc_block: dict, side: str, strategy: str, domain: str = "Overall") -> tuple[float, float]:
    if "mean" in auroc_block:
        m = auroc_block["mean"][side][strategy][domain]
        s = auroc_block["std"][side][strategy][domain]
    elif "single" in auroc_block:
        m = auroc_block["single"][side][strategy][domain]
        s = float("nan")
    else:
        return float("nan"), float("nan")
    return m, s


def _fmt(mean: float, std: float) -> str:
    if mean != mean:   # nan check
        return "   NA   "
    return f"{mean:.4f}±{std:.4f}"


def build_summary(all_results: dict) -> str:
    lines = []
    lines.append("IMT Evaluation Summary")
    lines.append(f"Config: thought={THOUGHT_STRATEGY}, response={RESPONSE_STRATEGY}, 5-fold CV, domain-specific threshold")
    lines.append("")

    for model, result in sorted(all_results.items()):
        if not result:
            continue
        lines.append(f"Model: {model}  ({result['runs']} run(s))")
        
        # Data availability summary
        if "data_counts" in result:
            for dc in result["data_counts"]:
                lines.append(f"  {dc['file']}: {dc['total']} total, thought={dc['thought']} valid, response={dc['response']} valid")
        
        auroc_b  = result["auroc"]
        binary_b = result["binary"]

        for side, strategy in (("thought", THOUGHT_STRATEGY), ("response", RESPONSE_STRATEGY)):
            auroc_m, auroc_s = _get_auroc(auroc_b, side, strategy)
            bin_block = binary_b.get("mean", binary_b.get("single", {}))
            side_block = bin_block.get(side, {}) if "mean" in binary_b else bin_block.get(side, {})

            lines.append(f"  {side.capitalize()}  (strategy={strategy})")
            lines.append(f"    AUROC         : {_fmt(auroc_m, auroc_s)}")

            for key, label in METRIC_LABELS.items():
                if "mean" in binary_b:
                    m = binary_b["mean"][side].get(key, float("nan"))
                    s = binary_b["std"][side].get(key, float("nan"))
                else:
                    base = binary_b["single"][side]
                    if key == "auprc_oof":
                        m = base.get("auprc_oof", float("nan"))
                        s = float("nan")
                    else:
                        m = base.get(f"{key}_mean", float("nan"))
                        s = base.get(f"{key}_std",  float("nan"))
                lines.append(f"    {label:<20}: {_fmt(m, s)}")
        
        lines.append("")

    return "\n".join(lines)


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate IMT audit results")
    parser.add_argument("--models", nargs="+", default=[DEFAULT_MODEL],
                        help=f"Model preset keys to evaluate (default: {DEFAULT_MODEL})")
    parser.add_argument("--all", action="store_true",
                        help="Evaluate all models found in results/imt_audit/")
    args = parser.parse_args()

    model_filter = None if args.all else args.models
    model_files  = find_model_files(IMT_AUDIT_RESULTS_DIR, model_filter)

    if not model_files:
        print(f"[ERROR] No IMT result files found in {IMT_AUDIT_RESULTS_DIR}")
        sys.exit(1)

    print(f"Found {len(model_files)} model(s): {', '.join(sorted(model_files))}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    LABELED_OUT_DIR.mkdir(parents=True, exist_ok=True)

    all_results = {}
    for model, files in sorted(model_files.items()):
        result = eval_model(model, files)
        all_results[model] = result
        out = OUT_DIR / f"{model}.json"
        with open(out, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"    -> {out.relative_to(ROOT)}")

    summary_path = OUT_DIR / "summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(build_summary(all_results))
    print(f"\nSummary: {summary_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
