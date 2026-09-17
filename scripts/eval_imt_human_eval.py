"""
scripts/eval_imt_human_eval.py
================================
Evaluate IMT audit results produced by scripts/run_imt_human_eval.py: AUROC +
5-fold binary metrics per (target model, auditor model) pair. Mirrors
scripts/eval_imt.py, but each target model's audit results are matched against
its own human-eval ground truth file (data/human_eval/models/<target>_dataset_with_labels.json)
instead of the deepseek labels.

Usage:
    python scripts/eval_imt_human_eval.py                        # evaluate all targets, all auditors found
    python scripts/eval_imt_human_eval.py --targets gpt4o         # restrict to one target
    python scripts/eval_imt_human_eval.py --auditors azure_gpt4o azure_claude_sonnet46

Output (results/eval/imt_human_eval/):
    <target>/<auditor>.json     per-(target, auditor) detailed metrics
    <target>/summary.txt        human-readable table for that target
    summary.txt                 combined table across all targets
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from project_paths import HUMAN_EVAL_AUDIT_RESULTS_DIR, HUMAN_EVAL_DATA_DIR, HUMAN_EVAL_EVAL_RESULTS_DIR
from scripts.eval_imt import build_summary, eval_model
from scripts.run_imt_human_eval import ALL_TARGETS
from workflows.imt_scores import find_model_files


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate IMT audit results for the human-eval datasets")
    parser.add_argument("--targets", nargs="+", default=ALL_TARGETS, choices=ALL_TARGETS,
                        help=f"Target-model datasets to evaluate (default: all of {ALL_TARGETS})")
    parser.add_argument("--auditors", nargs="+", default=None,
                        help="Auditor preset keys to evaluate (default: all found on disk)")
    args = parser.parse_args()

    HUMAN_EVAL_EVAL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    combined_lines = []

    for target in args.targets:
        results_dir = HUMAN_EVAL_AUDIT_RESULTS_DIR / target
        labels_path = HUMAN_EVAL_DATA_DIR / f"{target}_dataset_with_labels.json"

        if not results_dir.exists():
            print(f"[WARN] No audit results dir for target '{target}': {results_dir}")
            continue

        model_files = find_model_files(results_dir, args.auditors)
        if not model_files:
            print(f"[WARN] No IMT result files found in {results_dir}")
            continue

        print(f"\n=== target: {target}  ({len(model_files)} auditor(s)) ===")
        out_dir = HUMAN_EVAL_EVAL_RESULTS_DIR / target
        labeled_out_dir = ROOT / "results" / "imt_audit_human_eval_with_label" / target
        out_dir.mkdir(parents=True, exist_ok=True)
        labeled_out_dir.mkdir(parents=True, exist_ok=True)

        all_results = {}
        for auditor, files in sorted(model_files.items()):
            result = eval_model(auditor, files, labels_path=labels_path, labeled_out_dir=labeled_out_dir)
            all_results[auditor] = result
            if not result:
                continue

            out = out_dir / f"{auditor}.json"
            with open(out, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            print(f"    -> {out.relative_to(ROOT)}")

        # Merge in previously cached per-auditor results (e.g. from an earlier run with
        # a different --auditors filter) so the summary always covers everything on disk.
        for cached in out_dir.glob("*.json"):
            auditor = cached.stem
            if auditor not in all_results:
                with open(cached, encoding="utf-8") as f:
                    all_results[auditor] = json.load(f)

        summary = build_summary(all_results)
        summary_path = out_dir / "summary.txt"
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(summary)
        print(f"  Summary: {summary_path.relative_to(ROOT)}")

        combined_lines.append(f"##### target: {target} #####\n")
        combined_lines.append(summary)
        combined_lines.append("\n")

    combined_path = HUMAN_EVAL_EVAL_RESULTS_DIR / "summary.txt"
    with open(combined_path, "w", encoding="utf-8") as f:
        f.write("\n".join(combined_lines))
    print(f"\nCombined summary: {combined_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
