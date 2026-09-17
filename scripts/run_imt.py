"""
scripts/run_imt.py
==================
Run the IMT audit workflow (workflows/singleturn_iu_audit_workflow.py) for one or all
model presets, against either the main DeceptionBench-style dataset or the
human-eval target-model datasets.

Usage:
    # Main dataset — run default model (azure_gpt4o), once
    python scripts/run_imt.py

    # Main dataset — run a specific model, 3 times
    python scripts/run_imt.py --model azure_claude_sonnet46 --runs 3

    # Main dataset — run ALL models once, skipping ones already done
    python scripts/run_imt.py --all --skip-existing

    # Human-eval datasets — run one auditor against all 3 target-model datasets
    python scripts/run_imt.py --dataset human_eval --model azure_gpt4o

    # Human-eval datasets — run ALL auditors against ALL 3 target-model datasets
    python scripts/run_imt.py --dataset human_eval --all

    # Human-eval datasets — restrict to specific target-model datasets
    python scripts/run_imt.py --dataset human_eval --all --targets claude_sonnet46 gpt4o

Output:
    Main dataset:  results/imt_audit/imt_audit_<model>_results.json          (single run)
                   results/imt_audit/imt_audit_<model>_run<k>.json           (multi-run)
    Human-eval:    results/imt_audit_human_eval/<target>/imt_audit_<auditor>_results.json
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import LLM_PRESETS
from project_paths import (
    DEEPSEEK_DATASET,
    HUMAN_EVAL_AUDIT_RESULTS_DIR,
    HUMAN_EVAL_TARGETS,
    IMT_AUDIT_RESULTS_DIR,
    model_dataset_path,
)

WORKFLOW = ROOT / "workflows" / "singleturn_iu_audit_workflow.py"
DEFAULT_MODEL = "azure_gpt4o"
DEFAULT_WORKERS = 8
ALL_TARGETS = HUMAN_EVAL_TARGETS


def _run_once(model: str, input_file: Path, output: Path, workers: int, label: str) -> bool:
    cmd = [
        sys.executable, str(WORKFLOW),
        "--model",   model,
        "--input",   str(input_file),
        "--output",  str(output),
        "--workers", str(workers),
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    print(f"\n{'='*60}")
    print(f"[RUN] {label}  ->  {output.relative_to(ROOT)}")
    t0 = time.time()
    ok = subprocess.run(cmd, cwd=str(ROOT), env=env).returncode == 0
    print(f"[{'OK' if ok else 'FAIL'}] {label}  ({time.time()-t0:.0f}s)")
    return ok


def _run_main_dataset(args) -> dict:
    models = list(LLM_PRESETS.keys()) if args.all else [args.model]
    IMT_AUDIT_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    results = {}
    for model in models:
        for run_i in range(1, args.runs + 1):
            if args.runs == 1:
                out = IMT_AUDIT_RESULTS_DIR / f"imt_audit_{model}_results.json"
            else:
                out = IMT_AUDIT_RESULTS_DIR / f"imt_audit_{model}_run{run_i}.json"

            key = f"{model} run{run_i}"
            if args.skip_existing and out.exists():
                print(f"[SKIP] {key}")
                results[key] = "skip"
                continue
            results[key] = "ok" if _run_once(model, DEEPSEEK_DATASET, out, args.workers, key) else "fail"
    return results


def _run_human_eval_dataset(args) -> dict:
    auditors = [m for m in LLM_PRESETS.keys() if m not in args.exclude] if args.all else [args.model]

    for target in args.targets:
        input_file = model_dataset_path(target)
        if not input_file.exists():
            print(f"[ERROR] Missing {input_file}.")
            sys.exit(1)

    results = {}
    for target in args.targets:
        input_file = model_dataset_path(target)
        out_dir = HUMAN_EVAL_AUDIT_RESULTS_DIR / target
        out_dir.mkdir(parents=True, exist_ok=True)

        for auditor in auditors:
            out = out_dir / f"imt_audit_{auditor}_results.json"
            key = f"{auditor} on {target}"
            if args.skip_existing and out.exists():
                print(f"[SKIP] {key}")
                results[key] = "skip"
                continue
            results[key] = "ok" if _run_once(auditor, input_file, out, args.workers, key) else "fail"
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run IMT audit for one or all models")
    parser.add_argument("--dataset", choices=("main", "human_eval"), default="main",
                        help="Which dataset to run against (default: main)")
    parser.add_argument("--model",  default=DEFAULT_MODEL,
                        help=f"Model preset key (default: {DEFAULT_MODEL})")
    parser.add_argument("--all",    action="store_true",
                        help="Run all models defined in config.LLM_PRESETS")
    parser.add_argument("--exclude", nargs="+", default=[],
                        help="[--dataset human_eval] Auditor preset keys to exclude when using --all")
    parser.add_argument("--targets", nargs="+", default=ALL_TARGETS, choices=ALL_TARGETS,
                        help=f"[--dataset human_eval] Target-model datasets to audit (default: all of {ALL_TARGETS})")
    parser.add_argument("--runs",   type=int, default=1,
                        help="[--dataset main] Number of repeated runs per model (default: 1)")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS,
                        help=f"Concurrent workers per run (default: {DEFAULT_WORKERS})")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip if output file already exists")
    args = parser.parse_args()

    if args.dataset == "human_eval":
        results = _run_human_eval_dataset(args)
    else:
        results = _run_main_dataset(args)

    print(f"\n{'='*60}  SUMMARY")
    for key, status in results.items():
        icon = {"ok": "✓", "skip": "–", "fail": "✗"}.get(status, "?")
        print(f"  {icon} {key}")

    if any(s == "fail" for s in results.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
