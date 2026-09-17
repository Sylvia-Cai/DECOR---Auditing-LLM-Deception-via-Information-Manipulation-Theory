"""
scripts/run_imt.py
==================
Run the IMT audit workflow for one or all model presets.

Usage:
    # Run default model (azure_gpt4o), once
    python scripts/run_imt.py

    # Run a specific model, 3 times
    python scripts/run_imt.py --model azure_claude_sonnet46 --runs 3

    # Run ALL models once
    python scripts/run_imt.py --all

    # Skip runs whose output file already exists
    python scripts/run_imt.py --all --skip-existing

Output:
    results/imt_audit/imt_audit_<model>_results.json          (single run)
    results/imt_audit/imt_audit_<model>_run1.json  etc.       (multi-run)
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
from project_paths import DEFAULT_DATASET_NO_EVAL, IMT_AUDIT_RESULTS_DIR

WORKFLOW = ROOT / "workflows" / "imt_audit_workflow.py"
DEFAULT_MODEL = "azure_gpt4o"
DEFAULT_WORKERS = 8


def _run_once(model: str, output: Path, workers: int) -> bool:
    cmd = [
        sys.executable, str(WORKFLOW),
        "--model",   model,
        "--input",   str(DEFAULT_DATASET_NO_EVAL),
        "--output",  str(output),
        "--workers", str(workers),
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    print(f"\n{'='*60}")
    print(f"[RUN] {model}  ->  {output.name}")
    t0 = time.time()
    ok = subprocess.run(cmd, cwd=str(ROOT), env=env).returncode == 0
    print(f"[{'OK' if ok else 'FAIL'}] {model}  ({time.time()-t0:.0f}s)")
    return ok


def main() -> None:
    parser = argparse.ArgumentParser(description="Run IMT audit for one or all models")
    parser.add_argument("--model",  default=DEFAULT_MODEL,
                        help=f"Model preset key (default: {DEFAULT_MODEL})")
    parser.add_argument("--all",    action="store_true",
                        help="Run all models defined in config.LLM_PRESETS")
    parser.add_argument("--runs",   type=int, default=1,
                        help="Number of repeated runs per model (default: 1)")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS,
                        help=f"Concurrent workers per model (default: {DEFAULT_WORKERS})")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip if output file already exists")
    args = parser.parse_args()

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
            results[key] = "ok" if _run_once(model, out, args.workers) else "fail"

    print(f"\n{'='*60}  SUMMARY")
    for key, status in results.items():
        icon = {"ok": "✓", "skip": "–", "fail": "✗"}.get(status, "?")
        print(f"  {icon} {key}")

    if any(s == "fail" for s in results.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
