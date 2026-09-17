"""
scripts/run_imt_human_eval.py
==============================
Run the IMT audit workflow over the human-eval xlsx datasets (Claude-Sonnet-4.6,
GPT-4o, Qwen2.5-7B-Instruct target-model responses) — the same workflow used for
the deepseek dataset in scripts/run_imt.py, just pointed at different input files.

Run scripts/convert_human_eval_xlsx.py first to (re)generate the per-target-model
dataset files this script consumes.

Usage:
    # Run one auditor model against all 3 target-model datasets
    python scripts/run_imt_human_eval.py --model azure_gpt4o

    # Run ALL auditor models (config.LLM_PRESETS) against ALL 3 target-model datasets
    python scripts/run_imt_human_eval.py --all

    # Restrict to specific target-model datasets
    python scripts/run_imt_human_eval.py --all --targets claude_sonnet46 gpt4o

    # Skip runs whose output file already exists
    python scripts/run_imt_human_eval.py --all --skip-existing

Output:
    results/imt_audit_human_eval/<target>/imt_audit_<auditor>_results.json
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
from project_paths import HUMAN_EVAL_AUDIT_RESULTS_DIR, HUMAN_EVAL_DATA_DIR

WORKFLOW = ROOT / "workflows" / "imt_audit_workflow.py"
DEFAULT_MODEL = "azure_gpt4o"
DEFAULT_WORKERS = 8
ALL_TARGETS = ["claude_sonnet46", "gpt4o", "qwen25_7b"]


def _run_once(auditor: str, input_file: Path, output: Path, workers: int) -> bool:
    cmd = [
        sys.executable, str(WORKFLOW),
        "--model",   auditor,
        "--input",   str(input_file),
        "--output",  str(output),
        "--workers", str(workers),
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    print(f"\n{'='*60}")
    print(f"[RUN] auditor={auditor}  target={input_file.stem}  ->  {output.relative_to(ROOT)}")
    t0 = time.time()
    ok = subprocess.run(cmd, cwd=str(ROOT), env=env).returncode == 0
    print(f"[{'OK' if ok else 'FAIL'}] auditor={auditor} target={input_file.stem} ({time.time()-t0:.0f}s)")
    return ok


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run IMT audit for one or all auditor models over the human-eval datasets"
    )
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help=f"Auditor model preset key (default: {DEFAULT_MODEL})")
    parser.add_argument("--all", action="store_true",
                        help="Run all auditor models defined in config.LLM_PRESETS")
    parser.add_argument("--exclude", nargs="+", default=[],
                        help="Auditor preset keys to exclude when using --all")
    parser.add_argument("--targets", nargs="+", default=ALL_TARGETS, choices=ALL_TARGETS,
                        help=f"Target-model datasets to audit (default: all of {ALL_TARGETS})")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS,
                        help=f"Concurrent workers per run (default: {DEFAULT_WORKERS})")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip if output file already exists")
    args = parser.parse_args()

    auditors = [m for m in LLM_PRESETS.keys() if m not in args.exclude] if args.all else [args.model]

    for target in args.targets:
        no_label_path = HUMAN_EVAL_DATA_DIR / f"{target}_dataset_no_label.json"
        if not no_label_path.exists():
            print(f"[ERROR] Missing {no_label_path}. Run scripts/convert_human_eval_xlsx.py first.")
            sys.exit(1)

    results = {}
    for target in args.targets:
        input_file = HUMAN_EVAL_DATA_DIR / f"{target}_dataset_no_label.json"
        out_dir = HUMAN_EVAL_AUDIT_RESULTS_DIR / target
        out_dir.mkdir(parents=True, exist_ok=True)

        for auditor in auditors:
            out = out_dir / f"imt_audit_{auditor}_results.json"
            key = f"{auditor} on {target}"
            if args.skip_existing and out.exists():
                print(f"[SKIP] {key}")
                results[key] = "skip"
                continue
            results[key] = "ok" if _run_once(auditor, input_file, out, args.workers) else "fail"

    print(f"\n{'='*60}  SUMMARY")
    for key, status in results.items():
        icon = {"ok": "✓", "skip": "–", "fail": "✗"}.get(status, "?")
        print(f"  {icon} {key}")

    if any(s == "fail" for s in results.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
