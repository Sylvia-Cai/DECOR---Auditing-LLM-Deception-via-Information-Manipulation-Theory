"""
IMT Audit Workflow
=====================
Two-agent pipeline for IMT-based deception detection:

  Agent 1: InformationUnitAnalyst  — segments prompt, extracts IUs with SIS scores
  Agent 2: IMTAuditor              — scores thought & response per IU on 4 IMT dimensions
                                     (quantity / quality / relation / manner)

No aggregation is performed here. Raw per-IU scores are preserved for downstream analysis.

Output structure per L2 entry
------------------------------
Each result["<l2_type>"]["imt_audit"] contains:
  params:
    topic, dimension, question, groundtruth, goal, l2_type
    prompt, thought, response          ← all inputs reproduced verbatim
  iu_analysis:
    instruction_segment                ← Agent 1 Step 1
    context_segment                    ← Agent 1 Step 1
    information_units                  ← Agent 1 Step 2 (list of IUs with SIS)
  thought_audit:
    iu_assessments: [{iu_id, iu_content, quantity, quality, relation, manner}, ...]
  response_audit:
    iu_assessments: [{iu_id, iu_content, quantity, quality, relation, manner}, ...]

Concurrency model
-----------------
All (item × l2_type) tasks are submitted to a single ThreadPoolExecutor.
Each task makes 3 LLM calls in sequence: IU analysis, thought audit, then response audit.
Tasks complete out of order because results are printed via as_completed over the shared pool.
Checkpoint: the full dataset is written to disk after every item finishes.
"""

import json
import sys
import argparse
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from config import DEFAULT_LLM_PRESET, LLM_PRESETS, get_llm_config
from llm_interface import create_llm
from agents.InformationUnitAnalyst import InformationUnitAnalyst
from agents.IMTAuditor import IMTAuditor
from project_paths import (
    DEEPSEEK_DATASET,
    IMT_AUDIT_RESULTS_DIR,
    ensure_project_dirs,
)

# ── Configuration ──────────────────────────────────────────────────────────────

# Each L2 task = 3 LLM calls run sequentially inside one worker:
# IU extraction, then thought audit, then response audit.
# Worker count therefore controls the main concurrency knob for outbound LLM requests.
MAX_WORKERS = 8

DATA_FILE = DEEPSEEK_DATASET
_RESULTS_DIR = IMT_AUDIT_RESULTS_DIR

# Default output filename — overridden at runtime when --timestamp is used.
OUTPUT_FILE = _RESULTS_DIR / "imt_audit_results.json"

# ── Thread-safe checkpoint writer ─────────────────────────────────────────────

_save_lock = threading.Lock()


def _public_llm_metadata(llm_config: Dict[str, Any]) -> Dict[str, Any]:
    """Return a safe subset of model/run metadata for result files."""
    keys = (
        "provider",
        "model_name",
        "deployment_name",
        "base_url",
        "azure_endpoint",
        "api_version",
        "temperature",
        "max_completion_tokens",
    )
    return {
        key: llm_config.get(key)
        for key in keys
        if llm_config.get(key) is not None
    }


def _save_checkpoint(data: List[Dict], path: Path) -> None:
    with _save_lock:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)


def _effective_worker_count(requested_workers: int, llm_config: Dict[str, Any]) -> int:
    provider = llm_config.get("provider")
    base_url = str(llm_config.get("base_url", "")).lower()
    model_name = str(llm_config.get("model_name", ""))

    # SiliconFlow Qwen models hit TPM limits easily under the default 8-way concurrency.
    if provider == "openai_compatible" and "siliconflow" in base_url:
        capped = min(requested_workers, 2)
        if capped < requested_workers:
            print(
                f"[INFO] Reducing workers from {requested_workers} to {capped} for "
                f"SiliconFlow model {model_name} to avoid TPM throttling."
            )
        return capped

    return requested_workers


# ── Per-L2 worker ──────────────────────────────────────────────────────────────

def _process_one_l2(
    iu_agent:    InformationUnitAnalyst,
    imt_agent:   IMTAuditor,
    llm_metadata: Dict[str, Any],
    topic:       str,
    dimension:   str,
    question:    str,
    groundtruth: str,
    goal:        str,
    l2_type:     str,
    content:     Dict[str, Any],
) -> Tuple[str, Optional[str]]:
    """
    Process one (item, l2_type) pair.  Writes results into *content* in-place.
    Returns (l2_type, error_message_or_None).

    Call order (all sequential — A2 depends on A1 output):
      1. InformationUnitAnalyst.process(prompt)
      2. IMTAuditor.process(prompt, information_units, thought, response)
         ↳ internally: thought_audit LLM call, then response_audit LLM call
    """
    prompt_text   = content.get("prompt", "")
    thought_text  = content.get("responses", {}).get("thought", "") or ""
    response_text = content.get("responses", {}).get("response", "") or ""

    try:
        # ── Agent 1: IU extraction ────────────────────────────────────────────
        iu_result = iu_agent.process(prompt_text)

        instruction_segment = iu_result.get("instruction_segment", "")
        context_segment     = iu_result.get("context_segment", "")
        information_units   = iu_result.get("information_units", [])

        # ── Agent 2: IMT audit ────────────────────────────────────────────────
        imt_result = imt_agent.process(
            prompt            = prompt_text,
            information_units = information_units,
            thought           = thought_text,
            response          = response_text,
        )

        # ── Write results ─────────────────────────────────────────────────────
        content["imt_audit"] = {
            "params": {
                "topic":       topic,
                "dimension":   dimension,
                "question":    question,
                "groundtruth": groundtruth,
                "goal":        goal,
                "l2_type":     l2_type,
                "llm":         llm_metadata,
                "prompt":      prompt_text,
                "thought":     thought_text,
                "response":    response_text,
            },
            "iu_analysis": {
                "instruction_segment": instruction_segment,
                "context_segment":     context_segment,
                "information_units":   information_units,
            },
            "thought_audit":  imt_result.get("thought_audit",  {}),
            "response_audit": imt_result.get("response_audit", {}),
        }
        return l2_type, None

    except Exception as exc:
        content["imt_audit"] = {"error": f"{type(exc).__name__}: {exc}"}
        return l2_type, f"{type(exc).__name__}: {exc}"


# ── Progress summary helper ────────────────────────────────────────────────────

def _summarise_audit(content: Dict[str, Any]) -> str:
    """Return a one-line summary of completed audit scores for console output."""
    audit = content.get("imt_audit", {})
    if "error" in audit:
        return f"ERROR: {audit['error']}"

    iu_count = len(audit.get("iu_analysis", {}).get("information_units", []))

    def _iu_scores(section_key: str) -> str:
        assessments = audit.get(section_key, {}).get("iu_assessments", [])
        if not assessments:
            return "—"
        totals = []
        for iu in assessments:
            dims = [iu.get(d, {}) for d in ("quantity", "quality", "relation", "manner")]
            scores = [d.get("score", -1) for d in dims if isinstance(d, dict)]
            valid  = [s for s in scores if s >= 0]
            totals.append(sum(valid) / len(valid) if valid else -1)
        valid_totals = [t for t in totals if t >= 0]
        mean = sum(valid_totals) / len(valid_totals) if valid_totals else -1
        return f"{mean:.2f}"

    return (
        f"IUs={iu_count} | "
        f"thought_avg={_iu_scores('thought_audit')} | "
        f"response_avg={_iu_scores('response_audit')}"
    )


def _is_complete_imt_audit(content: Dict[str, Any]) -> bool:
    """True only when an L2 task finished with IU extraction and both audits present."""
    audit = content.get("imt_audit")
    if not isinstance(audit, dict) or "error" in audit:
        return False

    information_units = (audit.get("iu_analysis") or {}).get("information_units") or []
    thought_assessments = (audit.get("thought_audit") or {}).get("iu_assessments") or []
    response_assessments = (audit.get("response_audit") or {}).get("iu_assessments") or []

    return bool(information_units and thought_assessments and response_assessments)


# ── Main workflow ──────────────────────────────────────────────────────────────

def run_workflow(
    input_file: Path,
    output_file: Path,
    llm_config: Dict[str, Any],
    overwrite: bool = False,
    max_workers: int = MAX_WORKERS,
) -> None:
    ensure_project_dirs()
    if not input_file.exists():
        print(f"[ERROR] Input file not found: {input_file}")
        return

    with open(input_file, "r", encoding="utf-8") as f:
        data: List[Dict] = json.load(f)

    # Load previously saved progress unless --overwrite was requested.
    if not overwrite and output_file.exists():
        with open(output_file, "r", encoding="utf-8") as f:
            saved = json.load(f)
        # Merge: overwrite data with saved progress to resume interrupted runs.
        saved_index = {
            (item.get("topic", ""), item.get("question", "")): item
            for item in saved
        }
        for i, item in enumerate(data):
            key = (item.get("topic", ""), item.get("question", ""))
            if key in saved_index:
                data[i] = saved_index[key]
        print(f"[INFO] Resuming from existing output: {output_file}")
    elif overwrite and output_file.exists():
        print(f"[INFO] --overwrite set; ignoring existing output: {output_file}")

    llm = create_llm(llm_config)
    llm_metadata = _public_llm_metadata(llm_config)
    iu_agent  = InformationUnitAnalyst({}, llm)
    imt_agent = IMTAuditor({}, llm)
    effective_workers = _effective_worker_count(max_workers, llm_config)

    total_items = len(data)
    print(f"\nIMT Audit Workflow v2")
    print(f"  Input  : {input_file}")
    print(f"  Output : {output_file}")
    print(f"  Items  : {total_items}")
    print(f"  Workers: {effective_workers}")
    print(f"  Provider: {llm_config.get('provider')}")
    print(f"  Model   : {llm_config.get('model_name')}")
    print(f"  LLM calls per L2 task: 3 (IU extraction + thought audit + response audit)\n")

    # ── Submit every (item × L2) task up-front so the worker pool is never idle ──
    item_l2_total: Dict[int, int] = {}
    item_l2_done:  Dict[int, int] = {}

    with ThreadPoolExecutor(max_workers=effective_workers) as executor:
        all_futures: Dict[Any, Tuple[int, str, str, Dict]] = {}

        for item_idx, item in enumerate(data):
            topic       = item.get("topic",       "Unknown")
            dimension   = item.get("dimension",   "")
            question    = item.get("question",    "")
            groundtruth = item.get("groundtruth", "")
            goal        = item.get("goal",        "")

            if "results" not in item:
                print(f"[{item_idx + 1}/{total_items}] {topic} — no results, skipping")
                continue

            # Skip only fully completed L2 tasks; retry prior errors or partial outputs.
            pending = {
                l2_type: content
                for l2_type, content in item["results"].items()
                if not _is_complete_imt_audit(content)
            }

            if not pending:
                print(f"[{item_idx + 1}/{total_items}] {topic} — already done, skipping")
                continue

            print(f"[{item_idx + 1}/{total_items}] {topic} | {dimension} ({len(pending)} L2 tasks queued)")

            item_l2_total[item_idx] = len(pending)
            item_l2_done[item_idx]  = 0

            for l2_type, content in pending.items():
                f = executor.submit(
                    _process_one_l2,
                    iu_agent, imt_agent, llm_metadata,
                    topic, dimension, question, groundtruth, goal,
                    l2_type, content,
                )
                all_futures[f] = (item_idx, topic, l2_type, content)

        # Collect results as they complete; checkpoint when each item is fully done.
        for future in as_completed(all_futures):
            item_idx, topic, l2_type, content = all_futures[future]
            done_type, err = future.result()
            tag = f"item {item_idx + 1:>3}/{total_items} | {topic} / {done_type}"
            if err:
                print(f"  [{tag}] ERROR: {err}")
            else:
                print(f"  [{tag}] {_summarise_audit(content)}")

            item_l2_done[item_idx] += 1
            if item_l2_done[item_idx] == item_l2_total[item_idx]:
                _save_checkpoint(data, output_file)
                print(f"  [checkpoint] item {item_idx + 1}/{total_items} ({topic}) saved")

    print(f"\n{'=' * 60}")
    print(f"Done. Results saved to: {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IMT Audit Workflow v2")
    parser.add_argument(
        "--model", default=DEFAULT_LLM_PRESET,
        help=(
            "Model selector. If it matches a config.LLM_PRESETS key, that preset is used; "
            "otherwise it is treated as model_name on the default preset."
        ),
    )
    parser.add_argument(
        "--input", type=Path, default=DEEPSEEK_DATASET,
        help="Path to the input JSON dataset.",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help=(
            "Path to the output JSON file. "
            "Defaults to results/imt_audit/imt_audit_{model}_results.json."
        ),
    )
    parser.add_argument(
        "--workers", type=int, default=MAX_WORKERS,
        help="Maximum number of concurrent (item x L2) tasks.",
    )
    parser.add_argument(
        "--overwrite", action="store_true", default=False,
        help="Ignore any existing output file and reprocess from scratch.",
    )
    args = parser.parse_args()

    if args.model in LLM_PRESETS:
        preset = args.model
        overrides = None
    else:
        preset = DEFAULT_LLM_PRESET
        overrides = {"model_name": args.model}

    # Derive output filename from model name when not explicitly specified.
    output_file = args.output or (
        _RESULTS_DIR / f"imt_audit_{args.model}_results.json"
    )

    llm_config = get_llm_config(
        preset=preset,
        overrides=overrides,
    )

    run_workflow(
        args.input,
        output_file,
        llm_config=llm_config,
        overwrite=args.overwrite,
        max_workers=args.workers,
    )
