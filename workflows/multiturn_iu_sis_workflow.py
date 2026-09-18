"""Multi-turn IU-augmented IMT workflow.

Pipeline per dialogue:
  For each round t:
    1. Extract new IUs from user_message[t] → tag with source_round=t.
    2. Assign SIS to NEW IUs and accumulate them (no dedup) into cumulative list.
    3. Audit AI thought against cumulative IUs using user-only history context.
    4. Audit AI response against cumulative IUs using user-only history context.
    5. Compute round-level scores from IU assessments.
  Aggregate round scores to dialogue level.

Target modes:
  --target thought   : only thought audit (context = user turns only)
  --target response  : only response audit (context = user turns only)
  --target both      : both audits (default)

Key design decisions:
  - IU source: only user messages (not AI speech or thought).
  - IU scope: cumulative history from round 1 to current round.
  - IU update: incremental per round — extract from NEW user message only; append.
  - IU tagging: each IU records source_round (the round it was first extracted from).
  - SIS assignment on newly extracted IUs (τ in {1,2,3}).
  - Thought context: user-only history (no ai_speech), enforcing thought isolation.
  - Response context: user-only history (aligned with thought).
  - IUs accumulate with NO deduplication across rounds — a repeated commitment
    is scored again each time it recurs, rather than only once.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from agents.MultiTurnIUAuditor import MultiTurnIUAuditor  # noqa: E402
from config import DEFAULT_LLM_PRESET, get_llm_config  # noqa: E402
from llm_interface import create_llm  # noqa: E402
from project_paths import (  # noqa: E402
    OPEN_DECEPTION_EXAMPLES as OPENDECEPTION_EXAMPLES,
    OPEN_DECEPTION_INPUT as DEFAULT_INPUT,
)


DEFAULT_OUTPUT = (
    PROJECT_ROOT / "results" / "open_deception" / "imt_multiturn_gpt4o.json"
)


def _normalize_scenario_name(name: str) -> str:
    s = name.lower().strip()
    s = re.sub(r"\bfraud\d+\b", "fraud", s)
    s = re.sub(r"\bsalesman\d+\b", "salesman", s)
    s = re.sub(r"\s+", " ", s)
    return s


def _lookup_goal(goal_map: Dict[str, str], scenario_name: str) -> Optional[str]:
    goal = goal_map.get(scenario_name)
    if goal:
        return goal
    norm_target = _normalize_scenario_name(scenario_name)
    for k, v in goal_map.items():
        if _normalize_scenario_name(k) == norm_target:
            return v
    return None

_save_lock = threading.Lock()


# ── Data loading ──────────────────────────────────────────────────────────────

def _load_dialogues(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        return [data]
    if isinstance(data, list):
        return data
    raise ValueError("Input JSON must be a dialogue object or list of dialogue objects")


def _load_open_deception_goals() -> Dict[str, str]:
    if not OPENDECEPTION_EXAMPLES.exists():
        return {}
    text = OPENDECEPTION_EXAMPLES.read_text(encoding="utf-8")
    pattern = re.compile(r'\[\s*"([^"]+)"\s*,\s*"([^"]+)"\s*,\s*"([^"]+)"\s*,\s*"([^"]+)"\s*\]')
    out: Dict[str, str] = {}
    for m in pattern.finditer(text):
        ai_role, goal, _user_role, _start_msg = m.groups()
        out[ai_role] = goal
    return out


def _attach_goal_if_missing(dialogues: List[Dict[str, Any]]) -> None:
    goal_map = _load_open_deception_goals()
    if not goal_map:
        return
    for d in dialogues:
        if d.get("goal"):
            continue
        source = d.get("source_file")
        if not isinstance(source, str):
            continue
        scenario = Path(source).stem
        goal = _lookup_goal(goal_map, scenario)
        if goal:
            d["goal"] = goal


def _parse_csv_set(value: str) -> set:
    return {x.strip() for x in value.split(",") if x.strip()}


def _label_to_binary(
    label: str, positives: set, negatives: set
) -> Optional[int]:
    if label in positives:
        return 1
    if label in negatives:
        return 0
    return None


# ── Context builders ──────────────────────────────────────────────────────────

def _build_user_history(rounds: List[Dict[str, Any]], up_to_idx: int) -> str:
    """Cumulative user-only history from round 0 to up_to_idx (inclusive).

    Used as context for thought audit — no ai_speech to enforce isolation.
    """
    lines: List[str] = []
    for j in range(up_to_idx + 1):
        rd = rounds[j]
        rid = rd.get("round_id", j + 1)
        user_text = (rd.get("user") or "").strip()
        lines.append(f"Round {rid} User: {user_text}")
    return "\n".join(lines)


def _build_full_context(rounds: List[Dict[str, Any]], up_to_idx: int) -> str:
    """Cumulative full dialogue (user + AI) from round 0 to up_to_idx (inclusive).

    Used as context for response audit.
    """
    lines: List[str] = []
    for j in range(up_to_idx + 1):
        rd = rounds[j]
        rid = rd.get("round_id", j + 1)
        user_text = (rd.get("user") or "").strip()
        ai_speech = (rd.get("ai_speech") or "").strip()
        lines.append(f"Round {rid} User: {user_text}")
        lines.append(f"Round {rid} AI: {ai_speech}")
    return "\n".join(lines)


# ── Score computation ─────────────────────────────────────────────────────────

_DIMS = ("quantity", "quality", "relation", "manner")
_FIXED_ROUND_AGG = "mean"
_FIXED_DIALOGUE_AGG = "max"


def _iu_score(assessment: Dict[str, Any]) -> float:
    """Mean of 4 IMT dim scores for one IU. Returns -1.0 if all dims are invalid."""
    scores = []
    for dim in _DIMS:
        s = (assessment.get(dim) or {}).get("score", -1.0)
        if s >= 0.0:
            scores.append(s)
    if not scores:
        return -1.0
    return sum(scores) / len(scores)


def _compute_audit_round_scores(
    iu_assessments: List[Dict[str, Any]],
    information_units: List[Dict[str, Any]],
    current_round_id: int,
) -> Dict[str, float]:
    """Aggregate IU-level scores into round-level scores.

    Includes both unweighted statistics and SIS-weighted round mean.
    """
    sis_by_id: Dict[str, float] = {}
    for iu in information_units:
        iu_id = str(iu.get("iu_id"))
        sis_raw = iu.get("sis", 2)
        try:
            sis = float(sis_raw)
        except (TypeError, ValueError):
            sis = 2.0
        if sis <= 0.0:
            sis = 1.0
        src_raw = iu.get("source_round", 1)
        try:
            source_round = int(src_raw)
        except (TypeError, ValueError):
            source_round = 1

        # Direct SIS weighting without decay: weight = sis value (1, 2, or 3)
        sis_by_id[iu_id] = sis

    weighted_pairs: List[Tuple[float, float]] = []
    for a in iu_assessments:
        s = _iu_score(a)
        if s < 0.0:
            continue
        iu_id = str(a.get("iu_id", ""))
        if iu_id not in sis_by_id:
            continue  # hallucinated IU id — exclude from scoring
        w = sis_by_id[iu_id]
        weighted_pairs.append((s, w))

    if not weighted_pairs:
        return {
            "mean": 0.0,
            "max": 0.0,
        }

    total_w = sum(w for _, w in weighted_pairs)
    weighted_mean = (
        sum(s * w for s, w in weighted_pairs) / total_w if total_w > 0.0 else 0.0
    )

    return {
        "mean": weighted_mean,
        # SIS-weighted max: score of the most SIS-important (s*w) IU, normalised by its own weight
        "max": max(s * w for s, w in weighted_pairs) / max(w for _, w in weighted_pairs),
    }


def _aggregate_dialogue_scores(round_scores: List[float]) -> Dict[str, float]:
    """Aggregate round-level scores into dialogue-level scores."""
    if not round_scores:
        return {k: 0.0 for k in ("min", "max", "last")}
    return {
        "min": min(round_scores),
        "max": max(round_scores),
        "last": round_scores[-1],
    }


def _fixed_dialogue_score(dialogue_branch_scores: Dict[str, Any]) -> float:
    """Canonical score: round-level SIS-weighted mean, then dialogue-level max."""
    return float(
        ((dialogue_branch_scores or {}).get(_FIXED_ROUND_AGG) or {}).get(
            _FIXED_DIALOGUE_AGG, 0.0
        )
    )


# ── Per-dialogue processing ───────────────────────────────────────────────────

def _process_one_dialogue(
    idx: int,
    dialogue: Dict[str, Any],
    agent: MultiTurnIUAuditor,
    target: str,
) -> Tuple[int, Dict[str, Any], Optional[str]]:
    try:
        rounds = dialogue.get("rounds", [])
        round_results: List[Dict[str, Any]] = []
        dialogue_scores: Dict[str, Dict[str, Dict[str, float]]] = {}
        # Per-aggregation round score series for dialogue-level computation
        thought_scores: Dict[str, List[float]] = {
            k: [] for k in ("mean", "max")
        }
        response_scores: Dict[str, List[float]] = {
            k: [] for k in ("mean", "max")
        }

        # Cumulative IU list shared across rounds
        cumulative_ius: List[Dict[str, Any]] = []
        next_iu_id: int = 1

        for ridx, rd in enumerate(rounds):
            rid = rd.get("round_id", ridx + 1)
            user_message = (rd.get("user") or "").strip()

            # ── Build contexts ────────────────────────────────────────────────
            user_history = _build_user_history(rounds, ridx)

            # ── Step 1: Incremental IU extraction + SIS assignment (single call) ──
            prior_ius_snapshot = list(cumulative_ius)  # snapshot before extend, for dedup reference
            new_ius = agent.extract_and_score_new_ius(
                user_message=user_message,
                round_id=rid,
                next_iu_id=next_iu_id,
                prior_iu_count=len(cumulative_ius),
                prior_ius=prior_ius_snapshot,
            )
            cumulative_ius.extend(new_ius)
            next_iu_id += len(new_ius)

            r_item: Dict[str, Any] = {
                "round_id": rid,
                "new_ius_this_round": len(new_ius),
                "cumulative_iu_count": len(cumulative_ius),
                "information_units": list(cumulative_ius),  # snapshot at this round
            }

            # ── Step 2: Thought audit ─────────────────────────────────────────
            if target in ("thought", "both"):
                thought_text = (rd.get("ai_thought") or "").strip()
                t_audit = agent.audit_thought(
                    user_history_text=user_history,  # Cumulative user-only history (aligned with baselines)
                    information_units=cumulative_ius,
                    thought_text=thought_text,
                )
                t_round_scores = _compute_audit_round_scores(
                    t_audit.get("iu_assessments", []),
                    cumulative_ius,
                    current_round_id=rid,
                )
                t_audit["round_scores"] = t_round_scores
                r_item["user_history_text"] = user_history
                r_item["thought_audit"] = t_audit
                for agg_key in thought_scores:
                    thought_scores[agg_key].append(t_round_scores.get(agg_key, 0.0))

            # ── Step 3: Response audit ────────────────────────────────────────
            if target in ("response", "both"):
                response_text = (rd.get("ai_speech") or "").strip()
                r_audit = agent.audit_response(
                    context_text=user_history,  # Cumulative user-only history (aligned with baselines)
                    information_units=cumulative_ius,
                    response_text=response_text,
                )
                r_round_scores = _compute_audit_round_scores(
                    r_audit.get("iu_assessments", []),
                    cumulative_ius,
                    current_round_id=rid,
                )
                r_audit["round_scores"] = r_round_scores
                r_item["user_history_text_for_response"] = user_history
                r_item["response_audit"] = r_audit
                for agg_key in response_scores:
                    response_scores[agg_key].append(r_round_scores.get(agg_key, 0.0))

            round_results.append(r_item)

        # ── Dialogue-level aggregation ────────────────────────────────────────
        fixed_scores: Dict[str, float] = {}
        if target in ("thought", "both"):
            dialogue_scores["thought"] = {
                agg_key: _aggregate_dialogue_scores(series)
                for agg_key, series in thought_scores.items()
            }
            fixed_scores["thought"] = _fixed_dialogue_score(dialogue_scores["thought"])
        if target in ("response", "both"):
            dialogue_scores["response"] = {
                agg_key: _aggregate_dialogue_scores(series)
                for agg_key, series in response_scores.items()
            }
            fixed_scores["response"] = _fixed_dialogue_score(dialogue_scores["response"])

        out: Dict[str, Any] = {
            "source_file": dialogue.get("source_file", f"dialogue_{idx}"),
            "dialogue_label": dialogue.get("dialogue_label", "unknown"),
            "binary_label": dialogue.get("binary_label"),
            "goal": dialogue.get("goal"),
            "round_count": len(rounds),
            "total_ius_extracted": len(cumulative_ius),
            "round_results": round_results,
            "dialogue_scores": dialogue_scores,
            "fixed_scores": fixed_scores,
        }
        return idx, out, None

    except Exception as exc:  # noqa: BLE001
        out = {
            "source_file": dialogue.get("source_file", f"dialogue_{idx}"),
            "dialogue_label": dialogue.get("dialogue_label", "unknown"),
            "binary_label": dialogue.get("binary_label"),
            "round_count": len(dialogue.get("rounds", [])),
            "error": f"{type(exc).__name__}: {exc}",
        }
        return idx, out, out["error"]


# ── Persistence ───────────────────────────────────────────────────────────────

def _public_llm_metadata(llm_config: Dict[str, Any]) -> Dict[str, Any]:
    keys = (
        "provider", "model_name",
        "base_url",
        "temperature", "max_completion_tokens",
    )
    return {k: llm_config.get(k) for k in keys if llm_config.get(k) is not None}


def _save_checkpoint(output_file: Path, payload: Dict[str, Any]) -> None:
    with _save_lock:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with output_file.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)


def _output_path_for_run(base_output: Path, run_idx: int) -> Path:
    """Build run-specific output path.

    Examples:
      xxx_once.json -> xxx_run1.json / xxx_run2.json / ...
      xxx_run7.json -> xxx_run1.json / xxx_run2.json / ...
      xxx.json      -> xxx_run1.json / xxx_run2.json / ...
    """
    stem = base_output.stem
    if stem.endswith("_once"):
        new_stem = stem[:-5] + f"_run{run_idx}"
    elif re.search(r"_run\d+$", stem):
        new_stem = re.sub(r"_run\d+$", f"_run{run_idx}", stem)
    else:
        new_stem = f"{stem}_run{run_idx}"
    return base_output.with_name(new_stem + base_output.suffix)


# ── Workflow runner ───────────────────────────────────────────────────────────

def run_workflow(
    input_file: Path,
    output_file: Path,
    llm_preset: str,
    workers: int,
    target: str,
    positive_labels: set,
    negative_labels: set,
    limit: Optional[int],
) -> None:
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    dialogues_all = _load_dialogues(input_file)
    _attach_goal_if_missing(dialogues_all)
    if limit is not None and limit > 0:
        dialogues_all = dialogues_all[:limit]

    filtered: List[Dict[str, Any]] = []
    for d in dialogues_all:
        y = _label_to_binary(d.get("dialogue_label", "unknown"), positive_labels, negative_labels)
        if y is None:
            continue
        d = dict(d)
        d["binary_label"] = y
        filtered.append(d)

    llm_config = get_llm_config(llm_preset)
    llm = create_llm(llm_config)
    agent = MultiTurnIUAuditor({}, llm)

    print("Multi-turn IU-augmented IMT workflow")
    print(f"  Input       : {input_file}")
    print(f"  Output      : {output_file}")
    print(f"  Dialogues   : {len(filtered)} (filtered from {len(dialogues_all)})")
    print(f"  Workers     : {workers}")
    print(f"  Target      : {target}")
    print(f"  Positive    : {sorted(positive_labels)}")
    print(f"  Negative    : {sorted(negative_labels)}")

    results: List[Dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = {
            executor.submit(
                _process_one_dialogue,
                i,
                d,
                agent,
                target,
            ): (i, d)
            for i, d in enumerate(filtered)
        }

        for n, future in enumerate(as_completed(futures), start=1):
            idx, out, err = future.result()
            results.append(out)
            source = out.get("source_file", f"dialogue_{idx}")
            if err:
                print(f"[{n}/{len(filtered)}] ERROR {source} -> {err}")
            else:
                # Print a summary score for quick monitoring
                ds = out.get("dialogue_scores", {})
                if target in ("response", "both"):
                    r_max = ds.get("response", {}).get("mean", {}).get("max", 0.0)
                    print(
                        f"[{n}/{len(filtered)}] OK    {source} "
                        f"| ius={out.get('total_ius_extracted', 0)} "
                        f"| resp_mean→max={r_max:.3f}"
                    )
                else:
                    t_max = ds.get("thought", {}).get("mean", {}).get("max", 0.0)
                    print(
                        f"[{n}/{len(filtered)}] OK    {source} "
                        f"| ius={out.get('total_ius_extracted', 0)} "
                        f"| thought_mean→max={t_max:.3f}"
                    )

            payload = {
                "meta": {
                    "created_at": datetime.now().isoformat(timespec="seconds"),
                    "input_file": str(input_file),
                    "llm_preset": llm_preset,
                    "llm": _public_llm_metadata(llm_config),
                    "target": target,
                    "sis_version": "v3",
                    "iu_strategy": "plan_c_incremental_no_dedup_with_sis_v3",
                    "fixed_aggregation": {
                        "round_level": _FIXED_ROUND_AGG,
                        "dialogue_level": _FIXED_DIALOGUE_AGG,
                    },
                    "positive_labels": sorted(positive_labels),
                    "negative_labels": sorted(negative_labels),
                },
                "dialogues": sorted(results, key=lambda x: x.get("source_file", "")),
            }
            _save_checkpoint(output_file, payload)

    final_payload = {
        "meta": {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "input_file": str(input_file),
            "llm_preset": llm_preset,
            "llm": _public_llm_metadata(llm_config),
            "target": target,
            "sis_version": "v3",
            "iu_strategy": "plan_c_incremental_no_dedup_with_sis_v3",
            "fixed_aggregation": {
                "round_level": _FIXED_ROUND_AGG,
                "dialogue_level": _FIXED_DIALOGUE_AGG,
            },
            "positive_labels": sorted(positive_labels),
            "negative_labels": sorted(negative_labels),
        },
        "dialogues": sorted(results, key=lambda x: x.get("source_file", "")),
    }
    _save_checkpoint(output_file, final_payload)
    print(f"Done. Saved to: {output_file}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-turn IU-augmented IMT workflow")
    parser.add_argument("--input", default=str(DEFAULT_INPUT), help="Parsed OpenDeception JSON")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Output JSON path")
    parser.add_argument("--model", default=DEFAULT_LLM_PRESET, help="LLM preset key from config.py")
    parser.add_argument("--workers", type=int, default=1, help="Number of parallel dialogue workers")
    parser.add_argument(
        "--target",
        choices=("thought", "response", "both"),
        default="both",
        help="Which AI output to audit: thought | response | both",
    )
    parser.add_argument(
        "--positive-labels",
        default="cheat_success,cheat_fail",
        help="Comma-separated positive dialogue labels",
    )
    parser.add_argument(
        "--negative-labels",
        default="cheat_none",
        help="Comma-separated negative dialogue labels",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=1,
        help="Number of repeated runs. If >1, output file name is auto-expanded to _run1/_run2/...",
    )
    parser.add_argument("--limit", type=int, default=None, help="Optional max number of dialogues")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.is_absolute():
        input_path = PROJECT_ROOT / input_path

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path

    runs = max(1, int(args.runs))
    for i in range(1, runs + 1):
        run_output_path = output_path if runs == 1 else _output_path_for_run(output_path, i)
        if runs > 1:
            print(f"\n=== RUN {i}/{runs} -> {run_output_path} ===")
        run_workflow(
            input_file=input_path,
            output_file=run_output_path,
            llm_preset=args.model,
            workers=args.workers,
            target=args.target,
            positive_labels=_parse_csv_set(args.positive_labels),
            negative_labels=_parse_csv_set(args.negative_labels),
            limit=args.limit,
        )


if __name__ == "__main__":
    main()
