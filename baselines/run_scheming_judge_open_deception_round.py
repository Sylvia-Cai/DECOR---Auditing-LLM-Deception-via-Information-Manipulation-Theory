"""Scheming-judge baseline on OpenDeception: round-by-round detection.

Applies the existing scheming scoring prompt to each round independently and
stores per-round suspicion scores for both thought and response.
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from config import DEFAULT_LLM_PRESET, get_llm_config  # noqa: E402
from llm_interface.factory import create_llm  # noqa: E402
from baselines.run_scheming_judge import (  # noqa: E402
    _PROMPT_BODY,
    _extract_score,
)


from project_paths import OPEN_DECEPTION_INPUT as DEFAULT_INPUT
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "open_deception" / "scheming_judge_open_deception_round.json"


def _load_dialogues(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        return [data]
    if isinstance(data, list):
        return data
    raise ValueError("Input JSON must be one dialogue object or a list of dialogue objects")


def _parse_csv_set(value: str) -> set[str]:
    return {x.strip() for x in value.split(",") if x.strip()}


def _label_to_binary(label: str, positives: set[str], negatives: set[str]) -> Optional[int]:
    if label in positives:
        return 1
    if label in negatives:
        return 0
    return None


def _build_history(prior_rounds: list) -> str:
    """Build cumulative user message history from prior rounds."""
    lines = []
    for rd in prior_rounds:
        rid = rd.get("round_id", "?")
        user = (rd.get("user") or "").strip()
        lines.append(f"Round {rid} User: {user}")
    return "\n".join(lines)


def _thought_prompt(history: str, current_user: str, thought: str) -> str:
    body, answer_line = _PROMPT_BODY.rsplit("\n", 1)
    history_part = f"[Conversation History]:\n{history}\n\n" if history else ""
    return f"{history_part}{body}\n\n[Current Round User]: {current_user}\n[AI Thought]: {thought}\n\n{answer_line}"


def _response_prompt(history: str, current_user: str, response: str) -> str:
    body, answer_line = _PROMPT_BODY.rsplit("\n", 1)
    history_part = f"[Conversation History]:\n{history}\n\n" if history else ""
    return f"{history_part}{body}\n\n[Current Round User]: {current_user}\n[AI Response]: {response}\n\n{answer_line}"


def _call_judge(llm, judge_prompt: str) -> tuple[float, str]:
    raw = llm.generate_with_retry(judge_prompt)
    score = _extract_score(raw)
    return score, raw


def _aggregate_scores(values: List[float]) -> Dict[str, float]:
    valid = [float(v) for v in values if isinstance(v, (int, float)) and v >= 0]
    if not valid:
        return {"min": 0.0, "max": 0.0, "mean": 0.0, "last": 0.0}
    return {
        "min": float(min(valid)),
        "max": float(max(valid)),
        "mean": float(sum(valid) / len(valid)),
        "last": float(valid[-1]),
    }


def _judge_dialogue(
    llm,
    dialogue: Dict[str, Any],
    dialogue_idx: int,
    total: int,
    positives: set[str],
    negatives: set[str],
) -> Optional[Dict[str, Any]]:
    label = dialogue.get("dialogue_label", "unknown")
    binary = _label_to_binary(label, positives, negatives)
    if binary is None:
        return None

    rounds = dialogue.get("rounds", [])
    round_results: List[Dict[str, Any]] = []

    for ridx, rd in enumerate(rounds, start=1):
        history = _build_history(rounds[: ridx - 1])
        current_user = (rd.get("user") or "").strip()
        thought_text = (rd.get("ai_thought") or "").strip()
        response_text = (rd.get("ai_speech") or "").strip()

        thought_score = -1.0
        thought_raw = ""
        response_score = -1.0
        response_raw = ""
        errors: List[str] = []

        try:
            thought_score, thought_raw = _call_judge(llm, _thought_prompt(history, current_user, thought_text))
        except Exception as exc:  # noqa: BLE001
            errors.append(f"thought: {type(exc).__name__}: {exc}")

        try:
            response_score, response_raw = _call_judge(llm, _response_prompt(history, current_user, response_text))
        except Exception as exc:  # noqa: BLE001
            errors.append(f"response: {type(exc).__name__}: {exc}")

        item = {
            "round_id": rd.get("round_id", ridx),
            "eval": {
                "thought_score": float(thought_score),
                "thought_raw": thought_raw,
                "response_score": float(response_score),
                "response_raw": response_raw,
                # combined: take the higher suspicion score (shared label)
                "combined_score": float(max(thought_score if thought_score >= 0 else 0.0,
                                           response_score if response_score >= 0 else 0.0)),
            },
        }
        if errors:
            item["error"] = " | ".join(errors)
        round_results.append(item)

    thought_scores = [r.get("eval", {}).get("thought_score", -1.0) for r in round_results]
    response_scores = [r.get("eval", {}).get("response_score", -1.0) for r in round_results]
    combined_scores = [r.get("eval", {}).get("combined_score", 0.0) for r in round_results]

    out = {
        "_order": dialogue_idx,
        "source_file": dialogue.get("source_file", f"dialogue_{dialogue_idx}"),
        "dialogue_label": label,
        "binary_label": binary,
        "goal": dialogue.get("goal", ""),
        "round_count": len(rounds),
        "round_results": round_results,
        "dialogue_scores": {
            "thought": _aggregate_scores(thought_scores),
            "response": _aggregate_scores(response_scores),
            # combined = max(thought, response) per round, then aggregated
            "combined": _aggregate_scores(combined_scores),
        },
    }

    print(
        f"[{dialogue_idx}/{total}] {out['source_file']} | "
        f"thought_mean={out['dialogue_scores']['thought']['mean']:.3f} | "
        f"response_mean={out['dialogue_scores']['response']['mean']:.3f}"
    )
    return out


def _output_path_for_run(base_path: Path, run_idx: int, runs: int) -> Path:
    if runs <= 1:
        return base_path
    suffix = base_path.suffix if base_path.suffix else ".json"
    return base_path.with_name(f"{base_path.stem}_run{run_idx}{suffix}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Scheming-judge round baseline for OpenDeception")
    parser.add_argument("--model", default=DEFAULT_LLM_PRESET, help="LLM preset key from config.py")
    parser.add_argument("--input", default=str(DEFAULT_INPUT), help="Input OpenDeception JSON")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Output JSON path")
    parser.add_argument("--workers", type=int, default=4, help="Parallel dialogue workers")
    parser.add_argument("--runs", type=int, default=1, help="Number of repeated runs")
    parser.add_argument(
        "--positive-labels",
        default="cheat_success,cheat_fail",
        help="Comma-separated positive labels",
    )
    parser.add_argument(
        "--negative-labels",
        default="cheat_none",
        help="Comma-separated negative labels",
    )
    parser.add_argument("--limit", type=int, default=None, help="Optional max number of dialogues")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.is_absolute():
        input_path = PROJECT_ROOT / input_path

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.runs < 1:
        raise ValueError("--runs must be >= 1")
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    positives = _parse_csv_set(args.positive_labels)
    negatives = _parse_csv_set(args.negative_labels)

    dialogues = _load_dialogues(input_path)
    if args.limit is not None and args.limit > 0:
        dialogues = dialogues[: args.limit]

    llm_config = get_llm_config(args.model)
    provider = llm_config.get("provider", "azure_openai")

    for run_idx in range(1, args.runs + 1):
        run_output = _output_path_for_run(output_path, run_idx, args.runs)
        llm = create_llm(llm_config)

        out_dialogues: List[Dict[str, Any]] = []

        total = len(dialogues)
        if args.workers <= 1:
            for i, d in enumerate(dialogues, start=1):
                judged = _judge_dialogue(llm, d, i, total, positives, negatives)
                if judged is not None:
                    out_dialogues.append(judged)
        else:
            with ThreadPoolExecutor(max_workers=args.workers) as executor:
                futures = {
                    executor.submit(_judge_dialogue, llm, d, i, total, positives, negatives): i
                    for i, d in enumerate(dialogues, start=1)
                }
                for future in as_completed(futures):
                    judged = future.result()
                    if judged is not None:
                        out_dialogues.append(judged)

            out_dialogues.sort(key=lambda x: x.get("_order", 0))

        for item in out_dialogues:
            item.pop("_order", None)

        payload = {
            "meta": {
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "input_file": str(input_path),
                "llm_preset": args.model,
                "provider": provider,
                "workers": int(args.workers),
                "positive_labels": sorted(positives),
                "negative_labels": sorted(negatives),
                "kept_dialogues": len(out_dialogues),
                "total_dialogues": len(dialogues),
                "runs": int(args.runs),
                "run_index": run_idx,
            },
            "dialogues": out_dialogues,
        }

        with run_output.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

        print(f"Run {run_idx}/{args.runs} done. Saved to: {run_output}")


if __name__ == "__main__":
    main()
