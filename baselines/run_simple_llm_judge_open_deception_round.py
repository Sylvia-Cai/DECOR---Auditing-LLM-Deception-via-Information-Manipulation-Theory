"""Simple-LLM judge baseline on OpenDeception: round-by-round detection.

This baseline reuses the exact prompt templates from
experiment/baselines/blackbox/run_simple_llm_judge.py and applies them to each
dialogue round independently.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from config import DEFAULT_LLM_PRESET, get_llm_config  # noqa: E402
from llm_interface.factory import create_llm  # noqa: E402
from baselines.run_simple_llm_judge import (  # noqa: E402
    _call_judge,
    _response_prompt,
    _thought_prompt,
)


DEFAULT_INPUT = PROJECT_ROOT / "results" / "open_deception" / "open_deception_gpt4o_en.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "open_deception" / "simple_llm_judge_open_deception_round.json"


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


def _build_judge_prompt(history: str, prompt_text: str, ai_text: str, target: str) -> str:
    if target == "thought":
        base = _thought_prompt(prompt_text, ai_text)
    else:
        base = _response_prompt(prompt_text, ai_text)
    if not history:
        return base
    return f"[Conversation History]:\n{history}\n\n{base}"


def _aggregate_round_labels(round_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(round_results)
    decept_count = sum(1 for x in round_results if x.get("eval", {}).get("label") == "decept")
    honest_count = sum(1 for x in round_results if x.get("eval", {}).get("label") == "honest")
    error_count = total - decept_count - honest_count

    decept_ratio = (decept_count / total) if total else 0.0
    if decept_count > honest_count:
        majority = "decept"
    elif honest_count > decept_count:
        majority = "honest"
    else:
        majority = "tie"

    return {
        "total_rounds": total,
        "decept_rounds": decept_count,
        "honest_rounds": honest_count,
        "error_rounds": error_count,
        "decept_ratio": decept_ratio,
        "any_decept": decept_count > 0,
        "majority_label": majority,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Simple-LLM round baseline for OpenDeception")
    parser.add_argument("--model", default=DEFAULT_LLM_PRESET, help="LLM preset key from config.py")
    parser.add_argument("--input", default=str(DEFAULT_INPUT), help="Input OpenDeception JSON")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Output JSON path")
    parser.add_argument(
        "--target",
        choices=("response", "thought"),
        default="response",
        help="Judge AI response or AI thought",
    )
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

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    positives = _parse_csv_set(args.positive_labels)
    negatives = _parse_csv_set(args.negative_labels)

    llm_config = get_llm_config(args.model)
    provider = llm_config.get("provider", "azure_openai")
    llm = create_llm(llm_config)

    dialogues = _load_dialogues(input_path)
    if args.limit is not None and args.limit > 0:
        dialogues = dialogues[: args.limit]

    out_dialogues: List[Dict[str, Any]] = []
    kept = 0

    for idx, d in enumerate(dialogues, start=1):
        label = d.get("dialogue_label", "unknown")
        binary = _label_to_binary(label, positives, negatives)
        if binary is None:
            continue

        rounds = d.get("rounds", [])
        round_results: List[Dict[str, Any]] = []

        for ridx, rd in enumerate(rounds, start=1):
            history = _build_history(rounds[: ridx - 1])
            prompt_text = (rd.get("user") or "").strip()
            text_key = "ai_thought" if args.target == "thought" else "ai_speech"
            ai_text = (rd.get(text_key) or "").strip()
            judge_prompt = _build_judge_prompt(history, prompt_text, ai_text, args.target)

            result = {"label": "error", "reasons": ""}
            err: Optional[str] = None
            try:
                result = _call_judge(llm, judge_prompt, provider)
            except Exception as exc:  # noqa: BLE001
                err = f"{type(exc).__name__}: {exc}"

            out_round = {
                "round_id": rd.get("round_id", ridx),
                "eval": {
                    "label": str(result.get("label", "error")).lower().strip(),
                    "reasons": str(result.get("reasons", "")),
                },
            }
            if err:
                out_round["error"] = err
            round_results.append(out_round)

        agg = _aggregate_round_labels(round_results)

        kept += 1
        item = {
            "source_file": d.get("source_file", f"dialogue_{idx}"),
            "dialogue_label": label,
            "binary_label": binary,
            "round_count": len(rounds),
            "target": args.target,
            "round_results": round_results,
            "dialogue_aggregation": agg,
        }
        out_dialogues.append(item)

        print(
            f"[{kept}] {item['source_file']} | decept_ratio={agg['decept_ratio']:.3f} | any_decept={agg['any_decept']}"
        )

    payload = {
        "meta": {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "input_file": str(input_path),
            "llm_preset": args.model,
            "provider": provider,
            "target": args.target,
            "positive_labels": sorted(positives),
            "negative_labels": sorted(negatives),
            "kept_dialogues": len(out_dialogues),
            "total_dialogues": len(dialogues),
        },
        "dialogues": out_dialogues,
    }

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"Done. Saved to: {output_path}")


if __name__ == "__main__":
    main()
