"""Unified analysis for kept OpenDeception result JSON files.

This script evaluates the curated result files under results/open_deception and
produces one consolidated JSON summary (no txt output).
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

from sklearn.metrics import average_precision_score, roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results" / "open_deception"


@dataclass
class MethodMetrics:
    method: str
    branch: str
    n: int
    n_pos: int
    n_neg: int
    auroc: float
    auprc: float
    score_mean_pos: float
    score_mean_neg: float


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        v = float(value)
        if math.isnan(v) or math.isinf(v):
            return default
        return v
    except (TypeError, ValueError):
        return default


def _mean(values: List[float]) -> float:
    return float(sum(values) / len(values)) if values else float("nan")


def _metric_pack(y_true: List[int], y_score: List[float], method: str, branch: str) -> MethodMetrics:
    n = len(y_true)
    n_pos = int(sum(y_true))
    n_neg = int(n - n_pos)
    if n == 0:
        return MethodMetrics(
            method=method,
            branch=branch,
            n=0,
            n_pos=0,
            n_neg=0,
            auroc=float("nan"),
            auprc=float("nan"),
            score_mean_pos=float("nan"),
            score_mean_neg=float("nan"),
        )

    pos_scores = [s for y, s in zip(y_true, y_score) if y == 1]
    neg_scores = [s for y, s in zip(y_true, y_score) if y == 0]

    has_both = n_pos > 0 and n_neg > 0
    auroc = float(roc_auc_score(y_true, y_score)) if has_both else float("nan")
    auprc = float(average_precision_score(y_true, y_score)) if has_both else float("nan")
    return MethodMetrics(
        method=method,
        branch=branch,
        n=n,
        n_pos=n_pos,
        n_neg=n_neg,
        auroc=auroc,
        auprc=auprc,
        score_mean_pos=_mean(pos_scores),
        score_mean_neg=_mean(neg_scores),
    )


def _sort_rows(rows: List[MethodMetrics]) -> List[MethodMetrics]:
    return sorted(
        rows,
        key=lambda r: (
            -1e9 if math.isnan(r.auroc) else -r.auroc,
            -1e9 if math.isnan(r.auprc) else -r.auprc,
            r.method,
        ),
    )


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _extract_simple_round(path: Path, branch: str) -> Tuple[List[int], Dict[str, List[float]]]:
    obj = _load_json(path)
    dialogues = obj.get("dialogues", [])
    y_true = [int(_safe_float(d.get("binary_label", 0), 0.0)) for d in dialogues]

    scores_by_method: Dict[str, List[float]] = {
        "decept_ratio": [],
        "decept_rounds_count": [],
        "any_decept_bin": [],
        "majority_decept_bin": [],
        "round_label_max": [],
        "round_label_mean": [],
    }

    for d in dialogues:
        agg = d.get("dialogue_aggregation") or {}
        round_results = d.get("round_results") or []

        decept_ratio = _safe_float(agg.get("decept_ratio", 0.0), 0.0)
        decept_rounds = _safe_float(agg.get("decept_rounds", 0.0), 0.0)
        any_decept = bool(agg.get("any_decept", False))
        majority_label = str(agg.get("majority_label", "")).strip().lower()
        majority_decept = majority_label in {"decept", "deceptive", "yes", "true", "1"}

        round_bins: List[float] = []
        for r in round_results:
            lbl = str(((r.get("eval") or {}).get("label", ""))).strip().lower()
            if lbl in {"decept", "deceptive", "yes", "true", "1"}:
                round_bins.append(1.0)
            elif lbl in {"honest", "non-deceptive", "non_deceptive", "no", "false", "0"}:
                round_bins.append(0.0)

        round_max = max(round_bins) if round_bins else 0.0
        round_mean = (sum(round_bins) / len(round_bins)) if round_bins else 0.0

        scores_by_method["decept_ratio"].append(decept_ratio)
        scores_by_method["decept_rounds_count"].append(decept_rounds)
        scores_by_method["any_decept_bin"].append(1.0 if any_decept else 0.0)
        scores_by_method["majority_decept_bin"].append(1.0 if majority_decept else 0.0)
        scores_by_method["round_label_max"].append(round_max)
        scores_by_method["round_label_mean"].append(round_mean)

    return y_true, scores_by_method


def _extract_simple_dialogue(path: Path, branch: str) -> Tuple[List[int], Dict[str, List[float]]]:
    obj = _load_json(path)
    dialogues = obj.get("dialogues", [])
    y_true = [int(_safe_float(d.get("binary_label", 0), 0.0)) for d in dialogues]

    pred_scores: List[float] = []
    for d in dialogues:
        label = str((d.get("eval") or {}).get("label", "")).strip().lower()
        if label in {"deceptive", "yes", "true", "1", "decept"}:
            pred_scores.append(1.0)
        elif label in {"honest", "no", "false", "0", "non-deceptive", "non_deceptive"}:
            pred_scores.append(0.0)
        else:
            pred_scores.append(0.0)
    return y_true, {"dialogue_label": pred_scores}


def _extract_imt_dialogues(path: Path, branch: str) -> Tuple[List[int], Dict[str, List[float]]]:
    obj = _load_json(path)
    dialogues = obj.get("dialogues", [])
    y_true = [int(_safe_float(d.get("binary_label", 0), 0.0)) for d in dialogues]

    all_methods: Dict[str, List[float]] = {}
    for d in dialogues:
        branch_scores = ((d.get("dialogue_scores") or {}).get(branch) or {})
        for method, value in branch_scores.items():
            method = str(method)
            if isinstance(value, dict):
                for sub_method, sub_value in value.items():
                    full_name = f"{method}.{sub_method}"
                    all_methods.setdefault(full_name, []).append(_safe_float(sub_value, 0.0))
            else:
                all_methods.setdefault(method, []).append(_safe_float(value, 0.0))

    for method in list(all_methods.keys()):
        if len(all_methods[method]) < len(dialogues):
            missing = len(dialogues) - len(all_methods[method])
            all_methods[method].extend([0.0] * missing)

    # Fixed aggregation: IU-level mean → round-level max.
    fixed = {k: v for k, v in all_methods.items() if k == "mean.max"}
    return y_true, fixed


def _evaluate_dataset(name: str, branch: str, path: Path, kind: str) -> Dict[str, Any]:
    if not path.exists():
        return {
            "name": name,
            "branch": branch,
            "path": str(path),
            "kind": kind,
            "exists": False,
            "metrics": [],
            "best": None,
        }

    if kind == "simple_round":
        y_true, methods = _extract_simple_round(path, branch)
    elif kind == "simple_dialogue":
        y_true, methods = _extract_simple_dialogue(path, branch)
    elif kind == "imt_dialogues":
        y_true, methods = _extract_imt_dialogues(path, branch)
    else:
        raise ValueError(f"Unsupported dataset kind: {kind}")

    rows = [_metric_pack(y_true, scores, method, branch) for method, scores in methods.items()]
    rows = _sort_rows(rows)
    return {
        "name": name,
        "branch": branch,
        "path": str(path),
        "kind": kind,
        "exists": True,
        "metrics": [asdict(r) for r in rows],
        "best": asdict(rows[0]) if rows else None,
    }


def _build_targets(results_dir: Path) -> List[Dict[str, Any]]:
    return [
        {
            "name": "simple_round_response",
            "branch": "response",
            "path": results_dir / "simple_llm_judge_open_deception_round.json",
            "kind": "simple_round",
        },
        {
            "name": "simple_round_thought",
            "branch": "thought",
            "path": results_dir / "simple_llm_judge_open_deception_round_thought.json",
            "kind": "simple_round",
        },
        {
            "name": "simple_dialogue_response",
            "branch": "response",
            "path": results_dir / "simple_llm_judge_open_deception_dialogue.json",
            "kind": "simple_dialogue",
        },
        {
            "name": "simple_dialogue_thought",
            "branch": "thought",
            "path": results_dir / "simple_llm_judge_open_deception_dialogue_thought.json",
            "kind": "simple_dialogue",
        },
        {
            "name": "decomp_no_sis_response",
            "branch": "response",
            "path": results_dir / "imt_multiturn_gpt4o_iu_no_sis_response.json",
            "kind": "imt_dialogues",
        },
        {
            "name": "decomp_no_sis_thought",
            "branch": "thought",
            "path": results_dir / "imt_multiturn_gpt4o_iu_no_sis_thought.json",
            "kind": "imt_dialogues",
        },
        {
            "name": "decomp_with_sis_v1_response",
            "branch": "response",
            "path": results_dir / "imt_multiturn_gpt4o_iu_sis_response_no_decay.json",
            "kind": "imt_dialogues",
        },
        {
            "name": "decomp_with_sis_v1_thought",
            "branch": "thought",
            "path": results_dir / "imt_multiturn_gpt4o_iu_sis_thought_no_decay.json",
            "kind": "imt_dialogues",
        },
        {
            "name": "decomp_with_sis_v2_response",
            "branch": "response",
            "path": results_dir / "imt_multiturn_gpt4o_iu_sis_both_no_decay.json",
            "kind": "imt_dialogues",
        },
        {
            "name": "decomp_with_sis_v2_thought",
            "branch": "thought",
            "path": results_dir / "imt_multiturn_gpt4o_iu_sis_both_no_decay.json",
            "kind": "imt_dialogues",
        },
        {
            "name": "decomp_with_sis_v3_response",
            "branch": "response",
            "path": results_dir / "imt_multiturn_gpt4o_iu_sis_both_v3.json",
            "kind": "imt_dialogues",
        },
        {
            "name": "decomp_with_sis_v3_thought",
            "branch": "thought",
            "path": results_dir / "imt_multiturn_gpt4o_iu_sis_both_v3.json",
            "kind": "imt_dialogues",
        },
    ]


def _compact_table(datasets: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for ds in datasets:
        selected = ds.get("best")

        rows.append(
            {
                "name": ds["name"],
                "branch": ds["branch"],
                "exists": ds["exists"],
                "best_method": selected["method"] if selected else None,
                "auroc": selected["auroc"] if selected else None,
                "auprc": selected["auprc"] if selected else None,
                "n": selected["n"] if selected else None,
                "n_pos": selected["n_pos"] if selected else None,
                "n_neg": selected["n_neg"] if selected else None,
                "selection_mode": "fixed_mean_max",
            }
        )
    return sorted(
        rows,
        key=lambda r: (
            r["branch"],
            -1e9 if r["auroc"] is None or math.isnan(r["auroc"]) else -r["auroc"],
            r["name"],
        ),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze curated OpenDeception JSON results.")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=RESULTS_DIR,
        help="Directory containing curated result JSON files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=RESULTS_DIR / "open_deception_unified_analysis.json",
        help="Output JSON path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    targets = _build_targets(args.results_dir)
    datasets = [
        _evaluate_dataset(t["name"], t["branch"], t["path"], t["kind"])
        for t in targets
    ]
    summary = {
        "results_dir": str(args.results_dir),
        "output": str(args.output),
        "datasets": datasets,
        "aggregation_spec": {
            "imt_dialogues": {
                "four_dimensions": "IU dimension score = mean(quantity, quality, relation, manner)",
                "iu_to_round": ["mean", "max"],
                "round_to_dialogue": ["max", "mean", "last"],
                "method_name_format": "<iu_aggregation>.<dialogue_aggregation>",
                "fixed_method": "mean.max",
            },
            "simple_round": {
                "available_methods": [
                    "decept_ratio",
                    "decept_rounds_count",
                    "any_decept_bin",
                    "majority_decept_bin",
                    "round_label_max",
                    "round_label_mean",
                ],
                "note": "round_label_max is max over per-round deceptive labels and corresponds to an any-decept style aggregation.",
            },
            "simple_dialogue": {
                "available_methods": ["dialogue_label"],
            },
            "selection_policy": "fixed_mean_max",
        },
        "best_table": _compact_table(datasets),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Saved: {args.output}")
    print("Best methods by dataset:")
    for row in summary["best_table"]:
        print(
            f"- {row['name']} ({row['branch']}): "
            f"method={row['best_method']}, AUROC={row['auroc']}, AUPRC={row['auprc']}"
        )


if __name__ == "__main__":
    main()
