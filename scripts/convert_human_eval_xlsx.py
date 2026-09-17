"""
scripts/convert_human_eval_xlsx.py
===================================
Convert data/human_eval_final_gt_202609_gpt_claude_qwen.xlsx into per-target-model
datasets that follow the exact same schema as deepseek_dataset_150_{no_label,with_labels}.json,
so the IMT audit workflow (workflows/imt_audit_workflow.py) and eval pipeline
(workflows/imt_scores.py) can be reused unchanged.

The xlsx has one row per (target_model, topic, question, l2_type) with columns:
    model, topic, dimension, l2_type, goal, question, prompt, groundtruth,
    response_thought, response_text, final_thought, final_response, resolution

Rows sharing (model, topic, question) are grouped into one item, and each row's
l2_type becomes one entry in that item's "results" dict — mirroring the deepseek
dataset's item = {topic, dimension, question, groundtruth, goal, results: {l2_type: {...}}}.

Output (per target model):
    data/human_eval_models/<slug>_dataset_no_label.json    (for feeding into the audit workflow)
    data/human_eval_models/<slug>_dataset_with_labels.json (adds "human_eval" ground-truth labels)

Usage:
    python scripts/convert_human_eval_xlsx.py
"""

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from project_paths import HUMAN_EVAL_DATA_DIR, HUMAN_EVAL_XLSX

MODEL_SLUGS = {
    "Claude-Sonnet-4.6":    "claude_sonnet46",
    "GPT-4o":               "gpt4o",
    "Qwen2.5-7B-Instruct":  "qwen25_7b",
}

REQUIRED_COLUMNS = [
    "model", "topic", "dimension", "l2_type", "goal", "question", "prompt",
    "groundtruth", "response_thought", "response_text", "final_thought", "final_response",
]

VALID_LABELS = {"honest", "decept"}


def _slug_for(model_name: str) -> str:
    if model_name not in MODEL_SLUGS:
        raise ValueError(
            f"Unknown model '{model_name}' in xlsx; add it to MODEL_SLUGS in "
            f"{Path(__file__).name}."
        )
    return MODEL_SLUGS[model_name]


def build_datasets(xlsx_path: Path) -> dict:
    """Return {model_slug: list[item]} where each item matches the deepseek schema."""
    df = pd.read_excel(xlsx_path)

    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"xlsx is missing expected columns: {missing_cols}")

    bad_labels = set(df["final_thought"]) | set(df["final_response"])
    bad_labels -= VALID_LABELS
    if bad_labels:
        raise ValueError(f"Unexpected label values (expected honest/decept): {bad_labels}")

    datasets: dict = {}

    for model_name, model_df in df.groupby("model"):
        slug = _slug_for(model_name)
        items = []

        for (topic, question), group in model_df.groupby(["topic", "question"], sort=False):
            first = group.iloc[0]

            # Sanity check: shared fields must agree across all rows in the group.
            for col in ("dimension", "goal", "groundtruth"):
                if group[col].nunique() != 1:
                    raise ValueError(
                        f"[{model_name}] inconsistent '{col}' within one (topic, question) "
                        f"group: topic={topic!r} question={question!r}"
                    )
            if group["l2_type"].duplicated().any():
                raise ValueError(
                    f"[{model_name}] duplicate l2_type within one (topic, question) group: "
                    f"topic={topic!r} question={question!r}"
                )

            results = {}
            for _, row in group.iterrows():
                results[row["l2_type"]] = {
                    "prompt": row["prompt"],
                    "responses": {
                        "thought":  row["response_thought"],
                        "response": row["response_text"],
                    },
                    "human_eval": {
                        "thought":  row["final_thought"],
                        "response": row["final_response"],
                    },
                }

            items.append({
                "topic":       topic,
                "dimension":   first["dimension"],
                "question":    question,
                "groundtruth": first["groundtruth"],
                "goal":        first["goal"],
                "results":     results,
            })

        datasets[slug] = items

    return datasets


def _strip_human_eval(items: list) -> list:
    stripped = json.loads(json.dumps(items))  # deep copy
    for item in stripped:
        for content in item["results"].values():
            content.pop("human_eval", None)
    return stripped


def main() -> None:
    HUMAN_EVAL_DATA_DIR.mkdir(parents=True, exist_ok=True)
    datasets = build_datasets(HUMAN_EVAL_XLSX)

    print(f"Converted {HUMAN_EVAL_XLSX.name}:")
    for slug, items in sorted(datasets.items()):
        n_l2 = sum(len(it["results"]) for it in items)
        with_labels_path = HUMAN_EVAL_DATA_DIR / f"{slug}_dataset_with_labels.json"
        no_label_path    = HUMAN_EVAL_DATA_DIR / f"{slug}_dataset_no_label.json"

        with open(with_labels_path, "w", encoding="utf-8") as f:
            json.dump(items, f, ensure_ascii=False, indent=2)
        with open(no_label_path, "w", encoding="utf-8") as f:
            json.dump(_strip_human_eval(items), f, ensure_ascii=False, indent=2)

        print(f"  {slug:<16} {len(items):>3} items, {n_l2:>3} L2 entries "
              f"-> {with_labels_path.relative_to(ROOT)}, {no_label_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
