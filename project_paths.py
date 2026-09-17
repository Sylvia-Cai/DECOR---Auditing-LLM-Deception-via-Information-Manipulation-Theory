from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent

DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
BASELINES_DIR = PROJECT_ROOT / "baselines"

IMT_AUDIT_RESULTS_DIR = RESULTS_DIR / "imt_audit"
BASELINE_RESULTS_DIR = RESULTS_DIR / "baselines"
EVAL_RESULTS_DIR = RESULTS_DIR / "eval"
FIGURES_DIR = RESULTS_DIR / "figures"

DEFAULT_DATASET_NO_EVAL = DATA_DIR / "deepseek_dataset_150_no_label.json"
DEFAULT_DATASET_HUMAN_EVAL = DATA_DIR / "deepseek_dataset_150_with_labels.json"

HUMAN_EVAL_XLSX = DATA_DIR / "human_eval_final_gt_202609_gpt_claude_qwen.xlsx"
HUMAN_EVAL_DATA_DIR = DATA_DIR / "human_eval_models"
HUMAN_EVAL_AUDIT_RESULTS_DIR = RESULTS_DIR / "imt_audit_human_eval"
HUMAN_EVAL_EVAL_RESULTS_DIR = EVAL_RESULTS_DIR / "imt_human_eval"

# Not used in this repo (no legacy data layout here); kept only because
# workflows/imt_audit_workflow.py falls back to it when the primary dataset
# path is missing.
LEGACY_DATA_DIR = DATA_DIR / "_legacy"


def resolve_data_file(primary: Path, legacy: Path) -> Path:
    if primary.exists():
        return primary
    return legacy


def ensure_project_dirs() -> None:
    for path in (
        DATA_DIR,
        RESULTS_DIR,
        SCRIPTS_DIR,
        BASELINES_DIR,
        IMT_AUDIT_RESULTS_DIR,
        BASELINE_RESULTS_DIR,
        EVAL_RESULTS_DIR,
        FIGURES_DIR,
    ):
        path.mkdir(parents=True, exist_ok=True)
