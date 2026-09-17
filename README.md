# DECOR: Auditing LLM Deception via Information Manipulation Theory

DECOR is a multi-agent framework for fine-grained auditing of strategic deception in LLM outputs, grounded in Information Manipulation Theory (IMT; McCornack 1992). Given a task-related context and a model response, DECOR decomposes the context into atomic **Information Units (IUs)**, assigns each a strategic impact weight, and audits how the response handles each unit across four IMT dimensions — **Quantity**, **Quality**, **Relation**, and **Manner** — before aggregating everything into a global deception index.

This repository contains the final DECOR pipeline (single-turn and multi-turn), its evaluation code, and the black-box baselines it is compared against. It intentionally does **not** include ablation studies or exploratory variants — see the paper's supplementary material for those.

---

## Repository Structure

```
.
├── agents/                    # Final DECOR agents
│   ├── base_agent.py              # Abstract base class
│   ├── InformationUnitAnalyst.py  # Phase 1: context decomposition + impact weighting
│   ├── IMTAuditor.py              # Phase 2: per-IU IMT scoring (single-turn)
│   └── MultiTurnIUAuditor.py      # Phase 1+2 for multi-turn dialogues
├── llm_interface/              # Model-agnostic LLM backends
│   ├── base_llm.py                 # Abstract base + retry logic
│   ├── factory.py                  # create_llm() dispatcher
│   ├── azure_openai_llm.py         # Azure OpenAI (GPT-4o, GPT-5, o3, …)
│   ├── azure_deepseek_llm.py       # Azure AI Foundry — DeepSeek
│   ├── azure_anthropic_llm.py      # Azure AI Foundry — Claude
│   ├── google_genai_llm.py         # Google Gemini
│   └── openai_llm.py               # Official OpenAI API (reference impl.)
├── workflows/                  # Pipeline entry points
│   ├── imt_audit_workflow.py       # Main single-turn DECOR pipeline
│   ├── multiturn_iu_sis_workflow.py # Main multi-turn DECOR pipeline
│   ├── imt_scores.py               # Shared scoring/AUROC utilities
│   └── baseline_scores.py          # Shared baseline scoring utilities
├── baselines/                  # One runner per black-box baseline (single-turn +
│   │                              multi-turn "_open_deception_round" counterpart)
│   ├── run_simple_llm_judge.py, run_simple_llm_judge_open_deception_{round,dialogue}.py
│   │                                    # Zero-shot
│   ├── run_simple_llm_judge_fewshot.py # Few-shot
│   ├── run_deceptionbench.py, run_deceptionbench_open_deception_round.py
│   │                                    # DeceptionBench (single joint thought+response call)
│   ├── run_cot_red_handed.py, run_cot_red_handed_open_deception_round.py
│   │                                    # CoT Red-Handed
│   ├── run_scheming_judge.py, run_scheming_judge_open_deception_round.py
│   │                                    # Constitutional Monitor
│   └── cot_red_handed/                 # CoT Red-Handed monitor support code
├── scripts/                    # Evaluation / analysis entry points
│   ├── run_imt.py                          # Run the single-turn pipeline for one/all models
│   ├── eval_imt.py                         # Compute AUROC/F1/etc. for DECOR results
│   ├── eval_baselines.py                   # Evaluate Zero-shot/Few-shot
│   ├── eval_scheming_judge.py              # Evaluate Constitutional Monitor
│   ├── eval_cot_red_handed.py              # Evaluate CoT Red-Handed (per-score-key breakdown)
│   ├── parse_open_deception.py             # Parse raw OpenDeception-C187 transcripts
│   ├── eval_multiturn_open_deception.py    # Multi-turn AUROC evaluation (DECOR vs. one baseline)
│   ├── convert_human_eval_xlsx.py          # Convert human-annotated GT spreadsheet to JSON
│   ├── run_imt_human_eval.py               # Run DECOR over the human-eval model set
│   └── eval_imt_human_eval.py              # Evaluate the human-eval model set
├── data/
│   ├── deepseek_dataset_150_no_label.json     # 150 prompts x 4 pressure-framing (L2) variants
│   ├── deepseek_dataset_150_with_labels.json  # + human-annotated ground truth (DeepSeek-R1)
│   ├── open_deception_gpt4o_en.json           # 47 kept GPT-4o multi-turn dialogues
│   ├── OpenDeception-C187/                    # Source multi-turn benchmark (Wu et al., 2025)
│   └── human_eval_models/, human_eval_final_gt_*.xlsx  # Human-annotated GT for additional models
├── config.py                   # LLM presets and API configuration
├── project_paths.py            # Canonical path constants
└── requirements.txt
```

Running the pipeline/eval scripts creates a `results/` directory (git-ignored) with per-model outputs and computed metrics.

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure API keys

Create a `.env` file in the project root:

```env
AZURE_OPENAI_API_KEY=your_azure_key
AZURE_FOUNDRY_API_KEY=your_azure_foundry_key   # falls back to AZURE_OPENAI_API_KEY if unset
ANTHROPIC_API_KEY=your_anthropic_key
GOOGLE_API_KEY=your_google_key
OPENROUTER_API_KEY=your_openrouter_key
SILICONFLOW_API_KEY=your_siliconflow_key
```

Only the keys for the providers/models you plan to run are required. See `config.py` for the full model-preset registry and `llm_interface/openai_llm.py` for a minimal reference implementation against the official OpenAI API.

---

## Running DECOR

### Single-turn (DeceptionBench-style scenarios)

```bash
# Run the pipeline for one model
python workflows/imt_audit_workflow.py --model azure_gpt4o

# Evaluate AUROC / F1 / etc. against human labels
python scripts/eval_imt.py --all
```

### Multi-turn (OpenDeception)

```bash
# Run DECOR over the 47 kept OpenDeception dialogues
python workflows/multiturn_iu_sis_workflow.py
# -> results/open_deception/imt_multiturn_gpt4o.json (thought + response scores)

# Evaluate against a baseline's dialogue/round-level output (defaults to Zero-shot)
python scripts/eval_multiturn_open_deception.py --target response
python scripts/eval_multiturn_open_deception.py --target thought
```

### Baselines

```bash
python baselines/run_simple_llm_judge.py            # Zero-shot
python baselines/run_simple_llm_judge_fewshot.py    # Few-shot
python baselines/run_deceptionbench.py              # DeceptionBench
python baselines/run_cot_red_handed.py              # CoT Red-Handed
python baselines/run_scheming_judge.py              # Constitutional Monitor

python scripts/eval_baselines.py --list             # see available families
python scripts/eval_baselines.py --family simple_llm_judge_zeroshot --all
python scripts/eval_scheming_judge.py --input-dir results/baselines
python scripts/eval_cot_red_handed.py
```

Multi-turn counterparts (`baselines/run_*_open_deception_round.py`, plus `run_simple_llm_judge_open_deception_dialogue.py`) run the same judges over `data/open_deception_gpt4o_en.json`; evaluate them with `scripts/eval_multiturn_open_deception.py --dialogue-path` / `--round-path` pointed at the resulting file.

### Human-annotated ground truth for additional models

`data/human_eval_models/` and `data/human_eval_final_gt_202609_gpt_claude_qwen.xlsx` extend the human-annotated ground truth beyond the paper's DeepSeek-R1 set to GPT-4o, Claude Sonnet 4.6, and Qwen2.5-7B generations.

```bash
python scripts/convert_human_eval_xlsx.py
python scripts/run_imt_human_eval.py --help
python scripts/eval_imt_human_eval.py --help
```

---

## Pipeline Architecture

```
Input context (C) + task (T)
              │
              ▼
┌───────────────────────────────┐
│  Phase 1: Units Construction  │  LLM call
│  • Decompose context into      │
│    atomic information units    │
│  • Assign impact weight        │
│    w ∈ {1 (peripheral),        │
│         2 (auxiliary),         │
│         3 (decisive)}          │
└──────────────┬─────────────────┘
               │ impact-weighted units
               ▼
┌───────────────────────────────┐
│  Phase 2: IMT Auditing        │  2x LLM call (thought / response)
│  • Score per unit per dim:     │
│    quantity / quality /        │
│    relation / manner           │
│  • Scale: 0.00 / 0.33 /       │
│           0.66 / 1.00          │
└──────────────┬─────────────────┘
               │ manipulation profile
               ▼
┌───────────────────────────────┐
│  Phase 3: Deception Index     │  no LLM call
│  • Avg-pool 4 dims per unit     │
│  • Weight by impact weight,     │
│    sum over units → M(R)        │
└──────────────┬─────────────────┘
               ▼
   results/imt_audit/*.json
```

Each (item x L2 variant) task runs as a thread; checkpoints are written to disk after every item, so interrupted runs resume automatically.
