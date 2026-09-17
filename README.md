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
├── llm_interface/              # Model-agnostic LLM backends (3 classes, dispatched by protocol)
│   ├── base_llm.py                 # Abstract base + retry logic
│   ├── openai_llm.py               # Any OpenAI-protocol endpoint (OpenAI, Azure OpenAI,
│   │                                  Azure AI Foundry, OpenRouter, SiliconFlow, ...)
│   ├── anthropic_llm.py            # Anthropic Claude (direct API or Azure AI Foundry)
│   ├── google_genai_llm.py         # Google Gemini
│   └── __init__.py                 # create_llm() dispatcher (reads config["provider"])
├── workflows/                  # Pipeline entry points
│   ├── singleturn_iu_audit_workflow.py # Main single-turn DECOR pipeline
│   ├── multiturn_iu_sis_workflow.py    # Main multi-turn DECOR pipeline
│   ├── imt_scores.py               # Shared scoring/AUROC utilities
│   └── baseline_scores.py          # Shared baseline scoring utilities
├── baselines/                  # One subfolder per black-box baseline family, each with a
│   │                              single-turn runner and a multi-turn "_open_deception_round" twin
│   ├── simple_llm_judge/                   # Zero-shot + Few-shot
│   │   ├── run_simple_llm_judge.py, run_simple_llm_judge_open_deception_round.py
│   │   └── run_simple_llm_judge_fewshot.py
│   ├── deceptionbench/                     # DeceptionBench
│   │   └── run_deceptionbench.py, run_deceptionbench_open_deception_round.py
│   ├── cot_red_handed/                     # CoT Red-Handed
│   │   └── run_cot_red_handed.py, run_cot_red_handed_open_deception_round.py
│   └── scheming_judge/                     # Constitutional Monitor
│       └── run_scheming_judge.py, run_scheming_judge_open_deception_round.py
├── scripts/                    # Evaluation / analysis entry points
│   ├── run_imt.py                          # Run the single-turn pipeline (main or human-eval dataset)
│   ├── eval_imt.py                         # Compute AUROC/F1/etc. for DECOR results (main or human-eval dataset)
│   ├── eval_baselines.py                   # Evaluate all 5 single-turn baselines
│   ├── parse_open_deception.py             # Parse raw OpenDeception-C187 GPT-4o transcripts
│   └── eval_multiturn_open_deception.py    # Multi-turn AUROC evaluation (DECOR vs. one baseline)
├── data/
│   ├── singleturn/                             # One dataset per audited model (150 prompts x 4
│   │   │                                          pressure-framing (L2) variants = 600 scenarios)
│   │   ├── deepseek_r1/dataset_with_labels.json    # The paper's primary dataset
│   │   ├── gpt4o/dataset_with_labels.json          # Human-annotated GT extension to 3 more models
│   │   ├── claude_sonnet46/dataset_with_labels.json
│   │   └── qwen25_7b/dataset_with_labels.json
│   │       # Each file doubles as pipeline input and evaluation ground truth: every
│   │       # consumer reads only "prompt"/"responses", never "human_eval", so there's
│   │       # no separate no-label variant to keep in sync.
│   └── open_deception/
│       ├── OpenDeception-C187/                    # Raw source (Wu et al., 2025), trimmed to just
│       │   ├── transcripts/                           # the 50 GPT-4o transcripts the paper uses
│       │   └── examples.py                            # Per-scenario goal text (read as plain text)
│       └── open_deception_gpt4o_en.json           # generated — see "Multi-turn" below, not checked in
├── config.py                   # LLM presets and API configuration
├── project_paths.py            # Canonical path constants — every script/workflow above imports
│                                  its dataset paths from here rather than hardcoding them
├── .env.example                # Template for API keys / endpoints (copy to .env)
├── LICENSE                     # MIT — see "License" below for third-party data scope
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

```bash
cp .env.example .env
```

Then fill in only the keys for the providers you actually plan to use — `.env.example` documents every key `config.py` reads and which presets each one unlocks. You do **not** need every key: pick one text-generation provider and you can run the whole pipeline.

See "Reproducibility" below for how to pick a provider without access to the authors' own Azure deployment, and `config.py` for the full `LLM_PRESETS` registry.

---

## Reproducibility

### Determinism

- Every LLM call in the pipeline, baselines, and evaluation code uses `temperature=0.0` (`config.DEFAULT_TEMPERATURE`).
- All cross-validation and threshold-fitting (`workflows/imt_scores.py`) uses a fixed 5-fold stratified split with `random_state=42`.
- That said, **LLM APIs are not bit-exact reproducible even at temperature 0** — providers update model weights and sampling implementations over time without changing the model name, so re-running months later may not reproduce every individual audit verbatim. Aggregate metrics (AUROC, F1, etc. across the full 600-scenario / 47-dialogue evaluation sets) are stable in practice; do not expect a single audited item's score to match to the last decimal.

### Reproducing without the authors' Azure account

Most of `config.LLM_PRESETS` (`azure_*`) point at the authors' own Azure OpenAI / Azure AI Foundry deployment and will not work with your own Azure account unless you create deployments with matching names. If you don't have Azure access at all, use the **direct provider presets** instead — they need nothing but a normal API key from the vendor's own console:

| Preset | Vendor | Env var |
|---|---|---|
| `openai_gpt4o` | official OpenAI API | `OPENAI_API_KEY` |
| `anthropic_claude_sonnet46` | official Anthropic API | `ANTHROPIC_API_KEY` |
| `google_gemini25pro` / `google_gemini31pro` | official Google AI API | `GOOGLE_API_KEY` |

```bash
python workflows/singleturn_iu_audit_workflow.py --model openai_gpt4o
python scripts/eval_imt.py --all
```

works exactly like the `azure_gpt4o` examples throughout this README — every script takes `--model <preset>`, and the preset name is the only thing that changes.

### Using a model that isn't in `config.py`

`llm_interface/` has 3 provider classes, dispatched by `create_llm()` on `config["provider"]`:

- **`OpenAILLM`** — any endpoint that speaks the OpenAI chat-completions protocol: official OpenAI, Azure OpenAI Service, Azure AI Foundry (DeepSeek, Grok, ...), OpenRouter, SiliconFlow, or a self-hosted OpenAI-compatible server. Point it at a new vendor by setting `base_url` + `api_key` + `model_name` — no new code needed. OpenRouter in particular proxies most other vendors' models behind one OpenAI-compatible endpoint, so it's often the fastest way to try a model this repo doesn't have a preset for.
- **`AnthropicLLM`** — direct Anthropic API (`model_name` + `api_key`) or Azure AI Foundry (`azure_endpoint` + `deployment_name`).
- **`GoogleGenAILLM`** — direct Google Gemini API only (`model_name` + `api_key`); uses the native SDK because it needs the `safety_settings=BLOCK_NONE` override — adversarial deception-research prompts get filtered by Gemini's default safety policy otherwise.

To add a new preset, add a dict to `config.py` with the right `provider` + fields and give it a key in `LLM_PRESETS` — every script already accepts any preset key via `--model`.

---

## Running DECOR

### Single-turn (DeceptionBench-style scenarios)

```bash
# Run the pipeline for one model
python workflows/singleturn_iu_audit_workflow.py --model azure_gpt4o

# Evaluate AUROC / F1 / etc. against human labels
python scripts/eval_imt.py --all
```

### Multi-turn (OpenDeception)

```bash
# 1. Generate data/open_deception/open_deception_gpt4o_en.json from the raw transcripts
#    (deterministic; not checked in, so this is a one-time setup step)
python scripts/parse_open_deception.py

# 2. Run DECOR over the resulting dialogues
python workflows/multiturn_iu_sis_workflow.py
# -> results/open_deception/imt_multiturn_gpt4o.json (thought + response scores)

# 3. Evaluate against a baseline's round-level output (defaults to Zero-shot)
python scripts/eval_multiturn_open_deception.py --target response
python scripts/eval_multiturn_open_deception.py --target thought
```

### Baselines

```bash
python baselines/simple_llm_judge/run_simple_llm_judge.py            # Zero-shot
python baselines/simple_llm_judge/run_simple_llm_judge_fewshot.py    # Few-shot
python baselines/deceptionbench/run_deceptionbench.py                # DeceptionBench
python baselines/cot_red_handed/run_cot_red_handed.py                # CoT Red-Handed
python baselines/scheming_judge/run_scheming_judge.py                # Constitutional Monitor

python scripts/eval_baselines.py --list             # see available families
python scripts/eval_baselines.py --family simple_llm_judge_zeroshot --all
python scripts/eval_baselines.py --family cot_red_handed --all
python scripts/eval_baselines.py --family scheming_judge --all
```

Multi-turn counterparts (`baselines/<family>/run_*_open_deception_round.py`) run the same judges over `data/open_deception/open_deception_gpt4o_en.json` (run `scripts/parse_open_deception.py` first — see above), one round at a time, then aggregate per dialogue (matching the paper's Appendix C.1 protocol — cumulative context, max-pooled across turns). Evaluate any of them against DECOR with `scripts/eval_multiturn_open_deception.py --round-path <file> --target {thought,response}`; the evaluator auto-detects whether the file uses binary per-round labels (Zero-shot, DeceptionBench) or continuous per-round scores (CoT Red-Handed, Constitutional Monitor).

### Human-annotated ground truth for additional models

`data/singleturn/{gpt4o,claude_sonnet46,qwen25_7b}/` extend the human-annotated ground truth beyond the paper's DeepSeek-R1 set.

```bash
python scripts/run_imt.py --dataset human_eval --all
python scripts/eval_imt.py --dataset human_eval --help
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

---

## License

This repository's code is released under the [MIT License](LICENSE). That license does **not** cover `data/open_deception/OpenDeception-C187/`, which is redistributed from a third-party benchmark (Wu et al., 2025) under its own terms — attribution details for that directory are still being finalized.

## Citation

If you use DECOR in your research, please cite:

```bibtex
@inproceedings{decor2026,
  title     = {DECOR: Auditing LLM Deception via Information Manipulation Theory},
  author    = {},
  booktitle = {},
  year      = {2026}
}
```
