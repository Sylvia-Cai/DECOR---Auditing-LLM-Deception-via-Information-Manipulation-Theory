import os
from copy import deepcopy
from typing import Any, Dict, Optional

from dotenv import load_dotenv

load_dotenv()

DEFAULT_TEMPERATURE = 0.0
DEFAULT_MAX_TOKENS = 10000

_AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY", "")
_AZURE_FOUNDRY_API_KEY = os.getenv("AZURE_FOUNDRY_API_KEY", _AZURE_OPENAI_API_KEY)

_AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "https://YOUR-RESOURCE.openai.azure.com/")

_AZURE_BASE = {
    "provider": "azure_openai",
    "api_key": _AZURE_OPENAI_API_KEY,
    "azure_endpoint": _AZURE_OPENAI_ENDPOINT,
    "api_version": "2024-12-01-preview",
    "temperature": DEFAULT_TEMPERATURE,
    "max_completion_tokens": DEFAULT_MAX_TOKENS,
}

AZURE_GPT4O_CONFIG = {
    **_AZURE_BASE,
    "deployment_name": "gpt-4o",
    "model_name": "gpt-4o",
}

AZURE_GPT5_CONFIG = {
    **_AZURE_BASE,
    "deployment_name": "gpt-5",
    "model_name": "gpt-5",
    "skip_temperature": True,   # gpt-5 only supports default temperature
}

AZURE_GPT54_CONFIG = {
    **_AZURE_BASE,
    "deployment_name": "gpt-5.4",
    "model_name": "gpt-5.4",
}

# o3 is a reasoning model: temperature must be omitted from the API call.
AZURE_O3_CONFIG = {
    **_AZURE_BASE,
    "deployment_name": "o3",
    "model_name": "o3",
    "skip_temperature": True,   # OpenAILLM will omit temperature for this model
}

# o4-mini is a reasoning model: temperature must be omitted from the API call.
AZURE_O4_MINI_CONFIG = {
    **_AZURE_BASE,
    "deployment_name": "o4-mini",
    "model_name": "o4-mini",
    "skip_temperature": True,
}

_AZURE_FOUNDRY_ENDPOINT = os.getenv("AZURE_FOUNDRY_ENDPOINT", "https://YOUR-RESOURCE.services.ai.azure.com/openai/v1/")

_AZURE_FOUNDRY_OPENAI_BASE = {
    "provider": "openai_compatible",
    "api_key": _AZURE_FOUNDRY_API_KEY,
    "base_url": _AZURE_FOUNDRY_ENDPOINT,
    "temperature": DEFAULT_TEMPERATURE,
    "max_completion_tokens": DEFAULT_MAX_TOKENS,
}

_SILICONFLOW_BASE = {
    "provider": "openai_compatible",
    "api_key": os.getenv("SILICONFLOW_API_KEY", ""),
    "base_url": os.getenv("SILICONFLOW_BASE_URL", "https://api.siliconflow.cn/v1"),
    "temperature": DEFAULT_TEMPERATURE,
    "max_completion_tokens": DEFAULT_MAX_TOKENS,
}

SILICONFLOW_QWEN36_27B_CONFIG = {
    **_SILICONFLOW_BASE,
    "model_name": "Qwen/Qwen3.6-27B",
}

SILICONFLOW_QWEN35_122B_A10B_CONFIG = {
    **_SILICONFLOW_BASE,
    "model_name": "Qwen/Qwen3.5-122B-A10B",
}

SILICONFLOW_QWEN3_235B_A22B_2507_CONFIG = {
    **_SILICONFLOW_BASE,
    "model_name": "Qwen/Qwen3-235B-A22B-Instruct-2507",
}

AZURE_GROK41_FAST_NON_REASONING_CONFIG = {
    **_AZURE_FOUNDRY_OPENAI_BASE,
    "deployment_name": "grok-4-1-fast-nonrea",
    "model_name": "grok-4-1-fast-non-reasoning",
}

AZURE_GROK41_FAST_REASONING_CONFIG = {
    **_AZURE_FOUNDRY_OPENAI_BASE,
    "deployment_name": "grok-4-1-fast-reasoning",
    "model_name": "grok-4-1-fast-reasoning",
    "skip_temperature": True,
}

AZURE_GROK420_REASONING_CONFIG = {
    **_AZURE_FOUNDRY_OPENAI_BASE,
    "deployment_name": "grok-4-20",
    "model_name": "grok-4-20-reasoning",
    "skip_temperature": True,
}

AZURE_GROK420_NON_REASONING_CONFIG = {
    **_AZURE_FOUNDRY_OPENAI_BASE,
    "deployment_name": "grok-4-20-non-reasoning",
    "model_name": "grok-4-20-non-reasoning",
}

# Keep a backward-compatible alias pointing at gpt-5 (the original default).
AZURE_OPENAI_CONFIG = AZURE_GPT5_CONFIG

AZURE_DEEPSEEK_CONFIG = {
    "provider": "azure_deepseek",
    "api_key": _AZURE_FOUNDRY_API_KEY,
    # Azure AI Foundry OpenAI-compatible endpoint (openai.OpenAI, not AzureOpenAI)
    "azure_endpoint": _AZURE_FOUNDRY_ENDPOINT,
    "deployment_name": "DeepSeek-V4-Flash-0731",
    "model_name": "DeepSeek-V4-Flash-0731",
    "temperature": DEFAULT_TEMPERATURE,
    "max_completion_tokens": DEFAULT_MAX_TOKENS,
    # This Foundry deployment doesn't understand the newer max_completion_tokens/
    # top_p params; falls back to the classic Chat Completions param shape.
    "legacy_max_tokens": True,
}

AZURE_DEEPSEEK_V31_CONFIG = {
    **AZURE_DEEPSEEK_CONFIG,
    "deployment_name": "DeepSeek-V4-Pro",
    "model_name": "DeepSeek-V4-Pro",
}

AZURE_DEEPSEEK_V32_CONFIG = {
    **AZURE_DEEPSEEK_CONFIG,
    "deployment_name": "DeepSeek-V3.2",
    "model_name": "DeepSeek-V3.2",
}

_OPENROUTER_BASE = {
    "provider": "openrouter",
    "api_key": os.getenv("OPENROUTER_API_KEY", ""),
    "base_url": "https://openrouter.ai/api/v1",
    "default_temperature": DEFAULT_TEMPERATURE,
    "default_max_tokens": DEFAULT_MAX_TOKENS,
}

OPENROUTER_GROK4_CONFIG = {
    **_OPENROUTER_BASE,
    "model_name": "x-ai/grok-4.3",
}

OPENROUTER_QWEN3_CONFIG = {
    **_OPENROUTER_BASE,
    "model_name": "qwen/qwen3-235b-a22b",
}

_GOOGLE_BASE = {
    "provider": "google_genai",
    "api_key": os.getenv("GOOGLE_API_KEY", ""),
    "temperature": DEFAULT_TEMPERATURE,
    "max_completion_tokens": DEFAULT_MAX_TOKENS,
}

GOOGLE_GEMINI25_CONFIG = {
    **_GOOGLE_BASE,
    "model_name": "gemini-2.5-pro",
}

GOOGLE_GEMINI31_CONFIG = {
    **_GOOGLE_BASE,
    "model_name": "gemini-3.1-pro-preview",
}

_ANTHROPIC_BASE = {
    "provider": "azure_anthropic",
    "api_key": os.getenv("AZURE_ANTHROPIC_API_KEY", ""),
    "azure_endpoint": os.getenv("AZURE_ANTHROPIC_ENDPOINT", "https://YOUR-RESOURCE.services.ai.azure.com/anthropic/"),
    "max_completion_tokens": DEFAULT_MAX_TOKENS,
}

# Direct (non-Azure) Anthropic API — works with a plain console.anthropic.com key,
# no Azure account needed. Prefer this preset when reproducing without Azure access.
_ANTHROPIC_DIRECT_BASE = {
    "provider": "anthropic",
    "api_key": os.getenv("ANTHROPIC_API_KEY", ""),
    "max_completion_tokens": DEFAULT_MAX_TOKENS,
}

AZURE_CLAUDE_SONNET46_CONFIG = {
    **_ANTHROPIC_BASE,
    "deployment_name": "claude-sonnet-4-6",
    "model_name": "claude-sonnet-4-6",
}

AZURE_CLAUDE_OPUS46_CONFIG = {
    **_ANTHROPIC_BASE,
    "deployment_name": "claude-opus-4-6",
    "model_name": "claude-opus-4-6",
}

AZURE_CLAUDE_OPUS47_CONFIG = {
    **_ANTHROPIC_BASE,
    "deployment_name": "claude-opus-4-7",
    "model_name": "claude-opus-4-7",
}

ANTHROPIC_CLAUDE_SONNET46_CONFIG = {
    **_ANTHROPIC_DIRECT_BASE,
    "model_name": "claude-sonnet-4-6",
}

# Direct (official) OpenAI API — works with a plain platform.openai.com key,
# no Azure account needed. Prefer this preset when reproducing without Azure access.
OPENAI_GPT4O_CONFIG = {
    "provider": "openai",
    "api_key": os.getenv("OPENAI_API_KEY", ""),
    "model_name": "gpt-4o",
    "temperature": DEFAULT_TEMPERATURE,
    "max_completion_tokens": DEFAULT_MAX_TOKENS,
}

LLM_PRESETS = {
    # Direct provider APIs — no Azure account needed, just the vendor's own API key.
    # Use these to reproduce without access to the authors' Azure deployment.
    "openai_gpt4o":          OPENAI_GPT4O_CONFIG,
    "anthropic_claude_sonnet46": ANTHROPIC_CLAUDE_SONNET46_CONFIG,
    "google_gemini25pro":    GOOGLE_GEMINI25_CONFIG,
    "google_gemini31pro":    GOOGLE_GEMINI31_CONFIG,
    # Azure OpenAI — GPT series
    "azure_gpt4o":          AZURE_GPT4O_CONFIG,
    "azure_gpt5":           AZURE_GPT5_CONFIG,
    "azure_gpt54":          AZURE_GPT54_CONFIG,
    "azure_o3":             AZURE_O3_CONFIG,
    "azure_o4mini":         AZURE_O4_MINI_CONFIG,
    "azure_grok41_fast_nonreasoning": AZURE_GROK41_FAST_NON_REASONING_CONFIG,
    "azure_grok41_fast_reasoning":    AZURE_GROK41_FAST_REASONING_CONFIG,
    "azure_grok420_nonreasoning":     AZURE_GROK420_NON_REASONING_CONFIG,
    "azure_grok420_reasoning":        AZURE_GROK420_REASONING_CONFIG,
    # Azure — DeepSeek
    "azure_deepseek_r1":    AZURE_DEEPSEEK_CONFIG,
    "azure_deepseek_v31":   AZURE_DEEPSEEK_V31_CONFIG,
    "azure_deepseek_v32":   AZURE_DEEPSEEK_V32_CONFIG,
    # Azure — Anthropic Claude
    "azure_claude_sonnet46": AZURE_CLAUDE_SONNET46_CONFIG,
    "azure_claude_opus46":   AZURE_CLAUDE_OPUS46_CONFIG,
    "azure_claude_opus47":   AZURE_CLAUDE_OPUS47_CONFIG,
    # Other providers
    "openrouter_grok4":     OPENROUTER_GROK4_CONFIG,
    "openrouter_qwen3":     OPENROUTER_QWEN3_CONFIG,
    "siliconflow_qwen36_27b": SILICONFLOW_QWEN36_27B_CONFIG,
    "siliconflow_qwen35_122b_a10b": SILICONFLOW_QWEN35_122B_A10B_CONFIG,
    "siliconflow_qwen3_235b_a22b_2507": SILICONFLOW_QWEN3_235B_A22B_2507_CONFIG,
}

DEFAULT_LLM_PRESET = os.getenv("DECEPTION_LLM_PRESET", "azure_gpt4o")


def normalize_llm_config(config: Dict[str, Any]) -> Dict[str, Any]:
    normalized = deepcopy(config)
    if "temperature" in normalized and "default_temperature" not in normalized:
        normalized["default_temperature"] = normalized["temperature"]
    if "default_temperature" in normalized and "temperature" not in normalized:
        normalized["temperature"] = normalized["default_temperature"]
    if "max_completion_tokens" in normalized and "default_max_tokens" not in normalized:
        normalized["default_max_tokens"] = normalized["max_completion_tokens"]
    if "default_max_tokens" in normalized and "max_completion_tokens" not in normalized:
        normalized["max_completion_tokens"] = normalized["default_max_tokens"]
    return normalized


def get_llm_config(
    preset: Optional[str] = None,
    overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    preset_name = preset or DEFAULT_LLM_PRESET
    if preset_name not in LLM_PRESETS:
        choices = ", ".join(sorted(LLM_PRESETS))
        raise ValueError(f"Unknown LLM preset '{preset_name}'. Available presets: {choices}")

    config = normalize_llm_config(LLM_PRESETS[preset_name])
    if overrides:
        for key, value in overrides.items():
            if value is not None:
                config[key] = value
    return normalize_llm_config(config)


# Convenience instance using the default preset; consumed by legacy baselines and tests.
LLM_CONFIG = get_llm_config()