import os
from copy import deepcopy
from typing import Any, Dict, Optional

from dotenv import load_dotenv

load_dotenv()

DEFAULT_TEMPERATURE = 0.0
DEFAULT_MAX_TOKENS = 10000

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

_ANTHROPIC_DIRECT_BASE = {
    "provider": "anthropic",
    "api_key": os.getenv("ANTHROPIC_API_KEY", ""),
    "max_completion_tokens": DEFAULT_MAX_TOKENS,
}

ANTHROPIC_CLAUDE_SONNET46_CONFIG = {
    **_ANTHROPIC_DIRECT_BASE,
    "model_name": "claude-sonnet-4-6",
}

OPENAI_GPT4O_CONFIG = {
    "provider": "openai",
    "api_key": os.getenv("OPENAI_API_KEY", ""),
    "model_name": "gpt-4o",
    "temperature": DEFAULT_TEMPERATURE,
    "max_completion_tokens": DEFAULT_MAX_TOKENS,
}

LLM_PRESETS = {
    "openai_gpt4o":          OPENAI_GPT4O_CONFIG,
    "anthropic_claude_sonnet46": ANTHROPIC_CLAUDE_SONNET46_CONFIG,
    "google_gemini25pro":    GOOGLE_GEMINI25_CONFIG,
    "google_gemini31pro":    GOOGLE_GEMINI31_CONFIG,
    "openrouter_grok4":     OPENROUTER_GROK4_CONFIG,
    "openrouter_qwen3":     OPENROUTER_QWEN3_CONFIG,
    "siliconflow_qwen36_27b": SILICONFLOW_QWEN36_27B_CONFIG,
    "siliconflow_qwen35_122b_a10b": SILICONFLOW_QWEN35_122B_A10B_CONFIG,
    "siliconflow_qwen3_235b_a22b_2507": SILICONFLOW_QWEN3_235B_A22B_2507_CONFIG,
}

DEFAULT_LLM_PRESET = os.getenv("DECEPTION_LLM_PRESET", "openai_gpt4o")


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