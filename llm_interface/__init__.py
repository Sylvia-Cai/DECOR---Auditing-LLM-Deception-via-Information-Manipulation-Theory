from typing import Any, Dict

from .base_llm import BaseLLM
from .openai_llm import OpenAILLM
from .anthropic_llm import AnthropicLLM
from .google_genai_llm import GoogleGenAILLM


def create_llm(config: Dict[str, Any]):
    provider = config.get("provider", "openai")

    if provider in {"openai", "openai_compatible", "openrouter"}:
        return OpenAILLM(config)
    if provider == "anthropic":
        return AnthropicLLM(config)
    if provider == "google_genai":
        return GoogleGenAILLM(config)

    raise ValueError(
        "Unsupported provider "
        f"'{provider}'. Expected one of: openai, openai_compatible, openrouter, "
        "anthropic, google_genai"
    )


__all__ = [
    "create_llm",
    "BaseLLM",
    "OpenAILLM",
    "AnthropicLLM",
    "GoogleGenAILLM",
]
