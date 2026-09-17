from typing import Any, Dict

from .base_llm import BaseLLM
from .openai_llm import OpenAILLM
from .anthropic_llm import AnthropicLLM
from .google_genai_llm import GoogleGenAILLM


def create_llm(config: Dict[str, Any]):
    provider = config.get("provider", "azure_openai")

    if provider in {"azure_openai", "azure_deepseek", "openai", "openai_compatible", "openrouter"}:
        return OpenAILLM(config)
    if provider in {"anthropic", "azure_anthropic"}:
        return AnthropicLLM(config)
    if provider == "google_genai":
        return GoogleGenAILLM(config)

    raise ValueError(
        "Unsupported provider "
        f"'{provider}'. Expected one of: azure_openai, azure_deepseek, openai, "
        "openai_compatible, openrouter, anthropic, azure_anthropic, google_genai"
    )


__all__ = [
    "create_llm",
    "BaseLLM",
    "OpenAILLM",
    "AnthropicLLM",
    "GoogleGenAILLM",
]
