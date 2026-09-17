from typing import Any, Dict

from .openai_llm import OpenAILLM


def create_llm(config: Dict[str, Any]):
    provider = config.get("provider", "azure_openai")

    if provider in {"azure_openai", "azure_deepseek", "openai", "openai_compatible", "openrouter"}:
        return OpenAILLM(config)
    if provider in {"anthropic", "azure_anthropic"}:
        from .anthropic_llm import AnthropicLLM

        return AnthropicLLM(config)
    if provider == "google_genai":
        from .google_genai_llm import GoogleGenAILLM

        return GoogleGenAILLM(config)

    raise ValueError(
        "Unsupported provider "
        f"'{provider}'. Expected one of: azure_openai, azure_deepseek, openai, "
        "openai_compatible, openrouter, anthropic, azure_anthropic, google_genai"
    )
