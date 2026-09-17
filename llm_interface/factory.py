from typing import Any, Dict

from .azure_openai_llm import AzureOpenAILLM
from .azure_deepseek_llm import AzureDeepSeekLLM
from .openai_llm import OpenAILLM


def create_llm(config: Dict[str, Any]):
    provider = config.get("provider", "azure_openai")

    if provider == "azure_openai":
        return AzureOpenAILLM(config)
    if provider == "azure_deepseek":
        return AzureDeepSeekLLM(config)
    if provider in {"openai", "openai_compatible", "openrouter"}:
        return OpenAILLM(config)
    if provider == "google_genai":
        from .google_genai_llm import GoogleGenAILLM

        return GoogleGenAILLM(config)
    if provider == "azure_anthropic":
        from .azure_anthropic_llm import AzureAnthropicLLM

        return AzureAnthropicLLM(config)
    if provider == "anthropic":
        from .anthropic_llm import AnthropicLLM

        return AnthropicLLM(config)

    raise ValueError(
        "Unsupported provider "
        f"'{provider}'. Expected one of: azure_openai, azure_deepseek, azure_anthropic, "
        "anthropic, openai_compatible, openrouter, google_genai"
    )