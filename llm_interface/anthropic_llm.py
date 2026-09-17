from typing import Any, Dict

from .base_llm import BaseLLM


class AnthropicLLM(BaseLLM):
    """
    LLM interface for Anthropic Claude models: the official (direct) Anthropic
    API, or Claude hosted on Azure AI Foundry.

    Reference implementation for paper reproducibility — works with a plain
    console.anthropic.com API key by default (no Azure account needed).

    Dispatch:
        If ``azure_endpoint`` is set, uses the ``AnthropicFoundry`` client
        (Azure AI Foundry deployments — requires ``deployment_name`` too).
        Otherwise uses the direct ``Anthropic`` client with ``model_name``.

    Config keys:
        api_key                — Anthropic (or Azure Foundry) API key
        model_name              — Model identifier for direct API calls
                                  (e.g. "claude-sonnet-4-6")
        azure_endpoint          — Full Foundry endpoint URL (Azure only)
        deployment_name         — Model deployment name (Azure only)
        max_completion_tokens   — Output token budget (default 3000)
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        try:
            import anthropic
        except ImportError as exc:
            raise ImportError(
                "AnthropicLLM requires the 'anthropic' package. "
                "Install it with: pip install 'anthropic>=0.40.0'"
            ) from exc

        if not self.config.get("api_key"):
            raise ValueError("AnthropicLLM requires 'api_key' in config.")

        if self.config.get("azure_endpoint"):
            if not self.config.get("deployment_name"):
                raise ValueError("AnthropicLLM (Azure Foundry) requires 'deployment_name' in config.")
            self.client = anthropic.AnthropicFoundry(
                base_url=self.config["azure_endpoint"],
                api_key=self.config["api_key"],
            )
            self.model_name = self.config["deployment_name"]
        else:
            if not self.config.get("model_name"):
                raise ValueError("AnthropicLLM requires 'model_name' in config.")
            self.client = anthropic.Anthropic(api_key=self.config["api_key"])
            self.model_name = self.config["model_name"]

        self.default_max_tokens = self.config.get("max_completion_tokens", 3000)

    def generate(self, prompt: str, **kwargs) -> str:
        max_tokens = kwargs.get("max_completion_tokens", self.default_max_tokens)
        system_prompt = kwargs.get("system_prompt")

        create_kwargs: Dict[str, Any] = {
            "model": self.model_name,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system_prompt:
            create_kwargs["system"] = system_prompt

        try:
            response = self.client.messages.create(**create_kwargs)
            return response.content[0].text.strip()
        except Exception as exc:
            print(f"[AnthropicLLM] API call failed: {exc}")
            raise
