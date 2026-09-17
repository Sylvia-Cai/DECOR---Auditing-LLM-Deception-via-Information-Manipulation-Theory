from typing import Any, Dict

from .base_llm import BaseLLM


class AzureAnthropicLLM(BaseLLM):
    """
    LLM interface for Anthropic Claude models hosted on Azure AI Foundry.

    Uses the AnthropicFoundry client (anthropic>=0.40.0).
    Config keys:
        api_key        — Azure Anthropic API key (AZURE_ANTHROPIC_API_KEY)
        azure_endpoint — Full Foundry endpoint URL (including trailing /anthropic/)
        deployment_name — Model deployment name (e.g. "claude-sonnet-4-6")
        max_completion_tokens — Max output tokens (default 3000)
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        try:
            from anthropic import AnthropicBedrock as _  # noqa: F401 — verify package present
            from anthropic import AnthropicFoundry
            self._AnthropicFoundry = AnthropicFoundry
        except ImportError as exc:
            raise ImportError(
                "AzureAnthropicLLM requires the 'anthropic' package (>=0.40.0). "
                "Install it with: pip install 'anthropic>=0.40.0'"
            ) from exc

        required = ["api_key", "azure_endpoint", "deployment_name"]
        missing = [k for k in required if not self.config.get(k)]
        if missing:
            raise ValueError(f"AzureAnthropicLLM missing required config keys: {', '.join(missing)}")

        self.client = AnthropicFoundry(
            base_url=self.config["azure_endpoint"],
            api_key=self.config["api_key"],
        )
        self.deployment = self.config["deployment_name"]
        self.default_max_tokens = self.config.get("max_completion_tokens", 3000)

    def generate(self, prompt: str, **kwargs) -> str:
        max_tokens = kwargs.get("max_completion_tokens", self.default_max_tokens)

        system_prompt = kwargs.get("system_prompt")

        create_kwargs: Dict[str, Any] = {
            "model": self.deployment,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system_prompt:
            create_kwargs["system"] = system_prompt

        try:
            response = self.client.messages.create(**create_kwargs)
            return response.content[0].text.strip()

        except Exception as exc:
            print(f"[AzureAnthropicLLM] Error: {exc}")
            raise
