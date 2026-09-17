import openai
from typing import Dict, Any
from .base_llm import BaseLLM


class AzureDeepSeekLLM(BaseLLM):
    """
    LLM interface for DeepSeek-R1 hosted on Azure AI Foundry.

    Uses the OpenAI-compatible endpoint exposed by Azure
    (``https://<resource>.services.ai.azure.com/openai/v1/``).
    Config keys:
        api_key         — Azure API key
        azure_endpoint  — Full base URL including trailing slash
        deployment_name — Model deployment name (e.g. "DeepSeek-R1")
        temperature     — Sampling temperature (default 0.0)
        max_completion_tokens — Output token budget (default 2000)
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        required_keys = ["api_key", "azure_endpoint", "deployment_name"]
        missing = [k for k in required_keys if not self.config.get(k)]
        if missing:
            raise ValueError(f"AzureDeepSeekLLM missing required config keys: {', '.join(missing)}")

        # Azure AI Foundry exposes an OpenAI-compatible endpoint; use openai.OpenAI
        # with the custom base_url rather than openai.AzureOpenAI.
        self.client = openai.OpenAI(
            api_key=self.config["api_key"],
            base_url=self.config["azure_endpoint"],
        )
        self.deployment = self.config["deployment_name"]
        self.default_temperature = self.config.get("temperature", 0.0)
        self.default_max_output_tokens = self.config.get("max_completion_tokens", 2000)

    def generate(self, prompt: str, **kwargs) -> str:
        """
        Call the DeepSeek-R1 chat-completions endpoint and return the response text.
        """
        temperature = kwargs.get("temperature", self.default_temperature)
        max_tokens = kwargs.get("max_completion_tokens", self.default_max_output_tokens)

        system_prompt = kwargs.get("system_prompt")
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            completion = self.client.chat.completions.create(
                model=self.deployment,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return completion.choices[0].message.content.strip()

        except openai.APIConnectionError as exc:
            print(
                f"[AzureDeepSeekLLM] Connection error — endpoint: {self.config['azure_endpoint']}\n"
                f"Details: {exc}"
            )
            raise
        except Exception as exc:
            print(f"[AzureDeepSeekLLM] Unexpected error: {exc}")
            raise