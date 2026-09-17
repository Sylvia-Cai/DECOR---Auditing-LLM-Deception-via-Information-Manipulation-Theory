import openai
from typing import Dict, Any
from .base_llm import BaseLLM


class AzureOpenAILLM(BaseLLM):
    """
    LLM interface for Azure OpenAI deployments (GPT-4o, GPT-5, o3, etc.).

    Supports all Azure OpenAI chat-completion parameters.
    Reasoning models (e.g. o3) that reject temperature/top_p should set
    ``skip_temperature: true`` in their config entry.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: Configuration dictionary with keys:
                - api_key            : Azure OpenAI API key.
                - azure_endpoint     : Azure OpenAI endpoint URL.
                - api_version        : Azure OpenAI API version string.
                - deployment_name    : Azure deployment name.
                - max_completion_tokens : Default token budget (default 1000).
                - temperature        : Sampling temperature (default from config).
                - top_p              : Nucleus sampling parameter (default 1.0).
                - skip_temperature   : If True, omit temperature/top_p from the
                                       API call (required for o3 and gpt-5).
        """
        super().__init__(config)

        required_keys = ["api_key", "azure_endpoint", "api_version", "deployment_name"]
        missing_keys = [k for k in required_keys if not self.config.get(k)]
        if missing_keys:
            raise ValueError(f"AzureOpenAILLM missing required config keys: {', '.join(missing_keys)}")

        self.client = openai.AzureOpenAI(
            api_version=self.config["api_version"],
            azure_endpoint=self.config["azure_endpoint"],
            api_key=self.config["api_key"],
        )
        self.deployment = self.config["deployment_name"]

        self.default_max_completion_tokens = self.config.get("max_completion_tokens", 1000)
        self.default_temperature = self.config.get("temperature")
        self.default_top_p = self.config.get("top_p", 1.0)
        # Reasoning models (e.g. o3, gpt-5) do not accept temperature/top_p.
        self.skip_temperature = self.config.get("skip_temperature", False)

    def generate(self, prompt: str, **kwargs) -> str:
        """
        Call the Azure OpenAI chat-completions endpoint and return the response text.

        kwargs override instance defaults for: max_completion_tokens, temperature, top_p,
        and any additional supported parameters (n, stop, seed, response_format, etc.).
        """
        max_completion_tokens = kwargs.get("max_completion_tokens", self.default_max_completion_tokens)
        temperature = kwargs.get("temperature", self.default_temperature)
        top_p = kwargs.get("top_p", self.default_top_p)

        system_prompt = kwargs.get("system_prompt")
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        request_params: Dict[str, Any] = {
            "model": self.deployment,
            "messages": messages,
            "max_completion_tokens": max_completion_tokens,
        }
        if not self.skip_temperature:
            request_params["temperature"] = temperature
            request_params["top_p"] = top_p

        # Forward any additional supported parameters passed by the caller.
        passthrough = [
            "n", "stream", "stop", "frequency_penalty",
            "presence_penalty", "response_format", "seed",
            "logit_bias", "user",
        ]
        kwargs.pop("system_prompt", None)  # already consumed above
        for key in passthrough:
            if key in kwargs:
                request_params[key] = kwargs[key]

        try:
            response = self.client.chat.completions.create(**request_params)
            return response.choices[0].message.content.strip()
        except openai.BadRequestError:
            raise
        except Exception as exc:
            print(f"[AzureOpenAILLM] API call failed: {exc}")
            raise