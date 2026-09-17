import openai
from typing import Dict, Any
from .base_llm import BaseLLM


class OpenAILLM(BaseLLM):
    """
    LLM interface for the official OpenAI API and OpenAI-compatible endpoints.

    Designed as the reference implementation for paper reproducibility.
    Supports all chat-completion models including reasoning models (o1, o3)
    that require ``skip_temperature=True``.

    Also compatible with OpenRouter and other OpenAI-compatible providers by
    setting ``base_url`` in the config.

    Config keys:
        api_key               — OpenAI (or provider) API key
        model_name            — Model identifier (e.g. "gpt-4o")
        base_url              — API base URL (default: "https://api.openai.com/v1")
        max_completion_tokens — Output token budget (default 3000)
        temperature           — Sampling temperature (default 0.0)
        skip_temperature      — If True, omit temperature from the API call
                                (required for o1/o3 reasoning models)
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        if not self.config.get("api_key"):
            raise ValueError("OpenAILLM requires 'api_key' in config.")

        self.client = openai.OpenAI(
            api_key=self.config["api_key"],
            base_url=self.config.get("base_url", "https://api.openai.com/v1"),
        )
        self.model_name = self.config.get(
            "deployment_name",
            self.config.get("model_name", "gpt-4o"),
        )
        self.default_max_tokens = self.config.get(
            "max_completion_tokens",
            self.config.get("default_max_tokens", 3000),
        )
        self.default_temperature = self.config.get(
            "temperature",
            self.config.get("default_temperature", 0.0),
        )
        # Reasoning models (o1, o3) do not accept temperature in the API call.
        self.skip_temperature = self.config.get("skip_temperature", False)

    def generate(self, prompt: str, **kwargs) -> str:
        """
        Call the chat-completions endpoint and return the response text.

        kwargs override instance defaults for max_completion_tokens and temperature.
        Reasoning models may return None content when the reply is purely in
        reasoning tokens; this is handled gracefully by returning an empty string.
        """
        max_tokens = kwargs.get(
            "max_completion_tokens",
            kwargs.get("max_tokens", self.default_max_tokens),
        )
        temperature = kwargs.get("temperature", self.default_temperature)

        system_prompt = kwargs.get("system_prompt")
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        request_params: Dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "max_completion_tokens": max_tokens,
        }
        if not self.skip_temperature:
            request_params["temperature"] = temperature

        # Forward any additional supported parameters passed by the caller.
        _reserved = {"max_completion_tokens", "max_tokens", "temperature", "system_prompt"}
        for key, value in kwargs.items():
            if key not in _reserved:
                request_params[key] = value

        try:
            response = self.client.chat.completions.create(**request_params)
            content = response.choices[0].message.content
            # Reasoning models may return None content; fall back to empty string.
            return (content or "").strip()
        except openai.BadRequestError:
            raise
        except Exception as exc:
            print(f"[OpenAILLM] API call failed: {exc}")
            raise
