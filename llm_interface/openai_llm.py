import openai
from typing import Dict, Any
from .base_llm import BaseLLM


class OpenAILLM(BaseLLM):
    """
    LLM interface for any OpenAI-protocol-compatible endpoint: the official
    OpenAI API, OpenRouter, SiliconFlow, and any other provider that speaks
    the OpenAI chat-completions format.

    Reference implementation for paper reproducibility — to use a different
    vendor, just point ``base_url``/``api_key`` at it (see README).

    Config keys:
        api_key                — API key for the target provider
        model_name              — Model identifier (e.g. "gpt-4o")
        base_url                — API base URL (default: official OpenAI)
        max_completion_tokens  — Output token budget (default 3000)
        temperature             — Sampling temperature (default 0.0)
        top_p                   — Nucleus sampling parameter (default 1.0)
        skip_temperature        — If True, omit temperature/top_p from the
                                  API call (required for reasoning models
                                  such as o1/o3/gpt-5)
        legacy_max_tokens       — If True, send the older ``max_tokens`` param
                                  instead of ``max_completion_tokens`` and omit
                                  top_p/extra passthrough params. Needed for
                                  older OpenAI-compatible deployments that
                                  don't understand the newer param names.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        if not self.config.get("api_key"):
            raise ValueError("OpenAILLM requires 'api_key' in config.")

        self.client = openai.OpenAI(
            api_key=self.config["api_key"],
            base_url=self.config.get("base_url", "https://api.openai.com/v1"),
        )

        self.model_name = self.config.get("model_name", "gpt-4o")
        self.default_max_tokens = self.config.get(
            "max_completion_tokens",
            self.config.get("default_max_tokens", 3000),
        )
        self.default_temperature = self.config.get(
            "temperature",
            self.config.get("default_temperature", 0.0),
        )
        self.default_top_p = self.config.get("top_p", 1.0)
        # Reasoning models (o1, o3, gpt-5) do not accept temperature/top_p.
        self.skip_temperature = self.config.get("skip_temperature", False)
        self.legacy_max_tokens = self.config.get("legacy_max_tokens", False)

    def generate(self, prompt: str, **kwargs) -> str:
        """
        Call the chat-completions endpoint and return the response text.

        kwargs override instance defaults for max_completion_tokens, temperature,
        top_p, and any additional supported parameters (response_format, seed, etc.).
        Reasoning models may return None content when the reply is purely in
        reasoning tokens; this is handled gracefully by returning an empty string.
        """
        max_tokens = kwargs.get(
            "max_completion_tokens",
            kwargs.get("max_tokens", self.default_max_tokens),
        )
        temperature = kwargs.get("temperature", self.default_temperature)
        top_p = kwargs.get("top_p", self.default_top_p)

        system_prompt = kwargs.get("system_prompt")
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        request_params: Dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
        }
        if self.legacy_max_tokens:
            request_params["max_tokens"] = max_tokens
            if not self.skip_temperature:
                request_params["temperature"] = temperature
        else:
            request_params["max_completion_tokens"] = max_tokens
            if not self.skip_temperature:
                request_params["temperature"] = temperature
                request_params["top_p"] = top_p

            # Forward any additional supported parameters passed by the caller.
            _reserved = {"max_completion_tokens", "max_tokens", "temperature", "top_p", "system_prompt"}
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
