from typing import Any, Dict

from .base_llm import BaseLLM


class GoogleGenAILLM(BaseLLM):
    """
    LLM interface for Google Gemini models via the google-genai SDK.

    Safety filters are set to BLOCK_NONE for all harm categories so that
    adversarial prompts used in deception-detection research are not blocked.
    Config keys:
        api_key               — Google AI API key
        model_name            — Gemini model identifier (e.g. "gemini-2.5-pro")
        temperature           — Sampling temperature (default 0.0)
        max_completion_tokens — Output token budget (default 3000)
    """
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        try:
            from google import genai
            from google.genai import types as genai_types
        except ImportError as exc:
            raise ImportError(
                "GoogleGenAILLM requires the 'google-genai' package. "
                "Install it with: pip install google-genai"
            ) from exc

        api_key = self.config.get("api_key")
        model_name = self.config.get("model_name")
        if not api_key:
            raise ValueError("GoogleGenAILLM requires api_key in config")
        if not model_name:
            raise ValueError("GoogleGenAILLM requires model_name in config")

        self._client = genai.Client(api_key=api_key)
        self._types = genai_types
        self.model_name = model_name
        self.default_temperature = self.config.get("temperature", 0.0)
        self.default_max_output_tokens = self.config.get("max_completion_tokens", 3000)

    def generate(self, prompt: str, **kwargs) -> str:
        temperature = kwargs.get("temperature", self.default_temperature)
        max_output_tokens = kwargs.get("max_completion_tokens", self.default_max_output_tokens)
        system_prompt = kwargs.get("system_prompt")

        config_kwargs: Dict[str, Any] = {
            "temperature": temperature,
            "max_output_tokens": max_output_tokens,
        }
        if system_prompt:
            config_kwargs["system_instruction"] = system_prompt

        response = self._client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=self._types.GenerateContentConfig(
                **config_kwargs,
                safety_settings=[
                    self._types.SafetySetting(
                        category="HARM_CATEGORY_HARASSMENT",
                        threshold="BLOCK_NONE",
                    ),
                    self._types.SafetySetting(
                        category="HARM_CATEGORY_HATE_SPEECH",
                        threshold="BLOCK_NONE",
                    ),
                    self._types.SafetySetting(
                        category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
                        threshold="BLOCK_NONE",
                    ),
                    self._types.SafetySetting(
                        category="HARM_CATEGORY_DANGEROUS_CONTENT",
                        threshold="BLOCK_NONE",
                    ),
                ],
            ),
        )

        # finish_reason 2 = SAFETY block: no content parts are returned.
        # Attempt to access candidates[0].text safely; fall back to "" so
        # callers (which already handle empty/error strings) can continue.
        try:
            text = response.text
        except (ValueError, AttributeError, IndexError):
            # Log the finish_reason if available so it's visible in run logs.
            try:
                reason = response.candidates[0].finish_reason
            except Exception:
                reason = "unknown"
            raise ValueError(
                f"Gemini returned no text content (finish_reason={reason}). "
                "The prompt may have been blocked by safety filters."
            )

        return (text or "").strip()
