import random
import time
from abc import ABC, abstractmethod
from typing import Any, Dict

class BaseLLM(ABC):
    """
    Abstract base class for Large Language Model interfaces.
    """
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the LLM interface.

        Args:
            config: Configuration dictionary for the LLM (e.g., API keys, model names).
        """
        self.config = config

    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate text based on the given prompt.

        Args:
            prompt: The input prompt string.
            **kwargs: Additional keyword arguments for the LLM API (e.g., temperature, max_tokens).

        Returns:
            The generated text string.
        """
        pass

    def generate_with_retry(
        self,
        prompt: str,
        max_retries: int = 3,
        base_delay: float = 2.0,
        **kwargs,
    ) -> str:
        """Call generate() with exponential-backoff retry on transient failures."""
        last_exc: BaseException = RuntimeError("No attempts made")
        for attempt in range(max_retries):
            try:
                return self.generate(prompt, **kwargs)
            except Exception as exc:
                last_exc = exc
                if attempt < max_retries - 1:
                    msg = str(exc).lower()
                    is_rate_limited = (
                        "rate limit" in msg
                        or "rate_limit" in msg
                        or "429" in msg
                        or "tpm limit" in msg
                    )
                    effective_base_delay = 10.0 if is_rate_limited else base_delay
                    delay = effective_base_delay * (2 ** attempt) + random.uniform(0, 1)
                    print(
                        f"[LLM Retry {attempt + 1}/{max_retries - 1}] "
                        f"{type(exc).__name__}: {exc}. Retrying in {delay:.1f}s ..."
                    )
                    time.sleep(delay)
        raise last_exc
