from typing import Dict, Any, List


class BaseAgent:
    """Abstract base class for all agents in the IMT audit pipeline."""

    def __init__(self, config: Dict[str, Any], llm_interface: Any):
        self.config = config   # agent-specific configuration dict
        self.llm = llm_interface  # shared LLM interface (model-agnostic)

    def _format_prompt(self, template: str, **kwargs) -> str:
        """Fill a prompt template with keyword arguments."""
        return template.format(**kwargs)

    def _parse_llm_response(self, response: str) -> Dict[str, Any]:
        """Default response parser; subclasses may override for structured output."""
        return {"text": response}

    def _format_conversation_context(self, conversation_context: List[Dict[str, str]]) -> str:
        """Serialize a conversation history list into a readable string for prompts."""
        if not conversation_context:
            return "No previous conversation history."
        formatted = [f"{turn['speaker']}: {turn['utterance']}" for turn in conversation_context]
        return "\n".join(formatted)

    def process(self, *args, **kwargs) -> Any:
        """Main entry point; must be implemented by every concrete agent subclass."""
        raise NotImplementedError("Each agent must implement its own process() method.")