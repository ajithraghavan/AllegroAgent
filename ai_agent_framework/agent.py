from .client import Client
from .providers import ProviderResponse


class Agent:
    """Stateful AI Agent (Layer 3).

    Uses Client internally for LLM calls. Maintains conversation
    history and will support tools in the future.

    Usage:
        agent = Agent({
            "name": "Research Agent",
            "model": "ollama:llama3",
            "temperature": 0.1
        })
        response = agent.run("What is AI?")
        print(response)
    """

    def __init__(self, config: dict):
        """Initialize agent from config dict.

        Args:
            config: Dict with keys:
                - name (str): Agent name
                - model (str): Model in 'provider:model' format
                - temperature (float, optional): Sampling temperature
                - system_prompt (str, optional): System instruction
                - max_tokens (int, optional): Max response tokens
        """
        self.name = config.get("name", "Agent")
        self.model = config["model"]
        self.temperature = config.get("temperature")
        self.system_prompt = config.get("system_prompt")
        self.max_tokens = config.get("max_tokens")

        self._client = Client()
        self._history: list[dict] = []

    def run(self, prompt: str) -> str:
        """Send a prompt to the LLM and return the response text.

        Args:
            prompt: The user's message.

        Returns:
            The LLM's response as a string.
        """
        self._history.append({"role": "user", "content": prompt})

        messages = self._build_messages()

        kwargs = {}
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        if self.max_tokens is not None:
            kwargs["max_tokens"] = self.max_tokens

        response: ProviderResponse = self._client.chat(
            model=self.model,
            messages=messages,
            **kwargs,
        )

        self._history.append({"role": "assistant", "content": response.content})

        return response.content

    def _build_messages(self) -> list[dict]:
        """Build the full message list including system prompt."""
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.extend(self._history)
        return messages

    def reset(self):
        """Clear conversation history."""
        self._history.clear()

    def __repr__(self):
        return f"Agent(name='{self.name}', model='{self.model}')"
