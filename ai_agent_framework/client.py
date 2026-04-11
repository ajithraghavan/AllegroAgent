from .providers import get_provider, ProviderResponse
from .exceptions import InvalidModelFormatError


class Client:
    """Simple unified LLM client (Layer 2).

    Parses 'provider:model' syntax and routes to the correct provider.

    Usage:
        client = Client()
        response = client.chat(
            model="ollama:llama3",
            messages=[{"role": "user", "content": "Hello!"}]
        )
        print(response.content)
    """

    def __init__(self, **provider_kwargs):
        """Initialize client.

        Args:
            **provider_kwargs: Passed to provider constructors
                               (e.g. base_url for Ollama).
        """
        self._provider_kwargs = provider_kwargs
        self._providers = {}

    def _parse_model(self, model: str) -> tuple[str, str]:
        """Parse 'provider:model' into (provider_name, model_name)."""
        if ":" not in model:
            raise InvalidModelFormatError(
                f"Model '{model}' must follow 'provider:model' format. "
                f"Example: 'ollama:llama3'"
            )
        provider_name, model_name = model.split(":", 1)
        return provider_name.strip(), model_name.strip()

    def _get_provider(self, provider_name: str):
        """Get or create a cached provider instance."""
        if provider_name not in self._providers:
            self._providers[provider_name] = get_provider(
                provider_name, **self._provider_kwargs
            )
        return self._providers[provider_name]

    def chat(self, model: str, messages: list[dict], **kwargs) -> ProviderResponse:
        """Send a chat request to an LLM.

        Args:
            model: Model string in 'provider:model' format (e.g. "ollama:llama3")
            messages: List of message dicts with 'role' and 'content'.
            **kwargs: Additional params like temperature, max_tokens.

        Returns:
            ProviderResponse with the LLM's reply.
        """
        provider_name, model_name = self._parse_model(model)
        provider = self._get_provider(provider_name)
        return provider.generate(messages, model=model_name, **kwargs)
