from .base import BaseProvider, ProviderResponse
from .ollama import OllamaProvider
from ..exceptions import ProviderNotFoundError

# Provider registry — add new providers here
_PROVIDERS: dict[str, type[BaseProvider]] = {
    "ollama": OllamaProvider,
}


def get_provider(name: str, **kwargs) -> BaseProvider:
    """Get a provider instance by name.

    Args:
        name: Provider name (e.g. "ollama")
        **kwargs: Provider-specific init args (e.g. base_url)

    Returns:
        An instance of the requested provider.

    Raises:
        ProviderNotFoundError: If the provider is not registered.
    """
    provider_cls = _PROVIDERS.get(name)
    if not provider_cls:
        available = ", ".join(_PROVIDERS.keys())
        raise ProviderNotFoundError(
            f"Provider '{name}' not found. Available: {available}"
        )
    return provider_cls(**kwargs)


def register_provider(name: str, provider_cls: type[BaseProvider]):
    """Register a custom provider."""
    _PROVIDERS[name] = provider_cls
