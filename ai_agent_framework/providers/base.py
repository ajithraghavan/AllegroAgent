from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class ProviderResponse:
    """Standardized response from any provider."""
    content: str
    model: str
    provider: str
    usage: dict = None

    def __str__(self):
        return self.content


class BaseProvider(ABC):
    """Base interface that all providers must implement.

    Every provider translates between the framework's standard
    message format and its own API format. Client and Agent never
    see provider-specific details.
    """

    @abstractmethod
    def generate(self, messages: list[dict], **kwargs) -> ProviderResponse:
        """Send messages to the LLM and return a standardized response.

        Args:
            messages: List of dicts with 'role' and 'content' keys.
                      e.g. [{"role": "user", "content": "Hello"}]
            **kwargs: Provider-agnostic params like temperature, max_tokens.

        Returns:
            ProviderResponse with the LLM's reply.
        """
        pass
