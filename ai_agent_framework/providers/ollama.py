import requests

from .base import BaseProvider, ProviderResponse
from ..exceptions import ProviderError


class OllamaProvider(BaseProvider):
    """Provider for Ollama (local LLM server)."""

    DEFAULT_BASE_URL = "http://localhost:11434"

    def __init__(self, base_url: str = None):
        self.base_url = (base_url or self.DEFAULT_BASE_URL).rstrip("/")

    def generate(self, messages: list[dict], **kwargs) -> ProviderResponse:
        model = kwargs.pop("model", None)
        if not model:
            raise ProviderError("Model name is required for Ollama provider.")

        tools = kwargs.pop("tools", None)

        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
        }

        if tools:
            payload["tools"] = tools

        if "temperature" in kwargs:
            payload.setdefault("options", {})["temperature"] = kwargs.pop("temperature")

        if "max_tokens" in kwargs:
            payload.setdefault("options", {})["num_predict"] = kwargs.pop("max_tokens")

        try:
            resp = requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=120,
            )
            resp.raise_for_status()
        except requests.ConnectionError:
            raise ProviderError(
                f"Cannot connect to Ollama at {self.base_url}. "
                "Is Ollama running? Start it with: ollama serve"
            )
        except requests.HTTPError:
            error_detail = resp.text
            try:
                error_detail = resp.json().get("error", resp.text)
            except Exception:
                pass
            raise ProviderError(f"Ollama API error: {error_detail}")

        data = resp.json()
        message = data.get("message", {})

        tool_calls = message.get("tool_calls")

        return ProviderResponse(
            content=message.get("content", ""),
            model=model,
            provider="ollama",
            usage={
                "prompt_tokens": data.get("prompt_eval_count"),
                "completion_tokens": data.get("eval_count"),
            },
            tool_calls=tool_calls,
        )
