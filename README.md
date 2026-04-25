<p align="center">
  <img src="logo.png" alt="AllegroAgent logo" width="480">
</p>

# AllegroAgent

A lightweight, extensible Python framework for building stateful AI agents backed by local or remote LLMs. `Agent` is the single public entry point — configure it with a model string and a list of tools, and call `run()`. The framework handles conversation history, provider routing, and the tool-calling loop for you.

---

## Features

- **Stateful agents with tool use** — `Agent.run()` manages conversation history and automatically executes tools the LLM requests, looping until it produces a final answer.
- **Resumable conversations** — pass prior messages via `Agent(history=...)` to pick up where another session left off; export the current state any time with `agent.history()`.
- **Pluggable providers** — add new backends by implementing a single `generate()` method. Ollama ships in the box.
- **JSON Schema tools** — subclass `BaseTool`, declare parameters, and the agent exposes them to any function-calling-capable model.
- **Typed errors** — a small hierarchy of framework exceptions for clean error handling.
- **Zero heavy deps** — just `requests` at runtime.

---

## Architecture

```
┌─────────────────────────────────────────────┐
│  Agent            (public, stateful loop)   │
├─────────────────────────────────────────────┤
│  Providers        (Ollama, …)               │
└─────────────────────────────────────────────┘
```

`Agent` talks to providers through a small internal router (`_Client`) that parses the `provider:model` string and caches provider instances. You never need to touch it directly.

```
allegro_agent/
├── agent.py              # Agent — the public entry point
├── _client.py            # internal — provider router used by Agent
├── exceptions.py         # FrameworkError hierarchy
├── providers/
│   ├── base.py           # BaseProvider + ProviderResponse dataclass
│   ├── ollama.py         # OllamaProvider
│   └── __init__.py       # Provider registry (get/register_provider)
└── tools/
    ├── base.py           # BaseTool with JSON Schema + to_schema()
    └── file_write.py     # FileWriteTool (reference implementation)
```

---

## Installation

Requires Python **3.10+**.

Install from PyPI:

```bash
pip install allegro-agent
```

Or install from source (for development):

```bash
git clone https://github.com/ajithraghavan/AllegroAgent.git
cd AllegroAgent
pip install -e ".[dev]"
```

For the default Ollama provider, install and start Ollama locally:

```bash
ollama serve
ollama pull llama3
```

---

## Quick Start

```python
from allegro_agent import Agent, FileWriteTool

agent = Agent(
    name="Writer",
    model="ollama:llama3",
    temperature=0.1,
    system_prompt="You are a helpful writing assistant.",
    tools=[FileWriteTool()],
)

print(agent.run("Write a haiku about the ocean to ocean.txt"))
print(agent.run("Now write one about mountains to mountains.txt"))
agent.reset()  # clear history
```

What happens on each `run()`:

1. The agent sends the prompt + your tool schemas to the LLM.
2. If the LLM returns `tool_calls`, the agent executes each tool and feeds the results back.
3. Steps 1–2 repeat until the LLM returns a plain text answer (or `MAX_TOOL_ROUNDS = 10`).

---

## Resuming a conversation

`Agent` keeps conversation state in memory only. To carry a chat across processes, requests, or storage backends of your choice, capture it with `agent.history()` and seed a new `Agent` with `history=...`.

```python
import json
from allegro_agent import Agent

# Session 1 — have a conversation, then capture state.
agent = Agent(model="ollama:llama3", system_prompt="You are helpful.")
agent.run("My name is John.")
agent.run("Remember that.")

snapshot = agent.history()                   # list[dict], JSON-serializable
json.dump(snapshot, open("conv.json", "w"))  # caller picks the storage

# Session 2 — possibly in a fresh process — resume.
prior = json.load(open("conv.json"))
agent = Agent(model="ollama:llama3", system_prompt="You are helpful.", history=prior)
print(agent.run("What is my name?"))         # → references "John"
```

Notes:

- `history()` returns a deep copy, so mutating the returned list never affects the agent.
- `history` is validated on construction. Each item must be a dict with `role` (one of `"user"`, `"assistant"`, `"tool"`) and `content` (str). Malformed payloads raise `InvalidHistoryError`.
- The `"system"` role is **not** stored in history — system prompts continue to flow through the existing `system_prompt` kwarg, which the caller re-supplies (or omits) when constructing the resumed agent.
- The framework owns no I/O. Use JSON, a database, Redis, or anything else — it's just a list of plain dicts.

---

## Configuration

`Agent` accepts the following keyword arguments (all keyword-only):

| Argument        | Type            | Description                                        |
| --------------- | --------------- | -------------------------------------------------- |
| `model`         | `str`           | **Required.** Format: `provider:model`             |
| `name`          | `str`           | Display name (default `"Agent"`)                   |
| `temperature`   | `float`         | Sampling temperature                               |
| `max_tokens`    | `int`           | Max response tokens                                |
| `system_prompt` | `str`           | System instruction prepended to every call        |
| `tools`         | `list[BaseTool]`| Tools the agent may invoke                         |
| `history`       | `list[dict]`    | Prior conversation messages to resume from         |

---

## Extending the Framework

### Adding a new provider

Subclass `BaseProvider` and register it:

```python
from allegro_agent.providers.base import BaseProvider, ProviderResponse
from allegro_agent import register_provider

class OpenAIProvider(BaseProvider):
    def generate(self, messages, **kwargs) -> ProviderResponse:
        # call the API, translate the response
        return ProviderResponse(
            content="...",
            model=kwargs["model"],
            provider="openai",
            tool_calls=None,
        )

register_provider("openai", OpenAIProvider)
# now usable as model="openai:gpt-4"
```

### Adding a new tool

Subclass `BaseTool`, define JSON Schema parameters, implement `execute`:

```python
from allegro_agent import BaseTool

class AddTool(BaseTool):
    name = "add"
    description = "Add two integers."
    parameters = {
        "type": "object",
        "properties": {
            "a": {"type": "integer"},
            "b": {"type": "integer"},
        },
        "required": ["a", "b"],
    }

    def execute(self, **kwargs) -> str:
        return str(kwargs["a"] + kwargs["b"])
```

Pass an instance via the agent's `tools` list — the framework calls `to_schema()` to expose it to the LLM.

---

## Exceptions

All framework errors inherit from `FrameworkError`:

- `ProviderError` — provider-level failures (network, API errors)
- `ProviderNotFoundError` — unknown provider name in the registry
- `InvalidModelFormatError` — model string missing `provider:` prefix
- `ToolError` — raised by tools on execution failure
- `InvalidHistoryError` — `history` argument to `Agent` is malformed

---

## Testing

```bash
pytest
```

The reference end-to-end script in `tests/test_agent.py` exercises `Agent` against a live Ollama instance.

---

## Project Metadata

- **Package:** `allegro-agent` v0.3.0
- **Python:** ≥ 3.10
- **Runtime deps:** `requests`
- **License:** see `LICENSE`
