import copy
import json

from ._client import _Client
from .exceptions import InvalidHistoryError
from .providers import ProviderResponse
from .tools.base import BaseTool


_ALLOWED_HISTORY_ROLES = ("user", "assistant", "tool")


class Agent:
    """Stateful AI Agent with tool support.

    The single public entry point of the framework. Maintains conversation
    history and executes tools when the LLM requests them.

    Usage:

        from allegro_agent.tools import FileWriteTool

        agent = Agent(
            name="Research Agent",
            model="ollama:llama3",
            temperature=0.1,
            tools=[FileWriteTool()],
        )
        response = agent.run("Write a haiku to poem.txt")
        print(response)
    """

    MAX_TOOL_ROUNDS = 10

    def __init__(
        self,
        *,
        model: str,
        name: str = "Agent",
        temperature: float | None = None,
        max_tokens: int | None = None,
        system_prompt: str | None = None,
        tools: list[BaseTool] | None = None,
        history: list[dict] | None = None,
    ):
        """Initialize an Agent.

        Args:
            model: Model in 'provider:model' format (e.g. "ollama:llama3"). Required.
            name: Display name for the agent.
            temperature: Sampling temperature.
            max_tokens: Max response tokens.
            system_prompt: System instruction prepended to every call.
            tools: Tools the agent is allowed to invoke.
            history: Prior conversation messages to resume from. Each item must
                be a dict with 'role' (one of 'user', 'assistant', 'tool') and
                'content' (str). Assistant messages may include 'tool_calls'.
                The list is deep-copied so callers can mutate their original
                freely.
        """
        self.name = name
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.system_prompt = system_prompt

        self._client = _Client()
        if history is None:
            self._history: list[dict] = []
        else:
            self._validate_history(history)
            self._history = copy.deepcopy(history)

        self._tools: dict[str, BaseTool] = {}
        for tool in tools or []:
            self._tools[tool.name] = tool

    def run(self, prompt: str) -> str:
        """Send a prompt to the LLM and return the response text.

        If the LLM requests tool calls, executes them and continues
        the conversation until the LLM produces a final text response.

        Args:
            prompt: The user's message.

        Returns:
            The LLM's final response as a string.
        """
        self._history.append({"role": "user", "content": prompt})

        for _ in range(self.MAX_TOOL_ROUNDS):
            response = self._call_llm()

            if not response.tool_calls:
                # No tool calls — we have the final answer
                self._history.append({"role": "assistant", "content": response.content})
                return response.content

            # LLM wants to call tools — record its message and execute
            self._history.append({
                "role": "assistant",
                "content": response.content or "",
                "tool_calls": response.tool_calls,
            })

            for tool_call in response.tool_calls:
                result = self._execute_tool(tool_call)
                self._history.append({
                    "role": "tool",
                    "content": result,
                })

        # Safety: if we exhaust tool rounds, return whatever we have
        return response.content or "Max tool rounds reached."

    def _call_llm(self) -> ProviderResponse:
        """Make a single LLM call with current history and tools."""
        messages = self._build_messages()

        kwargs = {}
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        if self.max_tokens is not None:
            kwargs["max_tokens"] = self.max_tokens
        if self._tools:
            kwargs["tools"] = [tool.to_schema() for tool in self._tools.values()]

        return self._client.chat(
            model=self.model,
            messages=messages,
            **kwargs,
        )

    def _execute_tool(self, tool_call: dict) -> str:
        """Execute a single tool call and return the result string."""
        func = tool_call.get("function", {})
        tool_name = func.get("name", "")
        arguments = func.get("arguments", {})

        # Arguments may come as a JSON string from some providers
        if isinstance(arguments, str):
            arguments = json.loads(arguments)

        tool = self._tools.get(tool_name)
        if not tool:
            return f"Error: unknown tool '{tool_name}'"

        try:
            return tool.execute(**arguments)
        except Exception as e:
            return f"Error executing {tool_name}: {e}"

    def _build_messages(self) -> list[dict]:
        """Build the full message list including system prompt."""
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.extend(self._history)
        return messages

    def history(self) -> list[dict]:
        """Return a deep copy of the conversation messages so far.

        The returned list is a JSON-serializable snapshot the caller can store
        anywhere (file, database, network) and later pass back via the
        ``history`` constructor argument to resume the conversation.
        """
        return copy.deepcopy(self._history)

    def reset(self):
        """Clear conversation history."""
        self._history.clear()

    @staticmethod
    def _validate_history(history) -> None:
        """Validate the shape of a history payload passed to the constructor.

        Raises InvalidHistoryError with a message naming the offending index
        and problem so callers can fix bad payloads quickly.
        """
        if not isinstance(history, list):
            raise InvalidHistoryError(
                f"history must be a list, got {type(history).__name__}"
            )

        for i, msg in enumerate(history):
            if not isinstance(msg, dict):
                raise InvalidHistoryError(
                    f"history[{i}] must be a dict, got {type(msg).__name__}"
                )
            if "role" not in msg:
                raise InvalidHistoryError(f"history[{i}] is missing 'role'")
            if "content" not in msg:
                raise InvalidHistoryError(f"history[{i}] is missing 'content'")

            role = msg["role"]
            if role not in _ALLOWED_HISTORY_ROLES:
                raise InvalidHistoryError(
                    f"history[{i}] has invalid role {role!r}; "
                    f"must be one of {_ALLOWED_HISTORY_ROLES} "
                    f"(system prompts go through the 'system_prompt' kwarg, not history)"
                )

            if not isinstance(msg["content"], str):
                raise InvalidHistoryError(
                    f"history[{i}] 'content' must be a str, "
                    f"got {type(msg['content']).__name__}"
                )

            if role == "assistant" and "tool_calls" in msg:
                tool_calls = msg["tool_calls"]
                if not isinstance(tool_calls, list):
                    raise InvalidHistoryError(
                        f"history[{i}] 'tool_calls' must be a list, "
                        f"got {type(tool_calls).__name__}"
                    )
                for j, tc in enumerate(tool_calls):
                    if not isinstance(tc, dict):
                        raise InvalidHistoryError(
                            f"history[{i}].tool_calls[{j}] must be a dict, "
                            f"got {type(tc).__name__}"
                        )

    def __repr__(self):
        tool_names = list(self._tools.keys())
        return f"Agent(name='{self.name}', model='{self.model}', tools={tool_names})"
