from abc import ABC, abstractmethod


class BaseTool(ABC):
    """Base interface for all agent tools.

    Every tool must define its name, description, parameters schema,
    and an execute method. The schema is used to tell the LLM what
    arguments the tool accepts (JSON Schema format).

    Example:
        class MyTool(BaseTool):
            name = "my_tool"
            description = "Does something useful."
            parameters = {
                "type": "object",
                "properties": {
                    "input": {"type": "string", "description": "The input value"}
                },
                "required": ["input"]
            }

            def execute(self, **kwargs) -> str:
                return f"Processed: {kwargs['input']}"
    """

    name: str = ""
    description: str = ""
    parameters: dict = {}

    @abstractmethod
    def execute(self, **kwargs) -> str:
        """Run the tool with the given arguments.

        Args:
            **kwargs: Arguments matching the parameters schema.

        Returns:
            A string result to feed back to the LLM.
        """
        pass

    def to_schema(self) -> dict:
        """Convert tool to the function-calling schema format.

        Returns a dict compatible with Ollama/OpenAI function calling.
        """
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }
