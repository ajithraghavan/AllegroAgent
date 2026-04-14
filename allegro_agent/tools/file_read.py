from .base import BaseTool


class FileReadTool(BaseTool):
    """Tool that reads the contents of a file."""

    name = "file_read"
    description = (
        "Read the contents of a file at the given path and return it as a string."
    )
    parameters = {
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "Path to the file to read.",
            },
        },
        "required": ["file_path"],
    }

    def execute(self, **kwargs) -> str:
        file_path = kwargs["file_path"]

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()
        except FileNotFoundError:
            return f"File not found: {file_path}"
