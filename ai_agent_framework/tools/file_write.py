import os

from .base import BaseTool


class FileWriteTool(BaseTool):
    """Tool that writes content to a file.

    Creates parent directories if they don't exist.
    """

    name = "file_write"
    description = (
        "Write content to a file at the given path. "
        "Creates parent directories if they don't exist. "
        "Overwrites the file if it already exists."
    )
    parameters = {
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "Path to the file to write.",
            },
            "content": {
                "type": "string",
                "description": "Content to write into the file.",
            },
        },
        "required": ["file_path", "content"],
    }

    def execute(self, **kwargs) -> str:
        file_path = kwargs["file_path"]
        content = kwargs["content"]

        parent = os.path.dirname(file_path)
        if parent:
            os.makedirs(parent, exist_ok=True)

        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)

        return f"Successfully wrote {len(content)} characters to {file_path}"
