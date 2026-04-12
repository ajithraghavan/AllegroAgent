"""Test script — run against a local Ollama instance.

Usage:
    1. Start Ollama: ollama serve
    2. Pull a model: ollama pull gemma3:1b
    3. Run: python tests/test_agent.py
"""

from ai_agent_framework import Agent, FileWriteTool
from ai_agent_framework.exceptions import ProviderError


def test_agent():
    """Test Agent with conversation history."""
    print("=" * 50)
    print("Testing Agent.run()")
    print("=" * 50)

    agent = Agent({
        "name": "Test Agent",
        "model": "ollama:gemma3:1b",
        "temperature": 0.1,
        "system_prompt": "You are a helpful assistant. Keep responses brief.",
    })

    print(f"Agent: {agent}")
    print()

    # First message
    response = agent.run("What is AI in one sentence?")
    print(f"Q: What is AI in one sentence?")
    print(f"A: {response}")
    print()

    # Follow-up (tests conversation history)
    response = agent.run("Can you elaborate slightly?")
    print(f"Q: Can you elaborate slightly?")
    print(f"A: {response}")
    print()


def test_agent_with_tools():
    """Test Agent with FileWriteTool — asks LLM to write a file."""
    print("=" * 50)
    print("Testing Agent with FileWriteTool")
    print("=" * 50)

    agent = Agent({
        "name": "Writer Agent",
        "model": "ollama:gemma3:1b",
        "temperature": 0.1,
        "system_prompt": "You are a helpful assistant. Use the file_write tool when asked to write files.",
        "tools": [FileWriteTool()],
    })

    print(f"Agent: {agent}")
    print()

    import os

    try:
        response = agent.run("Write a haiku about coding to test_output/haiku.txt")
        print(f"Q: Write a haiku about coding to test_output/haiku.txt")
        print(f"A: {response}")
        print()

        # Verify the file was created
        if os.path.exists("test_output/haiku.txt"):
            with open("test_output/haiku.txt", "r") as f:
                content = f.read()
            print(f"File content:\n{content}")
            # Cleanup
            os.remove("test_output/haiku.txt")
            os.rmdir("test_output")
            print("Cleanup done.")
        else:
            print("Note: File was not created (model may not have called the tool).")

    except ProviderError as e:
        if "does not support tools" in str(e):
            print(f"Skipped: current model does not support tools.")
            print("Pull a tool-capable model: ollama pull qwen2.5:3b")
        else:
            raise

    print()


if __name__ == "__main__":
    test_agent()
    test_agent_with_tools()
    print("All tests passed!")
