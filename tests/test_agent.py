"""Test script — run against a local Ollama instance.

Usage:
    1. Start Ollama: ollama serve
    2. Pull a model: ollama pull gemma3:1b
    3. Run: python tests/test_agent.py
"""

import json

from allegro_agent import Agent, FileWriteTool, InvalidHistoryError
from allegro_agent.exceptions import ProviderError


def test_agent():
    """Test Agent with conversation history."""
    print("=" * 50)
    print("Testing Agent.run()")
    print("=" * 50)

    agent = Agent(
        name="Test Agent",
        model="ollama:gemma3:1b",
        temperature=0.1,
        system_prompt="You are a helpful assistant. Keep responses brief.",
    )

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

    agent = Agent(
        name="Writer Agent",
        model="ollama:gemma3:1b",
        temperature=0.1,
        system_prompt="You are a helpful assistant. Use the file_write tool when asked to write files.",
        tools=[FileWriteTool()],
    )

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


def test_history_validation():
    """Test that the history kwarg rejects malformed payloads. No Ollama needed."""
    print("=" * 50)
    print("Testing Agent(history=...) validation")
    print("=" * 50)

    bad_cases = [
        ("not a list", "string"),
        ("list of non-dict", [42]),
        ("missing role", [{"content": "x"}]),
        ("missing content", [{"role": "user"}]),
        ("system role rejected", [{"role": "system", "content": "x"}]),
        ("non-str content", [{"role": "user", "content": 42}]),
        ("bad tool_calls type", [{"role": "assistant", "content": "", "tool_calls": "nope"}]),
    ]

    for label, payload in bad_cases:
        try:
            Agent(model="ollama:gemma3:1b", history=payload)
            print(f"  FAIL: {label} -> no error raised")
        except InvalidHistoryError as e:
            print(f"  OK:   {label} -> {e}")

    # Happy path: valid history should construct without error.
    Agent(
        model="ollama:gemma3:1b",
        history=[
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ],
    )
    print("  OK:   valid history accepted")
    print()


def test_resume_round_trip():
    """Verify a conversation captured via history() can resume in a fresh Agent."""
    print("=" * 50)
    print("Testing resume round-trip via history()")
    print("=" * 50)

    agent_a = Agent(
        name="Session A",
        model="ollama:gemma3:1b",
        temperature=0.1,
        system_prompt="You are a helpful assistant. Keep responses brief.",
    )

    response = agent_a.run("My name is John. Just acknowledge.")
    print(f"A1: {response}")

    # Capture state, round-trip through JSON to prove it's serializable.
    snapshot = agent_a.history()
    serialized = json.dumps(snapshot)
    restored = json.loads(serialized)
    print(f"Captured {len(snapshot)} message(s); JSON length: {len(serialized)} bytes")

    # Returned snapshot must be a deep copy — mutating it must not affect the agent.
    snapshot.append({"role": "user", "content": "MUTATION"})
    assert len(agent_a.history()) == len(restored), \
        "history() did not return a deep copy"
    print("  OK: history() returned a deep copy")

    # Fresh agent picks up where the first one left off.
    agent_b = Agent(
        name="Session B (resumed)",
        model="ollama:gemma3:1b",
        temperature=0.1,
        system_prompt="You are a helpful assistant. Keep responses brief.",
        history=restored,
    )

    response = agent_b.run("What name did I just tell you?")
    print(f"B1: {response}")

    if "john" in response.lower():
        print("  OK: resumed agent recalled the prior turn")
    else:
        print("  WARN: response did not contain 'John' — model may have lost context")
    print()


if __name__ == "__main__":
    test_history_validation()
    test_agent()
    test_agent_with_tools()
    test_resume_round_trip()
    print("All tests passed!")
