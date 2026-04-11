"""Test script — run against a local Ollama instance.

Usage:
    1. Start Ollama: ollama serve
    2. Pull a model: ollama pull gemma3:1b
    3. Run: python tests/test_agent.py
"""

from ai_agent_framework import Client, Agent


def test_client():
    """Test Layer 2: Client direct chat."""
    print("=" * 50)
    print("Testing Client.chat()")
    print("=" * 50)

    client = Client()
    response = client.chat(
        model="ollama:gemma3:1b",
        messages=[{"role": "user", "content": "Say hello in one sentence."}],
        temperature=0.1,
    )

    print(f"Response: {response.content}")
    print(f"Model: {response.model}")
    print(f"Provider: {response.provider}")
    print(f"Usage: {response.usage}")
    print()


def test_agent():
    """Test Layer 3: Agent with conversation history."""
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


if __name__ == "__main__":
    test_client()
    test_agent()
    print("All tests passed!")
