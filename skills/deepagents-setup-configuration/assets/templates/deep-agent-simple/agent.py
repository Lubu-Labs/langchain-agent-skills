#!/usr/bin/env python3
"""
Simple Deep Agent Example

A basic Deep Agent with minimal configuration, demonstrating core functionality
with automatic middleware (todos, filesystem, subagents).
"""

import os
from deepagents import create_deep_agent
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic

# Load values from .env if present.
load_dotenv()


def get_weather(city: str) -> str:
    """
    Get current weather for a city.

    Args:
        city: Name of the city

    Returns:
        Weather information as a string
    """
    # In production, call a real weather API
    return f"Weather in {city}: Sunny, 72°F"


def search_web(query: str) -> str:
    """
    Search the web for information.

    Args:
        query: Search query

    Returns:
        Search results as a string
    """
    # In production, call a real search API
    return f"Search results for '{query}': [Example results...]"


# Create Deep Agent with automatic middleware
agent = create_deep_agent(
    model=ChatAnthropic(model="claude-sonnet-4-5-20250929"),
    tools=[get_weather, search_web],
    system_prompt="""You are a helpful AI assistant with access to weather and search tools.

For complex tasks, break them down using the write_todos tool.
Use the filesystem tools (read_file, write_file) to manage large amounts of information.
Delegate specialized tasks to subagents using the task tool when appropriate.""",
)


if __name__ == "__main__":
    if not os.getenv("ANTHROPIC_API_KEY") and not os.getenv("OPENAI_API_KEY"):
        print("Error: ANTHROPIC_API_KEY or OPENAI_API_KEY environment variable not set")
        print("Set it in .env or export it in your shell before running.")
        raise SystemExit(1)

    # Example usage
    result = agent.invoke({
        "messages": [
            {
                "role": "user",
                "content": "What's the weather in San Francisco? Also search for the latest tech news."
            }
        ]
    })

    print("\n=== Agent Response ===")
    print(result["messages"][-1].content)
