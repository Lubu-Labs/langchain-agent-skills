#!/usr/bin/env python3
"""
Basic Deep Agent Example

A working end-to-end example demonstrating:
- Agent creation with create_deep_agent()
- Custom tool definition and usage
- Message-based invocation
- Automatic middleware (todos, filesystem, subagents)
"""

import os
from typing import Optional
from deepagents import create_deep_agent
from langchain_anthropic import ChatAnthropic
from langchain.tools import tool

from dotenv import load_dotenv
load_dotenv()


# Define custom tools
@tool
def get_weather(city: str, units: str = "fahrenheit") -> str:
    """
    Get current weather for a city.

    Args:
        city: Name of the city
        units: Temperature units (fahrenheit or celsius)

    Returns:
        Weather information as a string
    """
    # In a real implementation, this would call a weather API
    weather_data = {
        "san francisco": {"temp": 72, "condition": "Sunny"},
        "new york": {"temp": 68, "condition": "Cloudy"},
        "london": {"temp": 59, "condition": "Rainy"},
        "tokyo": {"temp": 75, "condition": "Clear"},
    }

    city_lower = city.lower()
    if city_lower in weather_data:
        data = weather_data[city_lower]
        temp = data["temp"]
        if units == "celsius":
            temp = round((temp - 32) * 5/9)
            unit_symbol = "°C"
        else:
            unit_symbol = "°F"
        return f"Weather in {city}: {data['condition']}, {temp}{unit_symbol}"
    else:
        return f"Weather data not available for {city}"


@tool
def calculate(expression: str) -> str:
    """
    Evaluate a mathematical expression.

    Args:
        expression: Mathematical expression to evaluate (e.g., "2 + 2", "10 * 5")

    Returns:
        Result of the calculation
    """
    try:
        # Safe evaluation of mathematical expressions
        # In production, use a proper math parser library
        allowed_chars = set("0123456789+-*/()., ")
        if not all(c in allowed_chars for c in expression):
            return "Error: Expression contains invalid characters"

        result = eval(expression, {"__builtins__": {}}, {})
        return f"Result: {result}"
    except Exception as e:
        return f"Error evaluating expression: {str(e)}"


@tool
def search_documentation(topic: str, language: Optional[str] = None) -> str:
    """
    Search programming documentation.

    Args:
        topic: Topic to search for (e.g., "async/await", "list comprehension")
        language: Programming language (e.g., "python", "javascript")

    Returns:
        Documentation summary
    """
    # In a real implementation, this would search actual documentation
    docs = {
        "async/await": {
            "python": "Async/await in Python allows you to write asynchronous code using async def and await keywords.",
            "javascript": "Async/await in JavaScript provides a cleaner syntax for working with Promises."
        },
        "list comprehension": {
            "python": "List comprehensions provide a concise way to create lists: [x**2 for x in range(10)]"
        },
        "destructuring": {
            "javascript": "Destructuring allows unpacking values from arrays or properties from objects: const { name, age } = person;"
        }
    }

    topic_lower = topic.lower()
    if topic_lower in docs:
        if language and language.lower() in docs[topic_lower]:
            return f"Documentation for '{topic}' in {language}:\n\n{docs[topic_lower][language.lower()]}"
        else:
            # Return first available documentation
            lang, doc = next(iter(docs[topic_lower].items()))
            return f"Documentation for '{topic}' ({lang}):\n\n{doc}"
    else:
        return f"No documentation found for '{topic}'"


# Create the Deep Agent
agent = create_deep_agent(
    model=ChatAnthropic(
        model="claude-sonnet-4-5-20250929",
        temperature=0.7,
        api_key=os.getenv("ANTHROPIC_API_KEY")
    ),
    tools=[get_weather, calculate, search_documentation],
    system_prompt="""You are a helpful AI assistant with access to various tools.

Available tools:
- get_weather: Get current weather for any city
- calculate: Evaluate mathematical expressions
- search_documentation: Search programming documentation

You also have automatic middleware providing:
- write_todos: Break down complex tasks into subtasks
- read_file/write_file: Read and write files for managing information
- task: Delegate work to subagents (when configured)

When the user asks a question:
1. Determine which tools are needed
2. Use the tools to gather information
3. Provide a clear, helpful response

For complex multi-step tasks, use the write_todos tool to break them down.""",
)


def run_example(query: str) -> str:
    """
    Run a single query against the agent.

    Args:
        query: User query to process

    Returns:
        Agent's response
    """
    result = agent.invoke({
        "messages": [
            {
                "role": "user",
                "content": query
            }
        ]
    })

    # Extract the final message content
    return result["messages"][-1].content


def interactive_mode():
    """Run the agent in interactive mode."""
    print("=== Basic Deep Agent - Interactive Mode ===")
    print("Ask me anything! (Type 'exit' to quit)\n")

    while True:
        try:
            query = input("You: ").strip()
            if query.lower() in ["exit", "quit", "q"]:
                print("Goodbye!")
                break

            if not query:
                continue

            print("\nAgent: ", end="", flush=True)
            response = run_example(query)
            print(response)
            print()

        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}\n")


if __name__ == "__main__":
    import sys

    # Check if API key is set
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("Error: ANTHROPIC_API_KEY environment variable not set")
        print("Set it with: export ANTHROPIC_API_KEY='your-key-here'")
        print("Or add it to a .env file in this directory.")
        sys.exit(1)

    # If arguments provided, run single query
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        print(f"Query: {query}\n")
        response = run_example(query)
        print(f"Response: {response}")
    else:
        # Otherwise, run example queries
        print("=== Basic Deep Agent Examples ===\n")

        # Example 1: Simple tool usage
        print("Example 1: Weather Query")
        print("-" * 50)
        response = run_example("What's the weather like in San Francisco?")
        print(response)
        print()

        # Example 2: Calculation
        print("Example 2: Mathematical Calculation")
        print("-" * 50)
        response = run_example("Calculate 15 * 23 + 47")
        print(response)
        print()

        # Example 3: Documentation search
        print("Example 3: Documentation Search")
        print("-" * 50)
        response = run_example("How do async/await work in Python?")
        print(response)
        print()

        # Example 4: Multiple tools
        print("Example 4: Multiple Tools")
        print("-" * 50)
        response = run_example(
            "What's the weather in Tokyo and New York? "
            "Also calculate what 20% of 150 is."
        )
        print(response)
        print()

        # Offer interactive mode
        print("\nWould you like to try interactive mode? (y/n): ", end="", flush=True)
        choice = input().strip().lower()
        if choice == "y":
            print()
            interactive_mode()
