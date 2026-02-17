#!/usr/bin/env python3
"""
Todo-Driven Research Agent Example

Demonstrates effective use of write_todos for multi-step research tasks.
Shows the complete workflow:
1. Create initial todo list
2. Ask user approval
3. Update status as tasks complete
4. Re-send updated todos with write_todos as progress changes
"""

import os

from deepagents import create_deep_agent
from dotenv import load_dotenv
from langchain.tools import tool


# Define simple tools for demonstration
@tool
def search_documentation(query: str) -> str:
    """Search documentation for information."""
    # Simulated search results
    return f"""
Found 5 relevant documents for '{query}':
1. Deep Agents Overview - Planning with TodoListMiddleware
2. write_todos API Reference - Creating and managing todos
3. Example: Research Agent with Todos
4. Best Practices: Todo Granularity and Status Management
5. Troubleshooting: Common Todo Issues
"""


@tool
def read_document(title: str) -> str:
    """Read a document and extract key information."""
    # Simulated document content
    docs = {
        "Deep Agents Overview": "Deep Agents provide built-in planning via TodoListMiddleware. Use write_todos to break down complex tasks.",
        "write_todos API Reference": "write_todos accepts a list of todos. Each todo has 'content' and 'status' fields. Status can be pending, in_progress, or completed.",
        "Example: Research Agent with Todos": "Create todos for: search, read, analyze, synthesize. Update status as you complete each step.",
        "Best Practices": "Keep todos to 3-6 items. Always ask user approval. Update status promptly.",
        "Troubleshooting": "If todos aren't completing, check: (1) status updates, (2) system prompt guidance, (3) todo granularity.",
    }
    return docs.get(title, f"No content found for '{title}'")


@tool
def synthesize_findings(findings: list[str]) -> str:
    """Synthesize multiple findings into a coherent summary."""
    return f"Summary: Analyzed {len(findings)} findings. Key themes: planning, status management, best practices."


def main():
    """Run the todo-driven research agent."""
    load_dotenv()

    # Load API key from environment
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("Error: ANTHROPIC_API_KEY not set in environment")
        print("Copy .env.example to .env and add your API key")
        return

    # Create Deep Agent with TodoListMiddleware (included by default)
    agent = create_deep_agent(
        model="anthropic:claude-sonnet-4-5-20250929",
        tools=[search_documentation, read_document, synthesize_findings],
        system_prompt="""
You are a research assistant that uses todos for multi-step tasks.

For complex research tasks:
1. Call write_todos with your research plan (3-6 steps)
2. Ask the user: "Does this plan look good?"
3. Wait for user approval
4. Execute each step in order:
   - Update status to "in_progress" when starting a step
   - Do the work (search, read, analyze, etc.)
   - Update status to "completed" when finished
5. Re-send the full todo list with updated statuses as progress changes
6. Complete all todos before finishing

Example todo structure:
- Search for relevant documentation (pending → in_progress → completed)
- Read and extract key findings (pending → in_progress → completed)
- Analyze patterns and themes (pending → in_progress → completed)
- Synthesize into summary (pending → in_progress → completed)

IMPORTANT: Always update todo status as you work through the steps!
""",
    )

    # Example research query
    print("=" * 60)
    print("Todo-Driven Research Agent Example")
    print("=" * 60)
    print()
    print("Task: Summarize the provided Deep Agents notes and write_todos patterns")
    print()
    print("Expected workflow:")
    print("1. Agent creates todo list with write_todos")
    print("2. Agent asks: 'Does this plan look good?'")
    print("3. User approves (you can say 'yes' or 'proceed')")
    print("4. Agent executes todos step by step")
    print("5. Agent updates status for each completed todo")
    print("6. Agent finishes when all todos are completed")
    print()
    print("=" * 60)
    print()

    # Run the agent
    user_message = (
        "Using the available documentation tools, summarize Deep Agents planning "
        "best practices for write_todos and provide a concise checklist."
    )

    response = agent.invoke({"messages": [{"role": "user", "content": user_message}]})

    # Print final response
    print("\n" + "=" * 60)
    print("Final Response:")
    print("=" * 60)
    messages = response.get("messages") or []
    if not messages:
        print("No response")
        return

    final_message = messages[-1]
    final_content = getattr(final_message, "content", str(final_message))
    print(final_content)


if __name__ == "__main__":
    main()
