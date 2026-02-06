"""
Tool-calling agent state template.

Use this template for agents that use tools with tracking and error handling.
"""

from typing import Any
from langgraph.graph import MessagesState


class ToolCallingState(MessagesState):
    """State for agent with tools."""

    # Tool tracking
    tool_calls_made: list[str]
    tool_results: dict[str, Any]

    # Control flow
    should_continue: bool
    max_iterations: int
    current_iteration: int
