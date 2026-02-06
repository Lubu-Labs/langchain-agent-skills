"""
Workflow orchestration state template.

Use this template for task-based workflows with status tracking.
"""

from typing import TypedDict, Literal, Any


class WorkflowState(TypedDict):
    """State for task-based workflow."""

    # Task definition
    task: str

    # Status tracking
    status: Literal["pending", "in_progress", "completed", "failed"]
    current_step: str

    # Progress
    steps_completed: list[str]

    # Results
    results: dict[str, Any]

    # Error handling
    errors: list[str]
