"""
Research agent state template.

Use this template for research agents that gather and process information.
"""

from typing import TypedDict, Annotated


def extend_list(left: list, right: list) -> list:
    """Extend list reducer."""
    return left + right


class ResearchState(TypedDict):
    """State for research agent."""

    # Input
    query: str

    # Research results (accumulated)
    search_results: Annotated[list[dict], extend_list]

    # Processing
    processed_sources: list[str]

    # Output
    summary: str
    citations: list[str]
