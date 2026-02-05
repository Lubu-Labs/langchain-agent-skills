from __future__ import annotations

from typing import Annotated, Literal, TypedDict

from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages


class SupervisorState(TypedDict):
    """State for supervisor pattern."""

    messages: Annotated[list[BaseMessage], add_messages]
    next: Literal["researcher", "writer", "reviewer", "FINISH"]
    current_agent: str
    task_complete: bool


def supervisor_node(state: SupervisorState) -> dict:
    """Route to the next agent based on the latest message."""
    if state.get("task_complete"):
        return {"next": "FINISH", "current_agent": "supervisor"}

    last = state["messages"][-1].content.lower()
    if "research" in last:
        next_agent = "researcher"
    elif "write" in last:
        next_agent = "writer"
    elif "review" in last:
        next_agent = "reviewer"
    else:
        next_agent = "researcher"

    return {"next": next_agent, "current_agent": "supervisor"}


def researcher_node(state: SupervisorState) -> dict:
    """Researcher subagent."""
    return {
        "messages": [HumanMessage(content="Researcher: gathered quick notes.")],
        "current_agent": "researcher",
        "task_complete": True,
    }


def writer_node(state: SupervisorState) -> dict:
    """Writer subagent."""
    return {
        "messages": [HumanMessage(content="Writer: drafted a short response.")],
        "current_agent": "writer",
        "task_complete": True,
    }


def reviewer_node(state: SupervisorState) -> dict:
    """Reviewer subagent."""
    return {
        "messages": [HumanMessage(content="Reviewer: checked for clarity.")],
        "current_agent": "reviewer",
        "task_complete": True,
    }


def create_graph():
    graph = StateGraph(SupervisorState)

    graph.add_node("supervisor", supervisor_node)
    graph.add_node("researcher", researcher_node)
    graph.add_node("writer", writer_node)
    graph.add_node("reviewer", reviewer_node)

    graph.add_edge(START, "supervisor")

    graph.add_conditional_edges(
        "supervisor",
        lambda s: s["next"],
        {
            "researcher": "researcher",
            "writer": "writer",
            "reviewer": "reviewer",
            "FINISH": END,
        },
    )

    graph.add_edge("researcher", "supervisor")
    graph.add_edge("writer", "supervisor")
    graph.add_edge("reviewer", "supervisor")

    return graph.compile()


graph = create_graph()


if __name__ == "__main__":
    result = graph.invoke(
        {
            "messages": [HumanMessage(content="Please research the topic.")],
            "next": "researcher",
            "current_agent": "",
            "task_complete": False,
        }
    )

    print("Final messages:")
    for message in result["messages"]:
        print(f"- {message.content}")
