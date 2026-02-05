from __future__ import annotations

from typing import Annotated, Literal, TypedDict

from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages


def merge_dict(left: dict, right: dict) -> dict:
    return {**left, **right}


class HandoffState(TypedDict):
    """State for handoff pattern."""

    messages: Annotated[list[BaseMessage], add_messages]
    next_agent: Literal["researcher", "writer", "editor", "FINISH"]
    context: Annotated[dict, merge_dict]
    current_agent: str


def researcher_node(state: HandoffState) -> dict:
    return {
        "messages": [HumanMessage(content="Researcher: gathered key points.")],
        "context": {"research": "Key points"},
        "next_agent": "writer",
        "current_agent": "researcher",
    }


def writer_node(state: HandoffState) -> dict:
    research = state["context"].get("research", "")
    return {
        "messages": [HumanMessage(content=f"Writer: drafted using {research}.")],
        "context": {"draft": "Draft content"},
        "next_agent": "editor",
        "current_agent": "writer",
    }


def editor_node(state: HandoffState) -> dict:
    return {
        "messages": [HumanMessage(content="Editor: polished the draft.")],
        "context": {"final": "Final content"},
        "next_agent": "FINISH",
        "current_agent": "editor",
    }


def create_graph():
    graph = StateGraph(HandoffState)

    graph.add_node("researcher", researcher_node)
    graph.add_node("writer", writer_node)
    graph.add_node("editor", editor_node)

    graph.add_edge(START, "researcher")

    graph.add_conditional_edges(
        "researcher",
        lambda s: s["next_agent"],
        {"writer": "writer", "FINISH": END},
    )
    graph.add_conditional_edges(
        "writer",
        lambda s: s["next_agent"],
        {"editor": "editor", "FINISH": END},
    )
    graph.add_conditional_edges(
        "editor",
        lambda s: s["next_agent"],
        {"FINISH": END},
    )

    return graph.compile()


graph = create_graph()


if __name__ == "__main__":
    result = graph.invoke(
        {
            "messages": [HumanMessage(content="Create a short summary.")],
            "next_agent": "researcher",
            "context": {},
            "current_agent": "",
        }
    )

    print("Final context:")
    print(result.get("context", {}))
    print("Messages:")
    for message in result["messages"]:
        print(f"- {message.content}")
