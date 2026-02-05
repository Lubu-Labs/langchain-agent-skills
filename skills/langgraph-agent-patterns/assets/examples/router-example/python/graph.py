from __future__ import annotations

from typing import Annotated, Literal, TypedDict

from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages


class RouterState(TypedDict):
    """State for router pattern."""

    messages: Annotated[list[BaseMessage], add_messages]
    route: Literal["sales", "support", "billing"]


def router_node(state: RouterState) -> dict:
    query = state["messages"][0].content.lower()

    if any(word in query for word in ["buy", "purchase", "price", "pricing"]):
        route = "sales"
    elif any(word in query for word in ["help", "problem", "issue"]):
        route = "support"
    elif any(word in query for word in ["invoice", "payment", "billing"]):
        route = "billing"
    else:
        route = "support"

    return {"route": route}


def sales_agent(state: RouterState) -> dict:
    return {"messages": [HumanMessage(content="Sales: Here's pricing info.")]} 


def support_agent(state: RouterState) -> dict:
    return {"messages": [HumanMessage(content="Support: Let's troubleshoot.")]} 


def billing_agent(state: RouterState) -> dict:
    return {"messages": [HumanMessage(content="Billing: Here's your invoice.")]} 


def create_graph():
    graph = StateGraph(RouterState)

    graph.add_node("router", router_node)
    graph.add_node("sales", sales_agent)
    graph.add_node("support", support_agent)
    graph.add_node("billing", billing_agent)

    graph.add_edge(START, "router")

    graph.add_conditional_edges(
        "router",
        lambda s: s["route"],
        {
            "sales": "sales",
            "support": "support",
            "billing": "billing",
        },
    )

    graph.add_edge("sales", END)
    graph.add_edge("support", END)
    graph.add_edge("billing", END)

    return graph.compile()


graph = create_graph()


if __name__ == "__main__":
    result = graph.invoke(
        {
            "messages": [HumanMessage(content="Can I see pricing?")],
            "route": "support",
        }
    )

    print("Final messages:")
    for message in result["messages"]:
        print(f"- {message.content}")
