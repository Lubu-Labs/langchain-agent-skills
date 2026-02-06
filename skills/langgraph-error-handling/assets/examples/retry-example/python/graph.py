"""LangGraph retry example with RetryPolicy and state-based error recovery.

Demonstrates:
- RetryPolicy for transient errors (API calls)
- Recovery loop with Command routing from the agent node
- Retry count tracking in graph state
"""

from typing import Literal
from typing_extensions import NotRequired

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.types import RetryPolicy, Command
from langgraph.checkpoint.memory import InMemorySaver


# --- State ---
class State(MessagesState):
    search_result: NotRequired[str]
    error: NotRequired[str]
    retry_count: NotRequired[int]


# --- Nodes ---
def agent(state: State) -> Command[Literal["search", "__end__"]]:
    """Agent decides whether to search or finish."""
    error = state.get("error", "")
    retry_count = state.get("retry_count", 0)

    if retry_count >= 3:
        return Command(
            update={"messages": [AIMessage(content=f"Search failed after 3 retries: {error}")]},
            goto=END,
        )

    if error:
        # In a real app this node would call an LLM to adapt strategy.
        return Command(
            update={"messages": [AIMessage(content=f"Search failed ({error}), retrying with different query...")]},
            goto="search",
        )

    if state.get("search_result"):
        return Command(
            update={"messages": [AIMessage(content=f"Found: {state['search_result']}")]},
            goto=END,
        )

    return Command(goto="search")


def search(state: State):
    """Search node with error handling; transient errors are retried automatically."""
    try:
        # Simulate search — replace with actual API call
        query = state["messages"][-1].content if state["messages"] else "default"
        result = f"Results for: {query}"
        return {"search_result": result, "error": ""}
    except Exception as e:
        # Non-transient errors: store in state for LLM recovery
        return Command(
            update={
                "error": str(e),
                "retry_count": state.get("retry_count", 0) + 1,
            },
            goto="agent",
        )


# --- Graph ---
builder = StateGraph(State)

builder.add_node("agent", agent)
builder.add_node(
    "search",
    search,
    # RetryPolicy handles transient errors (network, rate limits) automatically
    retry_policy=RetryPolicy(max_attempts=3, initial_interval=1.0),
)

builder.add_edge(START, "agent")
builder.add_edge("search", "agent")
# `agent` uses Command for dynamic routing.

graph = builder.compile(checkpointer=InMemorySaver())


# --- Usage ---
if __name__ == "__main__":
    result = graph.invoke(
        {"messages": [HumanMessage(content="Search for LangGraph tutorials")]},
        {"configurable": {"thread_id": "demo-1"}},
    )
    for msg in result["messages"]:
        print(f"{msg.__class__.__name__}: {msg.content}")
