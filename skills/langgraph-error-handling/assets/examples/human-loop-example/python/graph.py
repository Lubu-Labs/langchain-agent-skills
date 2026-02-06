"""LangGraph human-in-the-loop approval example.

Demonstrates:
- interrupt() for human approval before sensitive operations
- Command(resume=...) for approve/reject flow
- Checkpointed pause/resume with thread_id
"""

from typing import Literal
from typing_extensions import NotRequired

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.types import Command, interrupt
from langgraph.checkpoint.memory import InMemorySaver


# --- State ---
class State(MessagesState):
    action: NotRequired[str]
    action_args: NotRequired[dict]
    result: NotRequired[str]


# --- Nodes ---
def plan_action(state: State):
    """Agent plans an action based on user request."""
    # Simulate planning; in practice this would be an LLM call.
    user_msg = str(state["messages"][-1].content) if state["messages"] else ""
    if "inactive" in user_msg.lower():
        return {
            "action": "delete_records",
            "action_args": {"table": "users", "filter": "inactive > 90 days"},
        }
    return {
        "action": "generate_report",
        "action_args": {"topic": "account-cleanup"},
    }


def human_review(state: State) -> Command[Literal["execute", "cancel"]]:
    """Pause for human approval before executing sensitive action."""
    is_approved = interrupt({
        "question": "Do you want to proceed with this action?",
        "action": state["action"],
        "args": state["action_args"],
    })

    if is_approved:
        return Command(goto="execute")
    else:
        return Command(goto="cancel")


def execute(state: State):
    """Execute the approved action."""
    action = state["action"]
    args = state["action_args"]
    # Simulate execution
    return {
        "result": f"Executed {action} with {args}",
        "messages": [AIMessage(content=f"Action completed: {action}")],
    }


def cancel(state: State):
    """Handle rejected action."""
    return {
        "result": "cancelled",
        "messages": [AIMessage(content="Action was rejected by reviewer.")],
    }


# --- Graph ---
builder = StateGraph(State)

builder.add_node("plan_action", plan_action)
builder.add_node("human_review", human_review)
builder.add_node("execute", execute)
builder.add_node("cancel", cancel)

builder.add_edge(START, "plan_action")
builder.add_edge("plan_action", "human_review")
# human_review uses Command for routing
builder.add_edge("execute", END)
builder.add_edge("cancel", END)

# A checkpointer is required for interrupt() pause/resume flows.
graph = builder.compile(checkpointer=InMemorySaver())


# --- Usage ---
if __name__ == "__main__":
    config = {"configurable": {"thread_id": "review-1"}}

    # Step 1: Run until interrupt
    result = graph.invoke(
        {"messages": [HumanMessage(content="Clean up inactive users")]},
        config,
    )
    print("Interrupted:", result.get("__interrupt__"))

    # Step 2: Resume with approval
    result = graph.invoke(Command(resume=True), config)
    for msg in result["messages"]:
        print(f"{msg.__class__.__name__}: {msg.content}")
