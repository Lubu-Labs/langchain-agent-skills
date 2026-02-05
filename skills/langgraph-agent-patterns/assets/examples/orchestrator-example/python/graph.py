from __future__ import annotations

from typing import Annotated, TypedDict
import operator

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send


class OrchestratorState(TypedDict):
    """State for orchestrator-worker pattern."""

    task: str
    subtasks: list[dict]
    results: Annotated[list[dict], operator.add]
    expected_count: int
    final_result: str


def orchestrator_node(state: OrchestratorState) -> dict:
    task = state["task"]
    subtasks = [
        {"id": 1, "query": f"Research part A of: {task}"},
        {"id": 2, "query": f"Research part B of: {task}"},
        {"id": 3, "query": f"Research part C of: {task}"},
    ]
    return {"subtasks": subtasks, "expected_count": len(subtasks)}


def worker_node(state: dict) -> dict:
    subtask = state["subtask"]
    result = f"Result for {subtask['query']}"
    return {"results": [{"id": subtask["id"], "result": result}]}


def aggregator_node(state: OrchestratorState) -> dict:
    results = state.get("results", [])
    expected = state.get("expected_count", 0)

    if expected and len(results) < expected:
        return {}

    final_result = "\n".join(r["result"] for r in results)
    return {"final_result": final_result}


def create_graph():
    graph = StateGraph(OrchestratorState)

    graph.add_node("orchestrator", orchestrator_node)
    graph.add_node("worker", worker_node)
    graph.add_node("aggregator", aggregator_node)

    graph.add_edge(START, "orchestrator")

    def dispatch_workers(state: OrchestratorState):
        return [Send("worker", {"subtask": subtask}) for subtask in state["subtasks"]]

    graph.add_conditional_edges("orchestrator", dispatch_workers)

    graph.add_edge("worker", "aggregator")
    graph.add_edge("aggregator", END)

    return graph.compile()


graph = create_graph()


if __name__ == "__main__":
    result = graph.invoke(
        {
            "task": "Prepare a short summary",
            "subtasks": [],
            "results": [],
            "expected_count": 0,
            "final_result": "",
        }
    )

    print("Final result:")
    print(result.get("final_result", "(no result)"))
