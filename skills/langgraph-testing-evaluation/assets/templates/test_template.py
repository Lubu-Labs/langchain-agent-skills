"""
Pytest template for LangGraph graph testing.

This template mirrors current LangGraph testing patterns from:
https://docs.langchain.com/oss/python/langgraph/test
"""

from collections.abc import Callable

import pytest
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict


class GraphState(TypedDict):
    my_key: str


def create_graph(transform: Callable[[str], str] | None = None) -> StateGraph:
    """Create an uncompiled graph so each test can compile with fresh state."""
    transform = transform or (lambda value: value.upper())

    def node1(state: GraphState) -> GraphState:
        return {"my_key": f"node1:{state['my_key']}"}

    def node2(state: GraphState) -> GraphState:
        return {"my_key": transform(state["my_key"])}

    graph = StateGraph(GraphState)
    graph.add_node("node1", node1)
    graph.add_node("node2", node2)
    graph.add_edge(START, "node1")
    graph.add_edge("node1", "node2")
    graph.add_edge("node2", END)
    return graph


@pytest.fixture
def compiled_graph():
    """Compile a fresh graph for each test."""
    checkpointer = MemorySaver()
    return create_graph().compile(checkpointer=checkpointer)


def test_individual_node_execution():
    """
    Unit-test a single node directly via compiled_graph.nodes.

    Note: invoking a node directly bypasses the graph checkpointer.
    """
    checkpointer = MemorySaver()
    compiled = create_graph().compile(checkpointer=checkpointer)
    result = compiled.nodes["node1"].invoke({"my_key": "initial_value"})
    assert result["my_key"] == "node1:initial_value"


def test_basic_graph_execution(compiled_graph):
    """Integration-style test across connected nodes."""
    result = compiled_graph.invoke(
        {"my_key": "initial_value"},
        config={"configurable": {"thread_id": "test-thread-1"}},
    )
    assert result["my_key"] == "NODE1:INITIAL_VALUE"


def test_custom_graph_behavior():
    """Example of injecting deterministic behavior for targeted tests."""
    custom = create_graph(transform=lambda value: f"[{value}]")
    compiled = custom.compile(checkpointer=MemorySaver())
    result = compiled.invoke(
        {"my_key": "initial_value"},
        config={"configurable": {"thread_id": "test-thread-2"}},
    )
    assert result["my_key"] == "[node1:initial_value]"


@pytest.mark.integration
def test_with_real_model_placeholder():
    """
    Optional pattern for real-LLM integration tests.

    Replace with your graph import and remove skip when API keys are configured.
    """
    pytest.skip("Integration test placeholder: requires real model credentials.")
