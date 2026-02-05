#!/usr/bin/env python3
"""
Generate supervisor-subagent graph boilerplate for LangGraph projects.

This script creates the basic structure for a supervisor-subagent pattern,
including state definition, supervisor node, subagent nodes, and graph assembly.

Usage:
    python3 generate_supervisor_graph.py <graph_name> [--subagents agent1,agent2,agent3] [--output output_dir] [--typescript]
"""

import argparse
import json
from pathlib import Path
from typing import List, Optional


def generate_python_state(graph_name: str, subagents: List[str]) -> str:
    """Generate Python state schema for supervisor pattern."""
    subagent_list = ', '.join(f'"{agent}"' for agent in subagents)
    return f'''"""State schema for {graph_name} supervisor graph."""
from typing import TypedDict, Literal, Annotated
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage


class {graph_name.title().replace('-', '')}State(TypedDict):
    """State for {graph_name} supervisor graph.

    Attributes:
        messages: Chat history with add_messages reducer
        next: Next agent to route to
        current_agent: Currently active agent
        task_complete: Whether the task is complete
    """
    messages: Annotated[list[BaseMessage], add_messages]
    next: Literal[{subagent_list}, "FINISH"]
    current_agent: str
    task_complete: bool
'''


def generate_python_supervisor(graph_name: str, subagents: List[str]) -> str:
    """Generate Python supervisor node implementation."""
    subagent_list = ', '.join(f'"{agent}"' for agent in subagents)
    subagent_desc = '\n        '.join(f'- {agent}: TODO: Describe {agent} capabilities' for agent in subagents)

    return f'''"""Supervisor node for {graph_name} graph."""
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from typing import Literal

# TODO: Import your LLM
# from langchain_openai import ChatOpenAI
# from langchain_anthropic import ChatAnthropic

# TODO: Initialize your LLM
# model = ChatOpenAI(model="gpt-4")
# model = ChatAnthropic(model="claude-3-5-sonnet-20241022")


SUPERVISOR_PROMPT = """You are a supervisor managing a team of agents.
Your role is to analyze the user's request and delegate tasks to the appropriate agent.

Available agents:
{subagent_desc}

Based on the conversation history and current task, decide which agent should act next.
If the task is complete, respond with "FINISH".

Consider:
1. What has been done so far
2. What still needs to be done
3. Which agent is best suited for the next step
"""


def supervisor_node(state: dict) -> dict:
    """Supervisor node that routes to appropriate subagent.

    Args:
        state: Current graph state

    Returns:
        Updated state with routing decision
    """
    messages = state["messages"]

    # TODO: Configure routing decision using LLM
    # Create prompt with system message and conversation history
    prompt = ChatPromptTemplate.from_messages([
        ("system", SUPERVISOR_PROMPT),
        MessagesPlaceholder(variable_name="messages"),
        ("system", "Who should act next? Respond with one of: {subagent_list}, FINISH")
    ])

    # TODO: Get routing decision from LLM
    # chain = prompt | model
    # response = chain.invoke({{"messages": messages}})
    # next_agent = response.content.strip()

    # Placeholder routing logic
    next_agent = "{subagents[0]}"  # TODO: Replace with LLM-based routing

    return {{
        "next": next_agent,
        "current_agent": "supervisor"
    }}
'''


def generate_python_subagent(agent_name: str) -> str:
    """Generate Python subagent node implementation."""
    return f'''"""Subagent node: {agent_name}."""
from langchain_core.messages import HumanMessage

# TODO: Import required tools and LLM for {agent_name}


def {agent_name}_node(state: dict) -> dict:
    """Node for {agent_name} agent.

    TODO: Implement {agent_name} logic:
    1. Extract relevant information from state
    2. Perform agent-specific tasks
    3. Use tools if needed
    4. Return updated state with results

    Args:
        state: Current graph state

    Returns:
        Updated state with {agent_name} results
    """
    messages = state["messages"]

    # TODO: Implement {agent_name} logic here
    result = f"{{agent_name}} processed the request"  # Placeholder

    return {{
        "messages": [HumanMessage(content=result)],
        "current_agent": "{agent_name}"
    }}
'''


def generate_python_graph(graph_name: str, subagents: List[str]) -> str:
    """Generate Python graph assembly code."""
    node_imports = '\n'.join(f'from .nodes import {agent}_node' for agent in subagents)
    add_nodes = '\n    '.join(f'graph.add_node("{agent}", {agent}_node)' for agent in subagents)
    add_edges = '\n    '.join(f'graph.add_edge("{agent}", "supervisor")' for agent in subagents)

    return f'''"""Graph assembly for {graph_name} supervisor pattern."""
from langgraph.graph import StateGraph, START, END
from .state import {graph_name.title().replace('-', '')}State
from .supervisor import supervisor_node
{node_imports}


def create_graph() -> StateGraph:
    """Create and compile the supervisor graph.

    Returns:
        Compiled StateGraph ready for execution
    """
    # Create graph with state schema
    graph = StateGraph({graph_name.title().replace('-', '')}State)

    # Add supervisor node
    graph.add_node("supervisor", supervisor_node)

    # Add subagent nodes
    {add_nodes}

    # Set entry point
    graph.add_edge(START, "supervisor")

    # Add conditional edges from supervisor to subagents
    graph.add_conditional_edges(
        "supervisor",
        lambda state: state["next"],
        {{
{chr(10).join(f'            "{agent}": "{agent}",' for agent in subagents)}
            "FINISH": END
        }}
    )

    # Add edges from subagents back to supervisor
    {add_edges}

    return graph.compile()


# Export the compiled graph
graph = create_graph()
'''


def generate_typescript_state(graph_name: str, subagents: List[str]) -> str:
    """Generate TypeScript state schema for supervisor pattern."""
    subagent_union = ' | '.join(f'"{agent}"' for agent in subagents)
    return f'''/**
 * State schema for {graph_name} supervisor graph.
 */
import {{ BaseMessage }} from "@langchain/core/messages";
import {{ Annotation }} from "@langchain/langgraph";

export const {graph_name.replace('-', '_').title()}StateAnnotation = Annotation.Root({{
  /**
   * Chat history with message reducer
   */
  messages: Annotation<BaseMessage[]>({{
    reducer: (left, right) => left.concat(right),
    default: () => [],
  }}),

  /**
   * Next agent to route to
   */
  next: Annotation<{subagent_union} | "FINISH">({{
    reducer: (_, right) => right,
    default: () => "{subagents[0]}",
  }}),

  /**
   * Currently active agent
   */
  currentAgent: Annotation<string>({{
    reducer: (_, right) => right,
    default: () => "supervisor",
  }}),

  /**
   * Whether the task is complete
   */
  taskComplete: Annotation<boolean>({{
    reducer: (_, right) => right,
    default: () => false,
  }}),
}});

export type {graph_name.replace('-', '_').title()}State = typeof {graph_name.replace('-', '_').title()}StateAnnotation.State;
'''


def generate_typescript_supervisor(graph_name: str, subagents: List[str]) -> str:
    """Generate TypeScript supervisor node implementation."""
    subagent_desc = '\\n        '.join(f'- {agent}: TODO: Describe {agent} capabilities' for agent in subagents)

    return f'''/**
 * Supervisor node for {graph_name} graph.
 */
import {{ HumanMessage }} from "@langchain/core/messages";
import {{ ChatPromptTemplate, MessagesPlaceholder }} from "@langchain/core/prompts";

// TODO: Import your LLM
// import {{ ChatOpenAI }} from "@langchain/openai";
// import {{ ChatAnthropic }} from "@langchain/anthropic";

// TODO: Initialize your LLM
// const model = new ChatOpenAI({{ model: "gpt-4" }});
// const model = new ChatAnthropic({{ model: "claude-3-5-sonnet-20241022" }});

const SUPERVISOR_PROMPT = `You are a supervisor managing a team of agents.
Your role is to analyze the user's request and delegate tasks to the appropriate agent.

Available agents:
{subagent_desc}

Based on the conversation history and current task, decide which agent should act next.
If the task is complete, respond with "FINISH".

Consider:
1. What has been done so far
2. What still needs to be done
3. Which agent is best suited for the next step
`;

export async function supervisorNode(state: any): Promise<Partial<any>> {{
  /**
   * Supervisor node that routes to appropriate subagent.
   */
  const messages = state.messages;

  // TODO: Configure routing decision using LLM
  // Create prompt with system message and conversation history
  const prompt = ChatPromptTemplate.fromMessages([
    ["system", SUPERVISOR_PROMPT],
    new MessagesPlaceholder("messages"),
    ["system", "Who should act next? Respond with one of: {', '.join(subagents)}, FINISH"]
  ]);

  // TODO: Get routing decision from LLM
  // const chain = prompt.pipe(model);
  // const response = await chain.invoke({{ messages }});
  // const nextAgent = response.content.trim();

  // Placeholder routing logic
  const nextAgent = "{subagents[0]}"; // TODO: Replace with LLM-based routing

  return {{
    next: nextAgent,
    currentAgent: "supervisor"
  }};
}}
'''


def generate_typescript_subagent(agent_name: str) -> str:
    """Generate TypeScript subagent node implementation."""
    return f'''/**
 * Subagent node: {agent_name}
 */
import {{ HumanMessage }} from "@langchain/core/messages";

// TODO: Import required tools and LLM for {agent_name}

export async function {agent_name}Node(state: any): Promise<Partial<any>> {{
  /**
   * Node for {agent_name} agent.
   *
   * TODO: Implement {agent_name} logic:
   * 1. Extract relevant information from state
   * 2. Perform agent-specific tasks
   * 3. Use tools if needed
   * 4. Return updated state with results
   */
  const messages = state.messages;

  // TODO: Implement {agent_name} logic here
  const result = `{agent_name} processed the request`; // Placeholder

  return {{
    messages: [new HumanMessage(result)],
    currentAgent: "{agent_name}"
  }};
}}
'''


def generate_typescript_graph(graph_name: str, subagents: List[str]) -> str:
    """Generate TypeScript graph assembly code."""
    node_imports = '\n'.join(f'import {{ {agent}Node }} from "./nodes";' for agent in subagents)
    add_nodes = '\n  '.join(f'.addNode("{agent}", {agent}Node)' for agent in subagents)
    routing_map = ',\n    '.join(f'{agent}: "{agent}"' for agent in subagents)
    add_edges = '\n  '.join(f'.addEdge("{agent}", "supervisor")' for agent in subagents)

    return f'''/**
 * Graph assembly for {graph_name} supervisor pattern.
 */
import {{ StateGraph, START, END }} from "@langchain/langgraph";
import {{ {graph_name.replace('-', '_').title()}StateAnnotation }} from "./state";
import {{ supervisorNode }} from "./supervisor";
{node_imports}

export function createGraph() {{
  /**
   * Create and compile the supervisor graph.
   */
  const graph = new StateGraph({{
    channels: {graph_name.replace('-', '_').title()}StateAnnotation,
  }})
    // Add supervisor node
    .addNode("supervisor", supervisorNode)

    // Add subagent nodes
    {add_nodes}

    // Set entry point
    .addEdge(START, "supervisor")

    // Add conditional edges from supervisor to subagents
    .addConditionalEdges(
      "supervisor",
      (state) => state.next,
      {{
        {routing_map},
        FINISH: END
      }}
    )

    // Add edges from subagents back to supervisor
    {add_edges};

  return graph.compile();
}}

// Export the compiled graph
export const graph = createGraph();
'''


def create_python_project(
    output_dir: Path,
    graph_name: str,
    subagents: List[str]
) -> None:
    """Create Python project structure with supervisor pattern."""
    package_name = graph_name.replace('-', '_')
    package_dir = output_dir / package_name
    package_dir.mkdir(parents=True, exist_ok=True)

    # Create __init__.py
    (package_dir / "__init__.py").write_text(
        f'"""Supervisor-subagent graph: {graph_name}."""\n'
        f'from .graph import graph\n\n'
        f'__all__ = ["graph"]\n'
    )

    # Create state.py
    (package_dir / "state.py").write_text(generate_python_state(graph_name, subagents))

    # Create supervisor.py
    (package_dir / "supervisor.py").write_text(generate_python_supervisor(graph_name, subagents))

    # Create nodes.py with all subagents
    nodes_content = f'"""Subagent nodes for {graph_name} graph."""\n\n'
    nodes_content += '\n\n'.join(generate_python_subagent(agent) for agent in subagents)
    (package_dir / "nodes.py").write_text(nodes_content)

    # Create graph.py
    (package_dir / "graph.py").write_text(generate_python_graph(graph_name, subagents))

    print(f"✅ Created Python supervisor graph at {package_dir}")
    print(f"\nNext steps:")
    print(f"1. Review and customize the generated files")
    print(f"2. Implement TODO items in supervisor.py and nodes.py")
    print(f"3. Configure your LLM in supervisor.py")
    print(f"4. Test the graph")


def create_typescript_project(
    output_dir: Path,
    graph_name: str,
    subagents: List[str]
) -> None:
    """Create TypeScript project structure with supervisor pattern."""
    src_dir = output_dir / "src"
    src_dir.mkdir(parents=True, exist_ok=True)

    # Create index.ts
    (src_dir / "index.ts").write_text(
        f'/**\n * Supervisor-subagent graph: {graph_name}\n */\n'
        f'export {{ graph }} from "./graph";\n'
    )

    # Create state.ts
    (src_dir / "state.ts").write_text(generate_typescript_state(graph_name, subagents))

    # Create supervisor.ts
    (src_dir / "supervisor.ts").write_text(generate_typescript_supervisor(graph_name, subagents))

    # Create nodes.ts with all subagents
    nodes_content = f'/**\n * Subagent nodes for {graph_name} graph.\n */\n\n'
    nodes_content += '\n\n'.join(generate_typescript_subagent(agent) for agent in subagents)
    (src_dir / "nodes.ts").write_text(nodes_content)

    # Create graph.ts
    (src_dir / "graph.ts").write_text(generate_typescript_graph(graph_name, subagents))

    print(f"✅ Created TypeScript supervisor graph at {src_dir}")
    print(f"\nNext steps:")
    print(f"1. Review and customize the generated files")
    print(f"2. Implement TODO items in supervisor.ts and nodes.ts")
    print(f"3. Configure your LLM in supervisor.ts")
    print(f"4. Test the graph")


def main():
    parser = argparse.ArgumentParser(
        description="Generate supervisor-subagent graph boilerplate for LangGraph"
    )
    parser.add_argument(
        "graph_name",
        help="Name of the graph (e.g., research-team, content-pipeline)"
    )
    parser.add_argument(
        "--subagents",
        default="researcher,writer,reviewer",
        help="Comma-separated list of subagent names (default: researcher,writer,reviewer)"
    )
    parser.add_argument(
        "--output",
        "-o",
        default=".",
        help="Output directory (default: current directory)"
    )
    parser.add_argument(
        "--typescript",
        action="store_true",
        help="Generate TypeScript code instead of Python"
    )

    args = parser.parse_args()

    output_dir = Path(args.output)
    subagents = [s.strip() for s in args.subagents.split(',')]

    if len(subagents) < 2:
        print("❌ Error: At least 2 subagents are required")
        return 1

    print(f"🚀 Generating supervisor graph: {args.graph_name}")
    print(f"   Subagents: {', '.join(subagents)}")
    print(f"   Language: {'TypeScript' if args.typescript else 'Python'}")
    print()

    if args.typescript:
        create_typescript_project(output_dir, args.graph_name, subagents)
    else:
        create_python_project(output_dir, args.graph_name, subagents)

    return 0


if __name__ == "__main__":
    exit(main())
