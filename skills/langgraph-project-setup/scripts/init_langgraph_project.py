#!/usr/bin/env python3
"""
Initialize a new LangGraph project with proper structure and configuration.

Usage:
    uv run init_langgraph_project.py <project-name> [options]

Fallback:
    python3 init_langgraph_project.py <project-name> [options]

Options:
    --path PATH          Output directory (default: current directory)
    --pattern PATTERN    Project pattern: simple or multiagent (default: simple)
    --graph-name NAME    Graph name for langgraph.json (default: agent)
    --python-version VER Python version: 3.11, 3.12, or 3.13 (default: 3.11)
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Literal


def create_directory_structure(
    project_path: Path,
    pattern: Literal["simple", "multiagent"]
) -> None:
    """Create the LangGraph project directory structure."""

    # Create main package directory (use underscore for Python package naming)
    package_name = project_path.name.replace("-", "_")
    package_dir = project_path / package_name
    package_dir.mkdir(parents=True, exist_ok=True)

    # Create __init__.py for main package
    (package_dir / "__init__.py").write_text("")

    if pattern == "simple":
        # Simple pattern: just agent.py at package root
        (package_dir / "agent.py").write_text(get_simple_agent_template())
    else:
        # Multi-agent pattern: utils directory with separated concerns
        utils_dir = package_dir / "utils"
        utils_dir.mkdir(exist_ok=True)

        # Create __init__.py for utils
        (utils_dir / "__init__.py").write_text("")

        # Create modular files
        (utils_dir / "state.py").write_text(get_state_template())
        (utils_dir / "nodes.py").write_text(get_nodes_template())
        (utils_dir / "tools.py").write_text(get_tools_template())
        (package_dir / "agent.py").write_text(get_multiagent_template(package_name))


def create_langgraph_json(
    project_path: Path,
    graph_name: str,
    python_version: str
) -> None:
    """Create langgraph.json configuration file."""

    package_name = project_path.name.replace("-", "_")

    config = {
        "dependencies": ["."],
        "graphs": {
            graph_name: f"./{package_name}/agent.py:graph"
        },
        "env": ".env",
        "python_version": python_version
    }

    config_path = project_path / "langgraph.json"
    config_path.write_text(json.dumps(config, indent=2) + "\n")


def create_env_file(project_path: Path) -> None:
    """Create .env template file with common environment variables."""

    env_content = """# LangSmith Configuration (optional - for tracing and monitoring)
# LANGSMITH_API_KEY=your-api-key-here
# LANGSMITH_TRACING=true
# LANGSMITH_PROJECT=your-project-name

# LLM Provider API Keys (uncomment the ones you need)
# OPENAI_API_KEY=your-openai-api-key-here
# ANTHROPIC_API_KEY=your-anthropic-api-key-here
# GOOGLE_API_KEY=your-google-api-key-here

# AWS Bedrock (if using AWS)
# AWS_ACCESS_KEY_ID=your-aws-access-key-id
# AWS_SECRET_ACCESS_KEY=your-aws-secret-key
# AWS_DEFAULT_REGION=us-east-1

# Other API Keys
# TAVILY_API_KEY=your-tavily-api-key-here
"""

    env_path = project_path / ".env"
    env_path.write_text(env_content)


def create_pyproject_toml(project_path: Path) -> None:
    """Create pyproject.toml with dependencies."""

    package_name = project_path.name.replace("-", "_")

    content = f"""[project]
name = "{package_name}"
version = "0.1.0"
description = "LangGraph agent application"
requires-python = ">=3.10"
dependencies = [
    "langgraph>=1.1.0",
    "langchain-core>=1.1.0",
    "langchain>=1.1.0",
]

[project.optional-dependencies]
openai = ["langchain-openai>=1.1.0"]
anthropic = ["langchain-anthropic>=1.1.0"]
google = ["langchain-google-genai>=4.0.0"]
dev = [
    "langgraph-cli[inmem]>=0.4.0",
    "pytest>=7.0.0",
]

[build-system]
requires = ["setuptools>=61.0"]
build-backend = "setuptools.build_meta"
"""

    toml_path = project_path / "pyproject.toml"
    toml_path.write_text(content)


def create_gitignore(project_path: Path) -> None:
    """Create .gitignore file."""

    content = """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual environments
venv/
env/
ENV/
.venv

# Environment variables
.env
.env.local

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# LangGraph
.langgraph/
"""

    gitignore_path = project_path / ".gitignore"
    gitignore_path.write_text(content)


def get_simple_agent_template() -> str:
    """Return template for simple agent.py."""
    return '''"""Simple LangGraph agent."""

from typing import Annotated
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage


class State(TypedDict):
    """Agent state."""
    messages: Annotated[list[BaseMessage], add_messages]


def call_model(state: State) -> dict:
    """Call the language model."""
    # TODO: Replace with your LLM provider
    # from langchain_openai import ChatOpenAI
    # model = ChatOpenAI(model="gpt-4")
    # response = model.invoke(state["messages"])
    # return {"messages": [response]}

    raise NotImplementedError("Please configure your LLM provider")


# Build graph
graph_builder = StateGraph(State)
graph_builder.add_node("call_model", call_model)
graph_builder.add_edge(START, "call_model")
graph_builder.add_edge("call_model", END)

graph = graph_builder.compile()
'''


def get_multiagent_template(package_name: str) -> str:
    """Return template for multi-agent agent.py."""
    return f'''"""Multi-agent LangGraph application."""

from langgraph.graph import StateGraph, START, END
from {package_name}.utils.state import AgentState
from {package_name}.utils.nodes import supervisor, worker_1, worker_2


def build_graph() -> StateGraph:
    """Build the multi-agent graph."""

    graph_builder = StateGraph(AgentState)

    # Add nodes
    graph_builder.add_node("supervisor", supervisor)
    graph_builder.add_node("worker_1", worker_1)
    graph_builder.add_node("worker_2", worker_2)

    # Add edges
    graph_builder.add_edge(START, "supervisor")
    graph_builder.add_conditional_edges(
        "supervisor",
        lambda state: state.get("next_agent", "END"),
        {{
            "worker_1": "worker_1",
            "worker_2": "worker_2",
            "END": END
        }}
    )
    graph_builder.add_edge("worker_1", "supervisor")
    graph_builder.add_edge("worker_2", "supervisor")

    return graph_builder.compile()


graph = build_graph()
'''


def get_state_template() -> str:
    """Return template for state.py."""
    return '''"""State definitions for the agent."""

from typing import Annotated, Literal
from typing_extensions import TypedDict

from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage


class AgentState(TypedDict):
    """State for multi-agent system."""
    messages: Annotated[list[BaseMessage], add_messages]
    next_agent: Literal["worker_1", "worker_2", "END"]
'''


def get_nodes_template() -> str:
    """Return template for nodes.py."""
    return '''"""Node functions for the agent."""

from langchain_core.messages import HumanMessage, AIMessage
from .state import AgentState


def supervisor(state: AgentState) -> dict:
    """Supervisor node that routes to workers."""
    messages = state["messages"]

    # TODO: Implement supervisor logic
    # This is a placeholder that routes to worker_1
    # In practice, you might use an LLM to decide routing

    if len(messages) > 2:
        return {"next_agent": "END"}

    return {"next_agent": "worker_1"}


def worker_1(state: AgentState) -> dict:
    """First worker node."""
    # TODO: Implement worker 1 logic
    response = AIMessage(content="Worker 1 response")
    return {"messages": [response]}


def worker_2(state: AgentState) -> dict:
    """Second worker node."""
    # TODO: Implement worker 2 logic
    response = AIMessage(content="Worker 2 response")
    return {"messages": [response]}
'''


def get_tools_template() -> str:
    """Return template for tools.py."""
    return '''"""Tools for the agent."""

from langchain_core.tools import tool


@tool
def example_tool(query: str) -> str:
    """Example tool that returns the input query.

    Args:
        query: The input query string

    Returns:
        The input query as output
    """
    return f"Tool received: {query}"


# List of all tools
tools = [example_tool]
'''


def main():
    parser = argparse.ArgumentParser(
        description="Initialize a new LangGraph project"
    )
    parser.add_argument(
        "project_name",
        help="Name of the project (e.g., my-agent)"
    )
    parser.add_argument(
        "--path",
        default=".",
        help="Output directory (default: current directory)"
    )
    parser.add_argument(
        "--pattern",
        choices=["simple", "multiagent"],
        default="simple",
        help="Project pattern (default: simple)"
    )
    parser.add_argument(
        "--graph-name",
        default="agent",
        help="Graph name for langgraph.json (default: agent)"
    )
    parser.add_argument(
        "--python-version",
        choices=["3.11", "3.12", "3.13"],
        default="3.11",
        help="Python version (default: 3.11)"
    )

    args = parser.parse_args()

    # Create project directory
    base_path = Path(args.path).resolve()
    project_path = base_path / args.project_name

    if project_path.exists():
        print(f"❌ Error: Directory {project_path} already exists")
        sys.exit(1)

    print(f"🚀 Initializing LangGraph project: {args.project_name}")
    print(f"   Pattern: {args.pattern}")
    print(f"   Location: {project_path}")
    print()

    # Create all files and directories
    create_directory_structure(project_path, args.pattern)
    create_langgraph_json(project_path, args.graph_name, args.python_version)
    create_env_file(project_path)
    create_pyproject_toml(project_path)
    create_gitignore(project_path)

    print("✅ Created project structure")
    print("✅ Created langgraph.json")
    print("✅ Created .env template")
    print("✅ Created pyproject.toml")
    print("✅ Created .gitignore")
    print()
    print("📦 Next steps:")
    print(f"   1. cd {args.project_name}")
    print(f"   2. Create a virtual environment: uv venv --python {args.python_version}")
    print("      Fallback: python3 -m venv venv")
    print("   3. Activate it: source .venv/bin/activate")
    print("      Fallback: source venv/bin/activate")
    print("   4. Install dependencies: uv pip install -e '.[dev]'")
    print("      Fallback: pip install -e '.[dev]'")
    print("   5. Configure .env with your API keys")
    print("   6. Start development server: langgraph dev")
    print()
    print("🎯 Happy building!")


if __name__ == "__main__":
    main()
