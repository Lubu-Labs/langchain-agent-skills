# Python Project Structure Examples

Standard directory layouts for LangGraph Python projects.

## Simple Agent Pattern

Recommended for single-agent applications with straightforward logic.

```
my-agent/
├── my_agent/              # Main package
│   ├── __init__.py
│   └── agent.py           # Graph definition
├── .env                   # Environment variables
├── .gitignore            # Git ignore rules
├── langgraph.json        # LangGraph config
└── pyproject.toml        # Python dependencies
```

**Use when:**
- Building a single agent
- Straightforward workflow
- Getting started quickly

## Multi-Agent Pattern

Recommended for complex applications with multiple agents or modular architecture.

```
my-agent/
├── my_agent/              # Main package
│   ├── utils/             # Utilities
│   │   ├── __init__.py
│   │   ├── state.py       # State definitions
│   │   ├── nodes.py       # Node functions
│   │   └── tools.py       # Tool definitions
│   ├── __init__.py
│   └── agent.py           # Graph builder
├── tests/                 # Test suite
│   ├── __init__.py
│   └── test_agent.py
├── .env                   # Environment variables
├── .gitignore            # Git ignore rules
├── langgraph.json        # LangGraph config
└── pyproject.toml        # Dependencies
```

**Use when:**
- Multiple agents or workers
- Complex state management
- Separated concerns (tools, nodes, state)

## With Requirements.txt

Alternative to pyproject.toml for simpler dependency management.

```
my-agent/
├── my_agent/
│   ├── __init__.py
│   └── agent.py
├── .env
├── .gitignore
├── langgraph.json
└── requirements.txt       # Instead of pyproject.toml
```

## Key Files

### __init__.py
Required for Python packages. Can be empty or contain package exports.

### agent.py
Contains the graph definition. Must export a compiled graph:

```python
from langgraph.graph import StateGraph

# ... build graph ...

graph = graph_builder.compile()  # Must be named 'graph'
```

### state.py (Multi-Agent)
State definitions using TypedDict:

```python
from typing_extensions import TypedDict
from typing import Annotated
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
```

### nodes.py (Multi-Agent)
Node functions that process state:

```python
def my_node(state: AgentState) -> dict:
    # Process state
    return {"messages": [response]}
```

### tools.py (Multi-Agent)
LangChain tool definitions:

```python
from langchain_core.tools import tool

@tool
def my_tool(input: str) -> str:
    """Tool description."""
    return result
```

## Dependencies Configuration

### pyproject.toml

```toml
[project]
name = "my-agent"
version = "0.1.0"
dependencies = [
    "langgraph>=1.1.0",
    "langchain-core>=1.1.0",
    "langchain-openai>=1.1.0",
]

[project.optional-dependencies]
dev = [
    "langgraph-cli[inmem]>=0.4.0",
    "pytest>=7.0.0",
]
```

### requirements.txt

```
langgraph>=1.1.0
langchain-core>=1.1.0
langchain-openai>=1.1.0
```

## Environment Variables

Standard .env file structure:

```bash
# LangSmith (optional)
LANGSMITH_API_KEY=your-key
LANGSMITH_TRACING=true
LANGSMITH_PROJECT=my-project

# LLM Provider
OPENAI_API_KEY=your-key
```

## References

- Use the init script: `uv run scripts/init_langgraph_project.py my-agent` (fallback: `python3 scripts/init_langgraph_project.py my-agent`)
- See langgraph-json-schema.md for config details
