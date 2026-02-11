# Simple Deep Agent

A minimal Deep Agent setup with built-in planning, filesystem, and subagent capabilities.

## Features

- ✅ Automatic task planning with `write_todos` tool
- ✅ Filesystem context management with `ls`, `read_file`, `write_file`, `edit_file`
- ✅ Subagent delegation with `task` tool
- ✅ Custom tools (weather, search)

## Setup

1. **Install dependencies**:
   ```bash
   uv sync
   ```

2. **Configure environment**:
   ```bash
   cp .env.example .env
   # Edit .env and add your API keys
   ```

3. **Run the agent**:
   ```bash
   uv run agent.py
   ```

## Project Structure

```
.
├── agent.py          # Main agent definition
├── pyproject.toml    # Dependencies
├── .env.example      # Environment template
└── README.md         # This file
```

## Customization

### Add Custom Tools

```python
def my_custom_tool(arg: str) -> str:
    """Tool description."""
    return result

agent = create_deep_agent(
    model=model,
    tools=[get_weather, search_web, my_custom_tool],  # Add your tool
    system_prompt="...",
)
```

### Configure Middleware

```python
from langchain.agents.middleware import TodoListMiddleware
from deepagents.middleware.filesystem import FilesystemMiddleware

agent = create_deep_agent(
    model=model,
    tools=tools,
    middleware=[
        TodoListMiddleware(system_prompt="Custom planning instructions"),
        FilesystemMiddleware(),
        # SubAgentMiddleware omitted - no subagents
    ],
)
```

### Add Persistence

```python
from langgraph.checkpoint.memory import InMemorySaver

agent = create_deep_agent(
    model=model,
    tools=tools,
    checkpointer=InMemorySaver(),  # Add checkpointing
)

# Use with thread_id for multi-turn conversations
result = agent.invoke(
    {"messages": [...]},
    config={"configurable": {"thread_id": "user-123"}},
)
```

## Next Steps

- Add more tools for your use case
- Configure persistence with checkpointers
- Set up long-term memory with Memory Store
- Deploy with LangSmith (see langsmith-deployment skill)
