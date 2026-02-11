# Deep Agents Reference

Comprehensive reference for Deep Agents configuration, middleware, backends, and migration patterns.

## create_deep_agent API

### Essential Parameters

```python
from deepagents import create_deep_agent

agent = create_deep_agent(
    model="claude-sonnet-4-5-20250929",  # Tool-calling model
    tools=[tool1, tool2],                 # Optional custom tools
    system_prompt="Instructions",        # Optional custom system prompt
    middleware=[...],                     # Optional custom middleware
    subagents=[...],                      # Optional specialist subagents
    store=memory_store,                   # Optional long-term store
    checkpointer=checkpointer,            # Optional thread persistence
    backend=backend_fn,                   # Optional filesystem backend
)
```

**Returns**: `CompiledStateGraph` compatible with LangGraph streaming, persistence, and Studio tooling.

### Key Parameters

- `model`: Model string (including `provider:model` format) or chat model object
- `tools`: List of functions/tools available to the agent
- `system_prompt`: Instructions layered on top of Deep Agents defaults
- `middleware`: Additional middleware hooks
- `subagents`: Specialist subagents for delegation/context isolation
- `store`: LangGraph store for cross-thread memory
- `checkpointer`: Checkpointer for thread-level state persistence
- `backend`: Filesystem backend factory/object (state, store, local disk, or composite)

## Default Middleware

Deep Agents includes these middleware by default:

1. `TodoListMiddleware` (planning with `write_todos`)
2. `FilesystemMiddleware` (`ls`, `read_file`, `write_file`, `edit_file`)
3. `SubAgentMiddleware` (delegation via `task`)
4. `SummarizationMiddleware` (history compression)
5. `AnthropicPromptCachingMiddleware` (prompt caching)
6. `PatchToolCallsMiddleware` (tool-call correction)

Conditional middleware:

- `MemoryMiddleware` when `memory` is provided
- `SkillsMiddleware` when `skills` is provided
- `HumanInTheLoopMiddleware` when `interrupt_on` is provided

### Custom Middleware Example

```python
from langchain.tools import tool
from langchain.agents.middleware import wrap_tool_call
from deepagents import create_deep_agent

@tool
def get_weather(city: str) -> str:
    """Get weather in a city."""
    return f"The weather in {city} is sunny."

@wrap_tool_call
def log_tool_calls(request, handler):
    print(f"Tool call: {request.name}")
    return handler(request)

agent = create_deep_agent(
    tools=[get_weather],
    middleware=[log_tool_calls],
)
```

## Backends

### StateBackend (default)

- Files live in graph state
- Ephemeral per thread

### StoreBackend

- Files live in LangGraph store
- Persistent across threads
- Requires passing `store=` to `create_deep_agent`

### FilesystemBackend

- Uses local disk
- Use `virtual_mode=True` with `root_dir` for path restrictions
- Use cautiously in production-exposed environments

### CompositeBackend

- Routes path prefixes to different backends (common pattern: `/memories/` persistent, everything else ephemeral)

```python
from deepagents import create_deep_agent
from deepagents.backends import CompositeBackend, StateBackend, StoreBackend
from langgraph.store.memory import InMemoryStore

store = InMemoryStore()

agent = create_deep_agent(
    store=store,
    backend=lambda rt: CompositeBackend(
        default=StateBackend(rt),
        routes={"/memories/": StoreBackend(rt)},
    ),
)
```

## Checkpointers and Persistence

For short-term memory/thread persistence, pass a checkpointer and invoke with a `thread_id`:

```python
from deepagents import create_deep_agent
from langgraph.checkpoint.memory import InMemorySaver

agent = create_deep_agent(
    model="claude-sonnet-4-5-20250929",
    checkpointer=InMemorySaver(),
)

result = agent.invoke(
    {"messages": [{"role": "user", "content": "Hello"}]},
    config={"configurable": {"thread_id": "demo-thread"}},
)
```

## Migration Patterns

### LangChain `create_agent` -> `create_deep_agent`

```python
# Before
from langchain.agents import create_agent
agent = create_agent(model=model, tools=tools, system_prompt=prompt)

# After
from deepagents import create_deep_agent
agent = create_deep_agent(model=model, tools=tools, system_prompt=prompt)
```

### Legacy `create_react_agent`

`langgraph.prebuilt.create_react_agent` is deprecated in LangGraph v1. Prefer `langchain.agents.create_agent` or `deepagents.create_deep_agent` depending on whether you want the Deep Agents harness.

### Supervisor Graph -> Built-in Subagents

```python
agent = create_deep_agent(
    model="claude-sonnet-4-5-20250929",
    subagents=[
        {"name": "researcher", "description": "Research specialist", "tools": [...], "system_prompt": "..."},
        {"name": "coder", "description": "Code specialist", "tools": [...], "system_prompt": "..."},
    ],
)
```

## Common Patterns

### Minimal Agent

```python
agent = create_deep_agent(
    model="claude-sonnet-4-5-20250929",
    tools=[my_tool],
)
```

### Persistent Checkpoints

```python
from langgraph.checkpoint.sqlite import SqliteSaver

agent = create_deep_agent(
    model="claude-sonnet-4-5-20250929",
    checkpointer=SqliteSaver.from_conn_string("checkpoints.db"),
)
```

### Human-in-the-Loop

```python
agent = create_deep_agent(
    model="claude-sonnet-4-5-20250929",
    checkpointer=checkpointer,
    interrupt_on={
        "write_file": True,
        "edit_file": True,
        "read_file": False,
    },
)
```

## Troubleshooting

**Model/tool-calling issues**

- Use a tool-calling-capable model
- Prefer explicit model identifiers (including `provider:model` format)

**Filesystem behavior is unexpected**

- Confirm backend choice (`StateBackend`, `StoreBackend`, `FilesystemBackend`, or `CompositeBackend`)
- If using `StoreBackend`, verify `store=` is configured

**Subagent delegation is weak**

- Improve subagent descriptions/system prompts
- Ensure subagents have the right specialized tools

**Performance overhead**

- Deep Agents adds harness overhead by design
- For very simple flows, consider plain LangChain/LangGraph agents

## See Also

- [Deep Agents Overview (Python)](https://docs.langchain.com/oss/python/deepagents/overview)
- [Customize Deep Agents (Python)](https://docs.langchain.com/oss/python/deepagents/customization)
- [Deep Agents Backends (Python)](https://docs.langchain.com/oss/python/deepagents/backends)
- [Deep Agents Overview (JavaScript)](https://docs.langchain.com/oss/javascript/deepagents/overview)
- [langgraph-project-setup](../../langgraph-project-setup/SKILL.md)
- [langgraph-agent-patterns](../../langgraph-agent-patterns/SKILL.md)
- [langsmith-deployment](../../langsmith-deployment/SKILL.md)
