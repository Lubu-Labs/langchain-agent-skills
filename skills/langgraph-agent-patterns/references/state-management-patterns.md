# State Management Patterns for Multi-Agent Systems

State schemas must support the coordination requirements of each multi-agent pattern. This reference covers state design for supervisor, router, orchestrator-worker, and handoff patterns.

## Supervisor Pattern State

### Basic Supervisor State

```python
from typing import TypedDict, Literal, Annotated
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage

class SupervisorState(TypedDict):
    """State for supervisor pattern."""
    # Message history with reducer
    messages: Annotated[list[BaseMessage], add_messages]

    # Routing decision
    next: Literal["agent1", "agent2", "agent3", "FINISH"]

    # Track current agent
    current_agent: str

    # Completion flag
    task_complete: bool
```

**TypeScript:**
```typescript
import { Annotation, messagesStateReducer } from "@langchain/langgraph";
import { BaseMessage } from "@langchain/core/messages";

const SupervisorState = Annotation.Root({
  messages: Annotation<BaseMessage[]>({
    reducer: messagesStateReducer,
    default: () => [],
  }),
  next: Annotation<"agent1" | "agent2" | "agent3" | "FINISH">(),
  currentAgent: Annotation<string>(),
  taskComplete: Annotation<boolean>(),
});
```

### Enhanced Supervisor State

Add metadata and context tracking:

```python
class EnhancedSupervisorState(TypedDict):
    """Supervisor state with enhanced tracking."""
    messages: Annotated[list[BaseMessage], add_messages]
    next: Literal["researcher", "writer", "reviewer", "FINISH"]
    current_agent: str

    # Agent history
    agent_history: list[str]

    # Shared context
    context: dict

    # Metadata
    start_time: float
    iteration_count: int
    cost_estimate: float
```

## Router Pattern State

### Basic Router State

```python
class RouterState(TypedDict):
    """State for router pattern."""
    # Input message
    messages: list[BaseMessage]

    # One-time routing decision
    route: Literal["sales", "support", "billing"]

    # Optional: routing confidence
    confidence: float
```

### Multi-Label Router State

Support routing to multiple agents:

```python
class MultiLabelRouterState(TypedDict):
    """Router state for multi-label routing."""
    messages: list[BaseMessage]

    # Multiple routes possible
    routes: list[str]

    # Confidence per route
    route_confidence: dict[str, float]

    # Primary route
    primary_route: str
```

## Orchestrator-Worker Pattern State

### Orchestrator State

```python
class OrchestratorState(TypedDict):
    """State for orchestrator-worker pattern."""
    # Original task
    task: str

    # Decomposed subtasks
    subtasks: list[dict]

    # Worker results (accumulated)
    results: Annotated[list[str], lambda x, y: x + y]

    # Aggregated final result
    final_result: str

    # Progress tracking
    completed_count: int
    total_count: int
```

### Worker State

```python
class WorkerState(TypedDict):
    """State for individual worker."""
    # Assigned subtask
    subtask: dict

    # Worker result
    result: str

    # Status tracking
    status: Literal["pending", "processing", "complete", "failed"]
    error: str | None
```

**TypeScript:**
```typescript
const OrchestratorState = Annotation.Root({
  task: Annotation<string>(),
  subtasks: Annotation<Array<any>>(),
  results: Annotation<string[]>({
    reducer: (left, right) => left.concat(right),
  }),
  finalResult: Annotation<string>(),
  completedCount: Annotation<number>(),
  totalCount: Annotation<number>(),
});
```

## Handoff Pattern State

### Basic Handoff State

```python
class HandoffState(TypedDict):
    """State for handoff pattern."""
    # Message history
    messages: Annotated[list[BaseMessage], add_messages]

    # Current agent
    current_agent: str

    # Next agent to hand off to
    next_agent: Literal["researcher", "writer", "editor", "FINISH"]

    # Shared context across handoffs
    context: dict
```

### Structured Handoff State

Use structured context:

```python
from pydantic import BaseModel

class HandoffContext(BaseModel):
    """Structured context for handoffs."""
    research_data: list[str] = []
    draft_versions: list[str] = []
    review_notes: list[str] = []
    metadata: dict = {}

class StructuredHandoffState(TypedDict):
    """Handoff state with structured context."""
    messages: Annotated[list[BaseMessage], add_messages]
    current_agent: str
    next_agent: str

    # Structured context
    context: HandoffContext

    # Handoff history
    handoff_history: list[str]
```

## Common State Patterns

### 1. Message Accumulation

Use `add_messages` reducer for chat history:

```python
from langgraph.graph.message import add_messages

class ChatState(TypedDict):
    """State with message accumulation."""
    messages: Annotated[list[BaseMessage], add_messages]
```

The `add_messages` reducer:
- Appends new messages to existing list
- Handles message updates by ID
- Supports `RemoveMessage`-based deletions when you need to prune history

### 2. List Accumulation

Accumulate results from multiple agents:

```python
def extend_list(left: list, right: list) -> list:
    """Reducer that extends lists."""
    return left + right

class AccumulatorState(TypedDict):
    """State with list accumulation."""
    results: Annotated[list[str], extend_list]
```

### 3. Dictionary Merging

Merge context dictionaries:

```python
def merge_dicts(left: dict, right: dict) -> dict:
    """Reducer that merges dicts."""
    return {**left, **right}

class MergeState(TypedDict):
    """State with dict merging."""
    context: Annotated[dict, merge_dicts]
```

### 4. Conditional Updates

Only update if value is present:

```python
def conditional_update(left: str, right: str | None) -> str:
    """Update only if right is not None."""
    return right if right is not None else left

class ConditionalState(TypedDict):
    """State with conditional updates."""
    status: Annotated[str, conditional_update]
```

## State Validation

### Pydantic Validation

Use Pydantic for state validation:

```python
from pydantic import BaseModel, validator

class ValidatedState(BaseModel):
    """State with validation."""
    messages: list[BaseMessage]
    current_agent: str
    iteration_count: int

    @validator("iteration_count")
    def check_iterations(cls, v):
        """Prevent infinite loops."""
        if v > 10:
            raise ValueError("Too many iterations")
        return v

    @validator("current_agent")
    def check_agent(cls, v):
        """Validate agent name."""
        valid_agents = {"researcher", "writer", "editor"}
        if v not in valid_agents:
            raise ValueError(f"Invalid agent: {v}")
        return v
```
**Note:** For `StateGraph` schemas, Pydantic validation is applied to the initial input state only. The graph output is not returned as a Pydantic instance, and subsequent node updates are not re-validated.

### Custom Validation

Add validation in nodes:

```python
def validate_state(state: SupervisorState) -> None:
    """Validate state before processing."""
    if not state.get("messages"):
        raise ValueError("Messages cannot be empty")

    if state.get("iteration_count", 0) > 20:
        raise ValueError("Maximum iterations exceeded")

def supervisor_node(state: SupervisorState) -> dict:
    """Supervisor with state validation."""
    validate_state(state)

    # Continue processing...
```

## State Size Management

### Context Summarization

Prevent state from growing too large:

```python
def summarize_messages(messages: list[BaseMessage], max_length: int = 10) -> list[BaseMessage]:
    """Summarize messages if too long."""
    if len(messages) <= max_length:
        return messages

    # Keep first and last messages, summarize middle
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import SystemMessage

    model = ChatOpenAI(model="gpt-4o-mini")
    summary_prompt = f"Summarize these messages: {messages[1:-5]}"
    summary = model.invoke(summary_prompt).content

    return [
        messages[0],
        SystemMessage(content=f"Summary: {summary}"),
        *messages[-5:]
    ]

def supervisor_with_summarization(state: SupervisorState) -> dict:
    """Supervisor that manages state size."""
    messages = state["messages"]

    if len(messages) > 15:
        messages = summarize_messages(messages)

    # Continue with summarized messages...
```

### Selective State Updates

Only update necessary fields:

```python
def minimal_update_agent(state: SupervisorState) -> dict:
    """Agent that updates only necessary fields."""
    # Don't re-send entire state
    return {
        "next": "next_agent",  # Only routing decision
        # No need to resend messages, context, etc.
    }
```

## Pattern-Specific Considerations

### Supervisor Pattern

Key requirements:
- Message accumulation with `add_messages`
- Routing decision field (`next`)
- Agent tracking (`current_agent`)
- Loop prevention (iteration counter)

```python
class SupervisorState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    next: str
    current_agent: str
    iteration_count: int  # For loop detection
```

### Router Pattern

Key requirements:
- Input messages
- One-time routing decision
- Optional confidence scores

```python
class RouterState(TypedDict):
    messages: list[BaseMessage]
    route: str
    confidence: float  # Optional
```

### Orchestrator-Worker

Key requirements:
- Task and subtasks
- Result accumulation
- Progress tracking

```python
class OrchestratorState(TypedDict):
    task: str
    subtasks: list[dict]
    results: Annotated[list[str], extend_list]
    completed: int
    total: int
```

### Handoff Pattern

Key requirements:
- Message accumulation
- Handoff target (`next_agent`)
- Shared context
- Handoff history (for loop prevention)

```python
class HandoffState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    next_agent: str
    context: dict
    handoff_history: list[str]
```

## TypeScript State Patterns

### Annotation API

```typescript
import { Annotation, messagesStateReducer } from "@langchain/langgraph";
import { BaseMessage } from "@langchain/core/messages";

// Message accumulation
const StateWithMessages = Annotation.Root({
  messages: Annotation<BaseMessage[]>({
    reducer: messagesStateReducer,
    default: () => [],
  }),
});

// Dictionary merging
const StateWithContext = Annotation.Root({
  context: Annotation<Record<string, any>>({
    reducer: (left, right) => ({ ...left, ...right }),
    default: () => ({}),
  }),
});

// Simple replacement
const StateWithRoute = Annotation.Root({
  route: Annotation<string>({
    reducer: (_, right) => right,
  }),
});
```

## Testing State Management

### Test Reducers

```python
def test_message_reducer():
    """Test message accumulation."""
    from langgraph.graph.message import add_messages
    from langchain_core.messages import HumanMessage

    existing = [HumanMessage(content="Hello")]
    new_msgs = [HumanMessage(content="World")]

    result = add_messages(existing, new_msgs)

    assert len(result) == 2
    assert result[1].content == "World"

def test_custom_reducer():
    """Test custom reducer."""
    def extend_list(left, right):
        return left + right

    existing = ["a", "b"]
    new = ["c", "d"]

    result = extend_list(existing, new)

    assert result == ["a", "b", "c", "d"]
```

### Test State Updates

```python
def test_state_updates():
    """Test state updates work correctly."""
    initial_state = {
        "messages": [],
        "next": "",
        "current_agent": ""
    }

    update = supervisor_node(initial_state)

    assert "next" in update
    assert update["next"] in ["agent1", "agent2", "FINISH"]
```

## Additional Resources

- LangGraph Graph API (Python): https://docs.langchain.com/oss/python/langgraph/graph-api
- LangGraph Graph API (JavaScript): https://docs.langchain.com/oss/javascript/langgraph/graph-api
- Reducers in LangGraph: https://docs.langchain.com/oss/python/langgraph/use-graph-api
