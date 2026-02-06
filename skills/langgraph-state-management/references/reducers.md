# State Reducers in LangGraph

Reducers control how state updates are merged when multiple nodes write to the same field. Understanding reducers is essential for managing state in multi-agent systems.

## What Are Reducers?

A reducer is a function that takes two values (existing state and new update) and returns the merged result:

```python
def reducer(existing_value: T, new_value: T) -> T:
    """Merge existing state with new update."""
    return merged_value
```

**When reducers are called:**
- A node returns a partial state update
- For each field in the update, LangGraph applies the field's reducer
- The reducer merges the existing value with the new value
- The merged result becomes the new state

## Built-in Reducers

### add_messages Reducer

The most commonly used reducer for chat applications. Handles message accumulation with smart merging.

**Python:**
```python
from langgraph.graph import MessagesState
from langchain_core.messages import AIMessage

class ChatState(MessagesState):
    """State with message accumulation via built-in add_messages reducer."""

# Usage in node
def chat_node(state: ChatState) -> dict:
    """Add a message to the conversation."""
    return {
        "messages": [AIMessage(content="Hello!")]
    }
```

**TypeScript:**
```typescript
import { StateSchema, MessagesValue } from "@langchain/langgraph";

const ChatState = new StateSchema({
  messages: MessagesValue,
});
```

**Features of add_messages:**
- Appends new messages to the list
- Updates existing messages by ID
- Deletes messages when passed with special deletion marker
- Preserves message order

**Advanced usage:**
```python
from langchain_core.messages import HumanMessage, RemoveMessage

# Append messages
{"messages": [HumanMessage(content="New message")]}

# Update message by ID
{"messages": [AIMessage(content="Updated", id="msg_123")]}

# Delete messages
{"messages": [RemoveMessage(id="msg_456")]}

# Delete ALL messages
from langgraph.graph.message import REMOVE_ALL_MESSAGES
{"messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES)]}
```

### Bypass Reducers with Overwrite

Use `Overwrite` to bypass a reducer and directly replace a state value:

```python
from langgraph.types import Overwrite

class State(TypedDict):
    messages: Annotated[list, operator.add]

def replace_messages(state: State):
    # Bypass the reducer — replaces the entire list
    return {"messages": Overwrite(["replacement message"])}

# Alternative JSON format:
def replace_messages_json(state: State):
    return {"messages": {"__overwrite__": ["replacement message"]}}
```

> **Note:** When nodes execute in parallel, only one node can use `Overwrite` on the same state key in a given super-step. Multiple overwrites on the same key will raise `InvalidUpdateError`.

### operator.add Reducer

Simple addition, useful for counters and numeric accumulation:

```python
from typing import TypedDict, Annotated
import operator

class CounterState(TypedDict):
    """State with numeric accumulation."""
    count: Annotated[int, operator.add]
    total_cost: Annotated[float, operator.add]

# Usage
def increment_node(state: CounterState) -> dict:
    """Increment counter."""
    return {"count": 1, "total_cost": 0.05}

# If existing count=5 and total_cost=0.10:
# After update: count=6, total_cost=0.15
```

## Custom Reducers

### List Extension

Extend lists without duplicates:

**Python:**
```python
from typing import TypedDict, Annotated

def extend_list(left: list, right: list) -> list:
    """Extend list with new items."""
    return left + right

class ListState(TypedDict):
    """State with list extension."""
    results: Annotated[list[str], extend_list]
    errors: Annotated[list[str], extend_list]

# Usage
def worker_node(state: ListState) -> dict:
    """Add results."""
    return {"results": ["result1", "result2"]}

# If existing results=["a", "b"]:
# After update: results=["a", "b", "result1", "result2"]
```

**TypeScript:**
```typescript
import { StateSchema, ReducedValue } from "@langchain/langgraph";
import { z } from "zod/v4";

const ListState = new StateSchema({
  results: new ReducedValue(
    z.array(z.string()).default(() => []),
    { reducer: (current, update) => current.concat(update) }
  ),
  errors: new ReducedValue(
    z.array(z.string()).default(() => []),
    { reducer: (current, update) => current.concat(update) }
  ),
});
```

### List Extension (No Duplicates)

Extend lists while avoiding duplicates:

```python
def extend_unique(left: list, right: list) -> list:
    """Extend list without duplicates."""
    existing = set(left)
    return left + [item for item in right if item not in existing]

class UniqueListState(TypedDict):
    """State with unique list extension."""
    tags: Annotated[list[str], extend_unique]
    visited_urls: Annotated[list[str], extend_unique]
```

### Dictionary Merging

Merge dictionaries with various strategies:

**Python (shallow merge):**
```python
def merge_dicts(left: dict, right: dict) -> dict:
    """Merge dicts, right overrides left."""
    return {**left, **right}

class DictState(TypedDict):
    """State with dict merging."""
    context: Annotated[dict, merge_dicts]
    metadata: Annotated[dict, merge_dicts]

# Usage
def node(state: DictState) -> dict:
    return {"context": {"new_key": "value"}}

# If existing context={"old_key": "old"}:
# After update: context={"old_key": "old", "new_key": "value"}
```

**Python (deep merge):**
```python
from copy import deepcopy

def deep_merge_dicts(left: dict, right: dict) -> dict:
    """Deep merge dictionaries."""
    result = deepcopy(left)

    for key, value in right.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge_dicts(result[key], value)
        else:
            result[key] = value

    return result

class DeepDictState(TypedDict):
    """State with deep dict merging."""
    config: Annotated[dict, deep_merge_dicts]
```

**TypeScript:**
```typescript
import { StateSchema, ReducedValue } from "@langchain/langgraph";
import { z } from "zod/v4";

const DictState = new StateSchema({
  context: new ReducedValue(
    z.record(z.string(), z.any()).default(() => ({})),
    { reducer: (current, update) => ({ ...current, ...update }) }
  ),
  metadata: new ReducedValue(
    z.record(z.string(), z.any()).default(() => ({})),
    { reducer: (current, update) => ({ ...current, ...update }) }
  ),
});
```

### Set Operations

Union, intersection, or difference for sets:

```python
def union_sets(left: set, right: set) -> set:
    """Union of two sets."""
    return left | right

def intersection_sets(left: set, right: set) -> set:
    """Intersection of two sets."""
    return left & right

def difference_sets(left: set, right: set) -> set:
    """Difference (left - right)."""
    return left - right

class SetState(TypedDict):
    """State with set operations."""
    all_tags: Annotated[set[str], union_sets]
    common_features: Annotated[set[str], intersection_sets]
    removed_items: Annotated[set[str], difference_sets]
```

### Conditional Updates

Only update if new value meets criteria:

```python
def conditional_update(left: str, right: str | None) -> str:
    """Update only if right is not None."""
    return right if right is not None else left

def update_if_greater(left: int, right: int) -> int:
    """Update only if right is greater."""
    return max(left, right)

def update_if_not_empty(left: str, right: str) -> str:
    """Update only if right is non-empty."""
    return right if right.strip() else left

class ConditionalState(TypedDict):
    """State with conditional updates."""
    status: Annotated[str, conditional_update]
    max_score: Annotated[int, update_if_greater]
    current_value: Annotated[str, update_if_not_empty]
```

### Replacement (Default Behavior)

When no reducer is specified, new value replaces old:

```python
class ReplacementState(TypedDict):
    """State with replacement (no reducer)."""
    current_agent: str  # New value replaces old
    route: str  # New value replaces old
    status: str  # New value replaces old

# Usage
def node(state: ReplacementState) -> dict:
    return {"current_agent": "researcher"}

# Existing current_agent="writer" becomes current_agent="researcher"
```

## Advanced Reducer Patterns

### Accumulator with Validation

```python
def validated_accumulator(left: list[dict], right: list[dict]) -> list[dict]:
    """Accumulate items with validation."""
    result = left.copy()

    for item in right:
        # Validate item
        if "id" in item and "data" in item:
            # Check for duplicates
            if not any(existing["id"] == item["id"] for existing in result):
                result.append(item)

    return result

class ValidatedState(TypedDict):
    """State with validated accumulation."""
    items: Annotated[list[dict], validated_accumulator]
```

### Size-Limited Accumulator

```python
def limited_list(left: list, right: list, max_size: int = 100) -> list:
    """Accumulate with size limit."""
    combined = left + right
    if len(combined) > max_size:
        # Keep most recent items
        return combined[-max_size:]
    return combined

# Create reducer with closure
def make_limited_reducer(max_size: int):
    def reducer(left: list, right: list) -> list:
        combined = left + right
        return combined[-max_size:] if len(combined) > max_size else combined
    return reducer

class LimitedState(TypedDict):
    """State with size-limited list."""
    recent_messages: Annotated[list[str], make_limited_reducer(50)]
```

### Weighted Merge

```python
from typing import Any

def weighted_merge(left: dict[str, float], right: dict[str, float]) -> dict[str, float]:
    """Merge dictionaries with weighted averaging."""
    result = left.copy()

    for key, value in right.items():
        if key in result:
            # Average the values
            result[key] = (result[key] + value) / 2
        else:
            result[key] = value

    return result

class WeightedState(TypedDict):
    """State with weighted merging."""
    confidence_scores: Annotated[dict[str, float], weighted_merge]
```

### Custom Message Reducer

```python
from langchain_core.messages import BaseMessage

def custom_message_reducer(
    left: list[BaseMessage],
    right: list[BaseMessage],
    max_messages: int = 20
) -> list[BaseMessage]:
    """Custom message accumulation with limits."""
    # Combine messages
    combined = left + right

    # Apply size limit
    if len(combined) > max_messages:
        # Keep first message (system) and recent messages
        if combined[0].type == "system":
            return [combined[0]] + combined[-(max_messages-1):]
        else:
            return combined[-max_messages:]

    return combined

class CustomChatState(TypedDict):
    """Chat state with custom message handling."""
    messages: Annotated[list[BaseMessage], custom_message_reducer]
```

## Reducer Patterns by Use Case

### Multi-Agent Coordination

```python
from langgraph.graph import MessagesState

def agent_history_reducer(left: list[str], right: list[str]) -> list[str]:
    """Track agent execution order."""
    return left + right

def agent_results_reducer(left: dict, right: dict) -> dict:
    """Collect results from multiple agents."""
    return {**left, **right}

class MultiAgentState(MessagesState):
    """State for multi-agent systems."""
    agent_history: Annotated[list[str], agent_history_reducer]
    agent_results: Annotated[dict[str, Any], agent_results_reducer]
```

### Parallel Execution

```python
def parallel_results_reducer(left: list, right: list) -> list:
    """Accumulate results from parallel workers."""
    return left + right

class ParallelState(TypedDict):
    """State for parallel execution."""
    subtasks: list[dict]  # No reducer (replacement)
    results: Annotated[list[str], parallel_results_reducer]
    errors: Annotated[list[str], parallel_results_reducer]
```

### Error Handling

```python
def error_accumulator(left: list[dict], right: list[dict]) -> list[dict]:
    """Accumulate errors with deduplication."""
    existing_errors = {e["message"] for e in left}
    result = left.copy()

    for error in right:
        if error["message"] not in existing_errors:
            result.append(error)
            existing_errors.add(error["message"])

    return result

class ErrorHandlingState(TypedDict):
    """State with error accumulation."""
    errors: Annotated[list[dict], error_accumulator]
    retry_count: Annotated[int, operator.add]
```

### Metrics Collection

```python
def metrics_reducer(left: dict[str, list[float]], right: dict[str, list[float]]) -> dict[str, list[float]]:
    """Accumulate metrics over time."""
    result = {k: v.copy() for k, v in left.items()}

    for key, values in right.items():
        if key in result:
            result[key].extend(values)
        else:
            result[key] = values

    return result

class MetricsState(TypedDict):
    """State for metrics collection."""
    metrics: Annotated[dict[str, list[float]], metrics_reducer]
    total_calls: Annotated[int, operator.add]
    total_cost: Annotated[float, operator.add]
```

## TypeScript Reducer Patterns

### Custom Reducers

```typescript
import { StateSchema, ReducedValue } from "@langchain/langgraph";
import { z } from "zod/v4";

// List extension without duplicates
function extendUnique(current: string[], update: string[]): string[] {
  const existing = new Set(current);
  return [...current, ...update.filter(item => !existing.has(item))];
}

const UniqueListState = new StateSchema({
  tags: new ReducedValue(
    z.array(z.string()).default(() => []),
    { reducer: extendUnique }
  ),
});

// Deep dictionary merge
function deepMerge(current: any, update: any): any {
  const result = { ...current };

  for (const key in update) {
    if (typeof result[key] === 'object' && typeof update[key] === 'object') {
      result[key] = deepMerge(result[key], update[key]);
    } else {
      result[key] = update[key];
    }
  }

  return result;
}

const DeepMergeState = new StateSchema({
  config: new ReducedValue(
    z.record(z.string(), z.any()).default(() => ({})),
    { reducer: deepMerge }
  ),
});

// Conditional update
function conditionalUpdate<T>(current: T, update: T | null): T {
  return update !== null ? update : current;
}

const ConditionalState = new StateSchema({
  status: new ReducedValue(
    z.string(),
    { reducer: conditionalUpdate }
  ),
});
```

## Testing Reducers

### Unit Testing

```python
def test_extend_list_reducer():
    """Test list extension reducer."""
    def extend_list(left: list, right: list) -> list:
        return left + right

    existing = ["a", "b"]
    new = ["c", "d"]
    result = extend_list(existing, new)

    assert result == ["a", "b", "c", "d"]
    assert existing == ["a", "b"]  # Original unchanged

def test_merge_dicts_reducer():
    """Test dictionary merge reducer."""
    def merge_dicts(left: dict, right: dict) -> dict:
        return {**left, **right}

    existing = {"key1": "value1"}
    new = {"key2": "value2"}
    result = merge_dicts(existing, new)

    assert result == {"key1": "value1", "key2": "value2"}

def test_add_messages_reducer():
    """Test message accumulation."""
    from langgraph.graph.message import add_messages
    from langchain_core.messages import HumanMessage

    existing = [HumanMessage(content="Hello")]
    new = [HumanMessage(content="World")]
    result = add_messages(existing, new)

    assert len(result) == 2
    assert result[0].content == "Hello"
    assert result[1].content == "World"
```

### Integration Testing

```python
def test_reducer_in_graph():
    """Test reducer behavior in graph."""
    from langgraph.graph import StateGraph

    def extend_list(left: list, right: list) -> list:
        return left + right

    class TestState(TypedDict):
        items: Annotated[list[str], extend_list]

    def node1(state: TestState) -> dict:
        return {"items": ["a", "b"]}

    def node2(state: TestState) -> dict:
        return {"items": ["c", "d"]}

    graph = StateGraph(TestState)
    graph.add_node("node1", node1)
    graph.add_node("node2", node2)
    graph.add_edge("node1", "node2")
    graph.set_entry_point("node1")

    app = graph.compile()
    result = app.invoke({"items": []})

    assert result["items"] == ["a", "b", "c", "d"]
```

## Common Pitfalls

### Mutation vs Immutability

```python
# ❌ Bad: Mutates existing state
def bad_reducer(left: list, right: list) -> list:
    left.extend(right)  # Mutates left!
    return left

# ✅ Good: Creates new list
def good_reducer(left: list, right: list) -> list:
    return left + right  # Creates new list
```

### Type Mismatches

```python
# ❌ Bad: Type mismatch
def bad_reducer(left: list[str], right: str) -> list[str]:
    return left + [right]  # Expects list, gets str

# ✅ Good: Consistent types
def good_reducer(left: list[str], right: list[str]) -> list[str]:
    return left + right
```

### Missing Default Values

```python
# ❌ Bad: No default, might be None
class BadState(TypedDict):
    items: Annotated[list[str], extend_list]

# ✅ Good: With default in TypeScript
import { StateSchema, ReducedValue } from "@langchain/langgraph";
import { z } from "zod/v4";

const GoodState = new StateSchema({
  items: new ReducedValue(
    z.array(z.string()).default(() => []),  // Prevents undefined issues
    { reducer: (current, update) => current.concat(update) }
  ),
});
```

### Over-Complicated Reducers

```python
# ❌ Bad: Too complex, hard to test
def complex_reducer(left: dict, right: dict) -> dict:
    # 50 lines of complex logic
    # Multiple nested conditions
    # Side effects
    pass

# ✅ Good: Simple, focused
def simple_reducer(left: dict, right: dict) -> dict:
    return {**left, **right}
```

## Best Practices

1. **Keep reducers pure**: No side effects, same input = same output
2. **Avoid mutations**: Create new objects instead of modifying existing
3. **Match types**: Reducer input and output types should match
4. **Test independently**: Unit test reducers before using in graphs
5. **Document behavior**: Explain what the reducer does and why
6. **Consider performance**: Large lists/dicts can be slow to copy
7. **Use built-ins when possible**: `add_messages` and `operator.add` are optimized

## Performance Considerations

### Memory Usage

```python
# Consider memory when accumulating large data
def memory_efficient_reducer(left: list, right: list, max_size: int = 1000) -> list:
    """Prevent unbounded growth."""
    combined = left + right
    if len(combined) > max_size:
        return combined[-max_size:]  # Keep recent items
    return combined
```

### Computation Cost

```python
# Expensive reducer - consider caching
def expensive_reducer(left: dict, right: dict) -> dict:
    """Expensive merge operation."""
    # Deep copy and complex merging
    result = deep_copy_and_merge(left, right)
    return result

# Better: Shallow merge when possible
def cheap_reducer(left: dict, right: dict) -> dict:
    """Cheap shallow merge."""
    return {**left, **right}
```

## Additional Resources

- LangGraph Reducers Documentation: https://docs.langchain.com/oss/python/langgraph/use-graph-api
- Message Reducers: https://docs.langchain.com/oss/python/langgraph/graph-api
- TypeScript StateSchema API: https://docs.langchain.com/oss/javascript/langgraph/use-graph-api
