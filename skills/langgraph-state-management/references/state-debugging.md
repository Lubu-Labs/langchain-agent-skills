# State Debugging Techniques

Debugging state management issues in LangGraph applications requires understanding how state flows through the graph, how reducers merge updates, and how to inspect state at each step.

## Common State Issues

### 1. State Not Updating

**Symptom:** Node returns update but state doesn't change.

**Common causes:**
- Node returns nothing or empty dict
- Reducer not merging correctly
- State field not in schema
- Type mismatch between update and schema

**Debug approach:**

```python
def debug_node(state: MyState) -> dict:
    """Node with debug logging."""
    print(f"Input state: {state}")

    # Create update
    update = {"count": state["count"] + 1}
    print(f"Returning update: {update}")

    return update

# Run and check output
result = app.invoke({"count": 0})
print(f"Final state: {result}")
```

### 2. State Overwritten

**Symptom:** Previous state values are lost.

**Common cause:** Using wrong reducer or no reducer.

```python
# ❌ Bad: No reducer, value gets replaced
class BadState(TypedDict):
    results: list[str]  # Each update replaces entire list

def node1(state: BadState) -> dict:
    return {"results": ["a", "b"]}

def node2(state: BadState) -> dict:
    return {"results": ["c", "d"]}  # Replaces ["a", "b"]!

# ✅ Good: Use reducer for accumulation
def extend_list(left: list, right: list) -> list:
    return left + right

class GoodState(TypedDict):
    results: Annotated[list[str], extend_list]

def node1(state: GoodState) -> dict:
    return {"results": ["a", "b"]}

def node2(state: GoodState) -> dict:
    return {"results": ["c", "d"]}  # Appends, final: ["a", "b", "c", "d"]
```

### 3. Reducer Not Applied

**Symptom:** Custom reducer doesn't seem to run.

**Common causes:**
- Reducer function has wrong signature
- Annotated type not correctly specified
- Reducer raises exception silently

```python
from typing import Annotated

# ❌ Bad: Wrong signature
def bad_reducer(x):  # Missing second parameter
    return x

# ✅ Good: Correct signature
def good_reducer(left: list, right: list) -> list:
    return left + right

class State(TypedDict):
    items: Annotated[list[str], good_reducer]

# Debug reducer directly
def test_reducer():
    """Test reducer in isolation."""
    left = ["a", "b"]
    right = ["c", "d"]
    result = good_reducer(left, right)
    assert result == ["a", "b", "c", "d"]
```

### 4. Messages Not Accumulating

**Symptom:** Only latest message appears in state.

**Common cause:** Not using `MessagesState` or the `add_messages` reducer.

```python
from typing import TypedDict
from langgraph.graph import MessagesState
from langchain_core.messages import BaseMessage

# ❌ Bad: Messages get replaced
class BadState(TypedDict):
    messages: list[BaseMessage]

# ✅ Good: Messages accumulate
class GoodState(MessagesState):
    pass
# or
class GoodState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
```

### 5. Unexpected State Values

**Symptom:** State contains unexpected or stale values.

**Debug with state inspection:**

```python
def inspect_state(state: MyState) -> dict:
    """Inspect all state fields."""
    print("=== State Inspection ===")
    for key, value in state.items():
        print(f"{key}: {value} (type: {type(value).__name__})")
    print("========================")
    return {}

# Add inspection node to graph
graph.add_node("inspect", inspect_state)
graph.add_edge("some_node", "inspect")
```

## Debugging Tools

### 1. Print Debugging

```python
def debug_node(state: MyState) -> dict:
    """Node with detailed logging."""
    print(f"\n{'='*50}")
    print(f"Node: {debug_node.__name__}")
    print(f"Input state keys: {list(state.keys())}")
    print(f"Messages count: {len(state.get('messages', []))}")
    print(f"Current iteration: {state.get('iteration', 0)}")
    print(f"{'='*50}\n")

    # Your logic here
    result = {"iteration": state.get("iteration", 0) + 1}

    print(f"Returning: {result}")
    return result
```

### 2. LangSmith Tracing

Enable tracing to see state flow in LangSmith:

```python
import os

# Enable LangSmith tracing
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_API_KEY"] = "your-api-key"
os.environ["LANGSMITH_PROJECT"] = "state-debugging"

# Run graph - traces appear in LangSmith
result = app.invoke({"messages": [...]})
```

**What to look for in traces:**
- State at each node entry
- State updates from each node
- Reducer operations
- Execution order

### 3. Checkpoint Inspection

Examine saved checkpoints to understand state evolution:

```python
from langgraph.checkpoint.memory import InMemorySaver

checkpointer = InMemorySaver()
app = graph.compile(checkpointer=checkpointer)

# Run graph
config = {"configurable": {"thread_id": "debug-1"}}
result = app.invoke({"count": 0}, config)

# Inspect checkpoints
checkpoints = list(checkpointer.list(config))
print(f"Found {len(checkpoints)} checkpoints")

for i, checkpoint in enumerate(checkpoints):
    print(f"\nCheckpoint {i}:")
    print(f"  ID: {checkpoint['id']}")
    print(f"  State: {checkpoint['checkpoint']['channel_values']}")
    print(f"  Metadata: {checkpoint['metadata']}")
```

### 4. State Snapshots

Create snapshot utility for debugging:

```python
from typing import TypedDict
import json
from datetime import datetime

class StateSnapshot:
    """Capture state snapshots for debugging."""

    def __init__(self):
        self.snapshots = []

    def capture(self, label: str, state: dict):
        """Capture state snapshot."""
        snapshot = {
            "timestamp": datetime.now().isoformat(),
            "label": label,
            "state": state.copy()
        }
        self.snapshots.append(snapshot)

    def export(self, filename: str):
        """Export snapshots to file."""
        with open(filename, 'w') as f:
            json.dump(self.snapshots, f, indent=2, default=str)

    def compare(self, idx1: int, idx2: int):
        """Compare two snapshots."""
        s1 = self.snapshots[idx1]
        s2 = self.snapshots[idx2]

        print(f"\nComparing '{s1['label']}' vs '{s2['label']}':")

        all_keys = set(s1["state"].keys()) | set(s2["state"].keys())

        for key in sorted(all_keys):
            v1 = s1["state"].get(key, "MISSING")
            v2 = s2["state"].get(key, "MISSING")

            if v1 != v2:
                print(f"  {key}: {v1} → {v2}")

# Usage
snapshots = StateSnapshot()

def node1(state: MyState) -> dict:
    snapshots.capture("node1_input", state)
    result = {"count": state["count"] + 1}
    snapshots.capture("node1_output", {**state, **result})
    return result

# After execution
snapshots.compare(0, 1)
snapshots.export("state_snapshots.json")
```

### 5. State Validators

Create validators to catch issues early:

```python
from typing import TypedDict
from pydantic import BaseModel, ValidationError

class StateValidator(BaseModel):
    """Validate state schema."""
    messages: list
    iteration: int
    max_iterations: int

    class Config:
        extra = "allow"  # Allow extra fields

def validate_state(state: dict) -> None:
    """Validate state structure."""
    try:
        StateValidator(**state)
    except ValidationError as e:
        print(f"❌ State validation failed:")
        for error in e.errors():
            field = ".".join(str(x) for x in error["loc"])
            print(f"  {field}: {error['msg']}")
        raise

def safe_node(state: MyState) -> dict:
    """Node with state validation."""
    validate_state(state)

    # Process with validated state
    return {"iteration": state["iteration"] + 1}
```

## Debugging Reducers

### Test Reducers in Isolation

```python
def test_custom_reducer():
    """Test reducer behavior."""
    def my_reducer(left: list, right: list) -> list:
        return left + right

    # Test basic case
    result = my_reducer(["a"], ["b"])
    assert result == ["a", "b"], f"Expected ['a', 'b'], got {result}"

    # Test empty lists
    result = my_reducer([], ["a"])
    assert result == ["a"]

    result = my_reducer(["a"], [])
    assert result == ["a"]

    # Test identity
    result = my_reducer([], [])
    assert result == []

    print("✅ All reducer tests passed")

test_custom_reducer()
```

### Debug Reducer Application

```python
from typing import Annotated

def debug_reducer(left: list, right: list) -> list:
    """Reducer with debug logging."""
    print(f"Reducer called:")
    print(f"  Left: {left}")
    print(f"  Right: {right}")
    result = left + right
    print(f"  Result: {result}")
    return result

class DebugState(TypedDict):
    items: Annotated[list[str], debug_reducer]

# Run and observe reducer calls
def node(state: DebugState) -> dict:
    return {"items": ["new_item"]}

graph = StateGraph(DebugState)
graph.add_node("node", node)
graph.set_entry_point("node")
app = graph.compile()

result = app.invoke({"items": ["existing"]})
# Output shows reducer being called
```

### Common Reducer Issues

```python
# Issue: Reducer mutates state
def bad_reducer(left: list, right: list) -> list:
    left.extend(right)  # ❌ Mutates left
    return left

# Fix: Create new list
def good_reducer(left: list, right: list) -> list:
    return left + right  # ✅ Creates new list

# Issue: Reducer has side effects
def bad_reducer_with_side_effects(left: list, right: list) -> list:
    # ❌ Side effect: writing to file
    with open("log.txt", "a") as f:
        f.write(f"{left} + {right}\n")
    return left + right

# Fix: Keep reducer pure
def good_reducer_pure(left: list, right: list) -> list:
    return left + right  # ✅ Pure function

# Issue: Type mismatch
def bad_typed_reducer(left: list[str], right: str) -> list[str]:
    return left + [right]  # ❌ Expects list, gets str

# Fix: Correct types
def good_typed_reducer(left: list[str], right: list[str]) -> list[str]:
    return left + right  # ✅ Correct types
```

## Debugging Message State

### Inspect Messages

```python
from langchain_core.messages import BaseMessage

def inspect_messages(state: dict) -> dict:
    """Detailed message inspection."""
    messages = state.get("messages", [])

    print(f"\n=== Messages ({len(messages)} total) ===")

    for i, msg in enumerate(messages):
        print(f"\n{i}. {msg.__class__.__name__}")
        print(f"   Content: {msg.content[:100]}...")
        print(f"   ID: {getattr(msg, 'id', 'N/A')}")

        if hasattr(msg, 'name'):
            print(f"   Name: {msg.name}")

        if hasattr(msg, 'tool_calls'):
            print(f"   Tool calls: {len(msg.tool_calls)}")

    print("=" * 40)
    return {}
```

### Debug Message Updates

```python
from langchain_core.messages import HumanMessage, AIMessage

# Test add_messages reducer behavior
from langgraph.graph.message import add_messages

def test_message_accumulation():
    """Test message accumulation."""
    existing = [
        HumanMessage(content="Hello", id="msg1"),
        AIMessage(content="Hi", id="msg2")
    ]

    new = [AIMessage(content="How are you?", id="msg3")]

    result = add_messages(existing, new)

    assert len(result) == 3
    assert result[0].id == "msg1"
    assert result[1].id == "msg2"
    assert result[2].id == "msg3"

def test_message_update():
    """Test updating existing message."""
    existing = [
        HumanMessage(content="Original", id="msg1")
    ]

    # Update message with same ID
    updated = [
        HumanMessage(content="Updated", id="msg1")
    ]

    result = add_messages(existing, updated)

    assert len(result) == 1
    assert result[0].content == "Updated"
```

## Debugging Graph Flow

### Visualize Graph

```python
from IPython.display import Image, display

# Generate graph visualization
try:
    display(Image(app.get_graph().draw_mermaid_png()))
except Exception:
    print(app.get_graph().draw_mermaid())
```

### Trace Execution Order

```python
from typing import TypedDict

class TracingState(TypedDict):
    """State with execution tracing."""
    value: int
    trace: list[str]

def make_traced_node(name: str):
    """Create node with execution tracing."""
    def node(state: TracingState) -> dict:
        trace = state.get("trace", [])
        trace.append(name)
        print(f"Executing: {name}")
        return {"trace": trace, "value": state["value"] + 1}
    return node

# Build graph with tracing
graph = StateGraph(TracingState)
graph.add_node("node1", make_traced_node("node1"))
graph.add_node("node2", make_traced_node("node2"))
graph.add_node("node3", make_traced_node("node3"))
graph.add_edge("node1", "node2")
graph.add_edge("node2", "node3")
graph.set_entry_point("node1")

app = graph.compile()
result = app.invoke({"value": 0, "trace": []})

print(f"Execution trace: {result['trace']}")
# Output: Execution trace: ['node1', 'node2', 'node3']
```

### Debug Conditional Edges

```python
def debug_router(state: MyState) -> str:
    """Router with debug logging."""
    next_node = decide_next_node(state)

    print(f"\n=== Router Decision ===")
    print(f"Current state: {state}")
    print(f"Routing to: {next_node}")
    print(f"=" * 40)

    return next_node

graph.add_conditional_edges(
    "router_node",
    debug_router,
    {
        "option1": "node1",
        "option2": "node2",
        "finish": END
    }
)
```

## Debugging Persistence

### Inspect Checkpoints

```python
def inspect_all_checkpoints(checkpointer, thread_id: str):
    """Inspect all checkpoints for debugging."""
    config = {"configurable": {"thread_id": thread_id}}
    checkpoints = list(checkpointer.list(config))

    print(f"\n=== Checkpoints for {thread_id} ({len(checkpoints)}) ===\n")

    for i, cp in enumerate(checkpoints):
        print(f"Checkpoint {i}:")
        print(f"  ID: {cp['id']}")
        print(f"  Parent: {cp.get('parent_id', 'N/A')}")

        # State snapshot
        state = cp['checkpoint']['channel_values']
        print(f"  State keys: {list(state.keys())}")

        # Metadata
        metadata = cp.get('metadata', {})
        print(f"  Step: {metadata.get('step', 'N/A')}")
        print(f"  Writes: {metadata.get('writes', 'N/A')}")
        print()
```

### Compare Checkpoints

```python
def compare_checkpoints(checkpointer, thread_id: str, idx1: int, idx2: int):
    """Compare two checkpoints."""
    config = {"configurable": {"thread_id": thread_id}}
    checkpoints = list(checkpointer.list(config))

    cp1 = checkpoints[idx1]['checkpoint']['channel_values']
    cp2 = checkpoints[idx2]['checkpoint']['channel_values']

    print(f"\nComparing checkpoint {idx1} vs {idx2}:")

    all_keys = set(cp1.keys()) | set(cp2.keys())

    for key in sorted(all_keys):
        v1 = cp1.get(key)
        v2 = cp2.get(key)

        if v1 != v2:
            print(f"  {key}:")
            print(f"    Before: {v1}")
            print(f"    After: {v2}")
```

### Debug Checkpoint Loading

```python
def debug_checkpoint_loading(app, thread_id: str):
    """Debug checkpoint loading behavior."""
    config = {"configurable": {"thread_id": thread_id}}

    # Get current state
    state = app.get_state(config)
    print(f"Current state: {state}")

    # Get state history
    history = app.get_state_history(config)
    print(f"\nState history:")
    for i, state in enumerate(history):
        print(f"  {i}. {state.values}")
```

## Performance Debugging

### Profile State Operations

```python
import time

def profile_node(state: MyState) -> dict:
    """Profile node execution."""
    start = time.time()

    # Your logic here
    result = expensive_operation(state)

    duration = time.time() - start
    print(f"Node took {duration:.3f}s")

    return result
```

### Measure State Size

```python
import sys
import json

def measure_state_size(state: dict):
    """Measure state size in memory."""
    # Approximate size
    size_bytes = sys.getsizeof(json.dumps(state, default=str))
    size_kb = size_bytes / 1024

    print(f"State size: {size_kb:.2f} KB")

    # Size by field
    for key, value in state.items():
        field_size = sys.getsizeof(json.dumps(value, default=str))
        print(f"  {key}: {field_size / 1024:.2f} KB")
```

## Common Debugging Patterns

### 1. Binary Search for Issues

```python
# Disable nodes one by one to find problematic node
def create_minimal_graph():
    """Minimal graph for debugging."""
    graph = StateGraph(MyState)

    # Add nodes one by one
    graph.add_node("node1", node1)
    # graph.add_node("node2", node2)  # Comment out
    # graph.add_node("node3", node3)  # Comment out

    # ... build graph

    return graph.compile()

# Gradually uncomment nodes to find issue
```

### 2. Assertion Checks

```python
def node_with_assertions(state: MyState) -> dict:
    """Node with defensive assertions."""
    # Check preconditions
    assert "messages" in state, "Missing messages field"
    assert len(state["messages"]) > 0, "Messages list is empty"
    assert state["iteration"] < 100, "Too many iterations"

    # Your logic here
    result = process(state)

    # Check postconditions
    assert "result" in result, "Missing result field"
    assert result["result"] is not None, "Result is None"

    return result
```

### 3. Diff State Changes

```python
def diff_state(before: dict, after: dict):
    """Show what changed in state."""
    print("\n=== State Changes ===")

    all_keys = set(before.keys()) | set(after.keys())

    for key in sorted(all_keys):
        before_val = before.get(key, "MISSING")
        after_val = after.get(key, "MISSING")

        if before_val != after_val:
            print(f"{key}:")
            print(f"  Before: {before_val}")
            print(f"  After: {after_val}")

# Usage
def node_with_diff(state: MyState) -> dict:
    before = state.copy()
    result = {"count": state["count"] + 1}
    after = {**state, **result}
    diff_state(before, after)
    return result
```

## Debugging Checklist

When debugging state issues:

1. **Verify schema definition**
   - [ ] All fields are defined
   - [ ] Types are correct
   - [ ] Reducers are specified for accumulating fields

2. **Check node returns**
   - [ ] Node returns a dict
   - [ ] Returned keys match state schema
   - [ ] Values have correct types

3. **Test reducers**
   - [ ] Reducer has correct signature
   - [ ] Reducer doesn't mutate state
   - [ ] Reducer handles edge cases

4. **Inspect execution**
   - [ ] Enable LangSmith tracing
   - [ ] Add print statements
   - [ ] Check execution order

5. **Validate state**
   - [ ] Use Pydantic for validation
   - [ ] Add assertions
   - [ ] Check for None values

6. **Review persistence**
   - [ ] Checkpoints are being saved
   - [ ] Thread IDs are correct
   - [ ] State is loading correctly

## Additional Resources

- LangSmith Debugging Guide: https://docs.langchain.com/langsmith/home
- LangGraph State Documentation: https://docs.langchain.com/oss/python/langgraph/graph-api
- LangGraph How-To Guides: https://docs.langchain.com/oss/python/langgraph/add-memory
- Python Debugging Documentation: https://docs.python.org/3/library/pdb.html
