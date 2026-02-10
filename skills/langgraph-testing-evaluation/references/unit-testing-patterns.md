# Unit Testing Patterns for LangGraph Agents

This reference provides comprehensive patterns for writing unit tests for LangGraph agents in both Python and JavaScript/TypeScript.

## Table of Contents

- [Testing Individual Nodes](#testing-individual-nodes)
- [Testing State Transitions](#testing-state-transitions)
- [Mocking LLM Responses](#mocking-llm-responses)
- [Testing Tool Calls](#testing-tool-calls)
- [Testing Conditional Edges](#testing-conditional-edges)
- [Pytest Patterns](#pytest-patterns)
- [Vitest/Jest Patterns](#vitestjest-patterns)

## Testing Individual Nodes

### Python Example (Pytest)

```python
import pytest
from your_agent import agent_node, AgentState

def test_agent_node_basic():
    """Test agent node with simple input."""
    state = AgentState(
        messages=[{"role": "user", "content": "Hello"}]
    )

    result = agent_node(state)

    assert "messages" in result
    assert len(result["messages"]) > 0
    assert result["messages"][-1]["role"] == "assistant"

def test_agent_node_with_tools():
    """Test agent node that should call tools."""
    state = AgentState(
        messages=[{"role": "user", "content": "What's the weather?"}]
    )

    result = agent_node(state)

    # Verify tool call was made
    assert any(
        "tool_calls" in msg
        for msg in result["messages"]
    )
```

### TypeScript Example (Vitest)

```typescript
import { describe, it, expect } from "vitest";
import { agentNode, type AgentState } from "./your-agent";

describe("Agent Node", () => {
  it("should process basic input", () => {
    const state: AgentState = {
      messages: [{ role: "user", content: "Hello" }],
    };

    const result = agentNode(state);

    expect(result.messages).toBeDefined();
    expect(result.messages.length).toBeGreaterThan(0);
    expect(result.messages[result.messages.length - 1].role).toBe("assistant");
  });

  it("should call tools when needed", () => {
    const state: AgentState = {
      messages: [{ role: "user", content: "What's the weather?" }],
    };

    const result = agentNode(state);

    const hasToolCall = result.messages.some((msg) => "tool_calls" in msg);
    expect(hasToolCall).toBe(true);
  });
});
```

## Testing State Transitions

### Python Example

```python
def test_state_accumulation():
    """Test that state accumulates correctly across nodes."""
    initial_state = AgentState(
        messages=[],
        metadata={"step": 0}
    )

    # First node
    state1 = node_a(initial_state)
    assert state1["metadata"]["step"] == 1

    # Second node
    state2 = node_b(state1)
    assert state2["metadata"]["step"] == 2
    assert len(state2["messages"]) > 0

def test_reducer_behavior():
    """Test custom reducer functions."""
    from operator import add

    state = {"items": [1, 2, 3]}
    new_items = [4, 5]

    # Test list concatenation reducer
    result = add(state["items"], new_items)
    assert result == [1, 2, 3, 4, 5]
```

### TypeScript Example

```typescript
describe("State Transitions", () => {
  it("should accumulate state correctly", () => {
    const initialState: AgentState = {
      messages: [],
      metadata: { step: 0 },
    };

    // First node
    const state1 = nodeA(initialState);
    expect(state1.metadata.step).toBe(1);

    // Second node
    const state2 = nodeB(state1);
    expect(state2.metadata.step).toBe(2);
    expect(state2.messages.length).toBeGreaterThan(0);
  });
});
```

## Mocking LLM Responses

### Python Example with Mock

```python
from unittest.mock import Mock, patch

def test_with_mocked_llm():
    """Test agent with mocked LLM responses."""
    mock_llm = Mock()
    mock_llm.invoke.return_value = {
        "role": "assistant",
        "content": "Mocked response"
    }

    # Inject mock into agent
    with patch('your_agent.llm', mock_llm):
        state = AgentState(messages=[{"role": "user", "content": "Hi"}])
        result = agent_node(state)

        assert mock_llm.invoke.called
        assert result["messages"][-1]["content"] == "Mocked response"

def test_with_tool_calls():
    """Test mocked LLM that returns tool calls."""
    mock_llm = Mock()
    mock_llm.invoke.return_value = {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "name": "search",
                "args": {"query": "weather"},
                "id": "call_123"
            }
        ]
    }

    with patch('your_agent.llm', mock_llm):
        state = AgentState(messages=[{"role": "user", "content": "Weather?"}])
        result = agent_node(state)

        assert len(result["messages"][-1]["tool_calls"]) == 1
        assert result["messages"][-1]["tool_calls"][0]["name"] == "search"
```

### TypeScript Example with Vi

```typescript
import { vi } from "vitest";

describe("Mocked LLM Tests", () => {
  it("should work with mocked LLM responses", async () => {
    const mockLLM = {
      invoke: vi.fn().mockResolvedValue({
        role: "assistant",
        content: "Mocked response",
      }),
    };

    const state: AgentState = {
      messages: [{ role: "user", content: "Hi" }],
    };

    const result = await agentNode(state, mockLLM);

    expect(mockLLM.invoke).toHaveBeenCalled();
    expect(result.messages[result.messages.length - 1].content).toBe(
      "Mocked response"
    );
  });

  it("should handle tool calls", async () => {
    const mockLLM = {
      invoke: vi.fn().mockResolvedValue({
        role: "assistant",
        content: "",
        tool_calls: [
          {
            name: "search",
            args: { query: "weather" },
            id: "call_123",
          },
        ],
      }),
    };

    const state: AgentState = {
      messages: [{ role: "user", content: "Weather?" }],
    };

    const result = await agentNode(state, mockLLM);

    expect(result.messages[result.messages.length - 1].tool_calls).toHaveLength(1);
    expect(result.messages[result.messages.length - 1].tool_calls[0].name).toBe(
      "search"
    );
  });
});
```

## Testing Tool Calls

### Python Example

```python
def test_tool_execution():
    """Test that tools execute correctly."""
    from your_agent import search_tool

    result = search_tool("weather in San Francisco")

    assert result is not None
    assert "temperature" in result or "forecast" in result

def test_tool_error_handling():
    """Test tool error handling."""
    from your_agent import search_tool

    # Test with invalid input
    result = search_tool("")

    # Should return error message, not raise exception
    assert "error" in result.lower() or result is None
```

### TypeScript Example

```typescript
describe("Tool Execution", () => {
  it("should execute tool correctly", async () => {
    const result = await searchTool("weather in San Francisco");

    expect(result).toBeDefined();
    expect(
      result.includes("temperature") || result.includes("forecast")
    ).toBe(true);
  });

  it("should handle tool errors gracefully", async () => {
    const result = await searchTool("");

    // Should return error message, not throw
    expect(result.toLowerCase()).toContain("error");
  });
});
```

## Testing Conditional Edges

### Python Example

```python
def test_routing_logic():
    """Test conditional edge routing."""
    from your_agent import should_continue, AgentState

    # State with tool calls should continue
    state_with_tools = AgentState(
        messages=[
            {
                "role": "assistant",
                "tool_calls": [{"name": "search"}]
            }
        ]
    )

    assert should_continue(state_with_tools) == "continue"

    # State without tool calls should end
    state_without_tools = AgentState(
        messages=[
            {
                "role": "assistant",
                "content": "Final answer"
            }
        ]
    )

    assert should_continue(state_without_tools) == "end"

def test_router_selects_correct_node():
    """Test router function."""
    from your_agent import route_to_specialist

    math_state = AgentState(
        messages=[{"role": "user", "content": "What is 2+2?"}]
    )
    assert route_to_specialist(math_state) == "math_agent"

    code_state = AgentState(
        messages=[{"role": "user", "content": "Write a function"}]
    )
    assert route_to_specialist(code_state) == "code_agent"
```

### TypeScript Example

```typescript
describe("Conditional Edges", () => {
  it("should route correctly based on state", () => {
    const stateWithTools: AgentState = {
      messages: [
        {
          role: "assistant",
          tool_calls: [{ name: "search" }],
        },
      ],
    };

    expect(shouldContinue(stateWithTools)).toBe("continue");

    const stateWithoutTools: AgentState = {
      messages: [
        {
          role: "assistant",
          content: "Final answer",
        },
      ],
    };

    expect(shouldContinue(stateWithoutTools)).toBe("end");
  });

  it("should route to correct specialist", () => {
    const mathState: AgentState = {
      messages: [{ role: "user", content: "What is 2+2?" }],
    };
    expect(routeToSpecialist(mathState)).toBe("math_agent");

    const codeState: AgentState = {
      messages: [{ role: "user", content: "Write a function" }],
    };
    expect(routeToSpecialist(codeState)).toBe("code_agent");
  });
});
```

## Pytest Patterns

### Fixtures

```python
import pytest
from langgraph.checkpoint.memory import MemorySaver

@pytest.fixture
def memory_saver():
    """Provide a fresh memory saver for each test."""
    return MemorySaver()

@pytest.fixture
def compiled_graph(memory_saver):
    """Provide a compiled graph with memory."""
    from your_agent import build_graph

    graph = build_graph()
    return graph.compile(checkpointer=memory_saver)

@pytest.fixture
def mock_llm():
    """Provide a mocked LLM."""
    from unittest.mock import Mock

    mock = Mock()
    mock.invoke.return_value = {
        "role": "assistant",
        "content": "Test response"
    }
    return mock
```

### Parametrized Tests

```python
@pytest.mark.parametrize("input_text,expected_node", [
    ("What's 2+2?", "math_agent"),
    ("Write code", "code_agent"),
    ("General question", "general_agent"),
])
def test_routing_parametrized(input_text, expected_node):
    """Test routing with multiple inputs."""
    state = AgentState(
        messages=[{"role": "user", "content": input_text}]
    )

    result = route_to_specialist(state)
    assert result == expected_node
```

### Test Caching

```python
# pytest.ini or pyproject.toml
# [tool.pytest.ini_options]
# cache_dir = "tests/.pytest_cache"

# Use with LangSmith test cache
import os
os.environ["LANGSMITH_TEST_CACHE"] = "tests/cassettes"

@pytest.mark.llm  # Mark tests that use real LLMs
def test_with_real_llm(compiled_graph):
    """Test with real LLM (cached)."""
    state = {"messages": [{"role": "user", "content": "Hello"}]}

    result = compiled_graph.invoke(state)
    assert len(result["messages"]) > 1
```

## Vitest/Jest Patterns

### Setup Files

```typescript
// vitest.config.ts
import { defineConfig } from "vitest/config";

export default defineConfig({
  test: {
    globals: true,
    environment: "node",
    setupFiles: ["./tests/setup.ts"],
  },
});

// tests/setup.ts
import { beforeEach, vi } from "vitest";

beforeEach(() => {
  // Reset mocks before each test
  vi.clearAllMocks();
});
```

### Parameterized Tests

```typescript
describe.each([
  ["What's 2+2?", "math_agent"],
  ["Write code", "code_agent"],
  ["General question", "general_agent"],
])("Routing with input: %s", (inputText, expectedNode) => {
  it(`should route to ${expectedNode}`, () => {
    const state: AgentState = {
      messages: [{ role: "user", content: inputText }],
    };

    const result = routeToSpecialist(state);
    expect(result).toBe(expectedNode);
  });
});
```

### Snapshot Testing

```typescript
it("should match snapshot", () => {
  const state: AgentState = {
    messages: [{ role: "user", "content": "Test input" }],
  };

  const result = processState(state);

  expect(result).toMatchSnapshot();
});
```

## Best Practices

1. **Test isolation**: Each test should be independent
2. **Mock external dependencies**: Always mock LLM calls in unit tests
3. **Test edge cases**: Empty inputs, invalid data, error conditions
4. **Use descriptive test names**: Make failures easy to understand
5. **Keep tests fast**: Unit tests should run in milliseconds
6. **Test one thing**: Each test should verify a single behavior
7. **Use fixtures/setup**: Reduce duplication across tests
8. **Test both success and failure paths**: Don't just test happy paths
