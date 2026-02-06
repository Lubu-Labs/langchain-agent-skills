# LLM-Based Error Recovery in LangGraph

LLM-recoverable errors are failures that the language model can understand and correct, such as tool failures, invalid arguments, or parsing issues. Instead of crashing or retrying blindly, you feed the error back to the LLM so it can adjust its approach.

## When to Use LLM-Based Recovery

Use this pattern when errors are caused by incorrect LLM decisions:

- **Tool failures**: Wrong parameters, invalid SQL queries, missing required fields
- **Parsing issues**: Malformed JSON, schema validation failures
- **Logic errors**: Business rule violations, constraint failures
- **Transient failures that need context**: Rate limits where the LLM should try a different approach

**Don't use for:**
- Network errors (use retry policies instead)
- Authentication failures (fix configuration)
- Programming errors (bubble up for debugging)
- Missing user information (use interrupt() instead)

## Error Type Classification

| Error Type | Who Fixes It | Strategy | When to Use |
|-----------|-------------|----------|-------------|
| Transient errors | System | Retry policy | Network issues, rate limits |
| LLM-recoverable | LLM | Store in state, loop back | Tool failures, parsing issues |
| User-fixable | Human | interrupt() | Missing information |
| Unexpected | Developer | Let bubble up | Unknown issues |

## 1. Command-Based Error Routing

The `Command` object allows you to both update state and control flow in a single return.

### Python

```python
from langgraph.types import Command
from typing import Literal

def execute_tool(state: State) -> Command[Literal["agent", "execute_tool"]]:
    try:
        result = run_tool(state['tool_call'])
        return Command(update={"tool_result": result}, goto="agent")
    except ToolError as e:
        # Let the LLM see what went wrong and try again
        return Command(
            update={"tool_result": f"Tool error: {str(e)}"},
            goto="agent"
        )
```

### TypeScript

```typescript
import { Command, GraphNode } from "@langchain/langgraph";

const executeTool: GraphNode<typeof State> = async (state, config) => {
  try {
    const result = await runTool(state.toolCall);
    return new Command({
      update: { toolResult: result },
      goto: "agent",
    });
  } catch (error) {
    // Let the LLM see what went wrong and try again
    return new Command({
      update: { toolResult: `Tool error: ${error}` },
      goto: "agent"
    });
  }
}
```

**Key points:**
- `update` modifies the state
- `goto` determines the next node
- The agent node sees the error in `tool_result` and can adjust its approach

## 2. ToolNode Error Handling

`ToolNode` provides built-in error handling for tool execution. Instead of crashing, it returns error messages to the LLM.

### Python

#### Catch All Errors

```python
from langgraph.prebuilt import ToolNode

# Returns error message to LLM for any exception
tool_node = ToolNode(tools, handle_tool_errors=True)
```

#### Custom Error Message

```python
# Fixed error message for all failures
tool_node = ToolNode(
    tools,
    handle_tool_errors="Something went wrong, please try again."
)
```

#### Custom Error Handler Function

```python
def handle_error(e: Exception) -> str:
    if isinstance(e, ValueError):
        return f"Invalid input: {e}"
    elif isinstance(e, KeyError):
        return f"Missing required field: {e}"
    return f"Unexpected error: {e}"

tool_node = ToolNode(tools, handle_tool_errors=handle_error)
```

#### Specific Exception Types

```python
# Only catch ValueError and TypeError, let others bubble up
tool_node = ToolNode(tools, handle_tool_errors=(ValueError, TypeError))
```

### TypeScript

#### Catch All Errors

```typescript
import { ToolNode } from "@langchain/langgraph/prebuilt";

// Returns error message to LLM for any exception
const toolNode = new ToolNode(tools, { handleToolErrors: true });
```

#### Custom Error Message

```typescript
// Fixed error message for all failures
const toolNode = new ToolNode(tools, {
  handleToolErrors: "Something went wrong, please try again."
});
```

**Note:** TypeScript ToolNode currently supports boolean or string values. For custom handler functions, use middleware (see section 4).

## 3. Integration with `create_agent` (LangChain v1+)

For prebuilt agents, use LangChain's `create_agent` / `createAgent` in v1.

- `create_react_agent` (Python) and `createReactAgent` (TypeScript) are deprecated in LangGraph v1
- In LangChain v1 agent APIs, tool error handling should be configured via middleware
- `ToolNode` remains useful for custom LangGraph workflows, but is not the primary error-handling entrypoint for `create_agent`

## 4. Middleware-Based Error Handling

Middleware provides more control over tool execution and error handling. This is part of LangChain's agent API (not LangGraph's graph-based API).

### Python

```python
from langchain.agents import create_agent
from langchain.agents.middleware import wrap_tool_call
from langchain.messages import ToolMessage

@wrap_tool_call
def handle_tool_errors(request, handler):
    """Handle tool execution errors with custom messages."""
    try:
        return handler(request)
    except Exception as e:
        # Return a custom error message to the model
        return ToolMessage(
            content=f"Tool error: Please check your input and try again. ({str(e)})",
            tool_call_id=request.tool_call["id"]
        )

agent = create_agent(
    model="claude-sonnet-4-5-20250929",
    tools=[check_weather, search_web],
    middleware=[handle_tool_errors]
)
```

### TypeScript

```typescript
import { createAgent, createMiddleware, ToolMessage } from "langchain";

const handleToolErrors = createMiddleware({
  name: "HandleToolErrors",
  wrapToolCall: async (request, handler) => {
    try {
      return await handler(request);
    } catch (error) {
      // Return a custom error message to the model
      return new ToolMessage({
        content: `Tool error: Please check your input and try again. (${error})`,
        tool_call_id: request.toolCall.id!,
      });
    }
  },
});

const agent = createAgent({
  model: "claude-sonnet-4-5-20250929",
  tools: [checkWeather, searchWeb],
  middleware: [handleToolErrors],
});
```

**When to use middleware:**
- Need to handle only specific tool errors
- Want to add logging or monitoring
- Require conditional error handling based on tool name or state
- Need to transform errors before returning to LLM

## 5. Error Recovery Loop Pattern

Complete example showing agent → tool → error → retry with max attempts tracking.

### Python

```python
from typing import Annotated, TypedDict, Literal
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command
from langchain_anthropic import ChatAnthropic

class State(TypedDict):
    messages: Annotated[list, "conversation history"]
    retry_count: int
    max_retries: int

def agent_node(state: State) -> Command[Literal["tools", END]]:
    """Agent decides what to do next."""
    model = ChatAnthropic(model="claude-sonnet-4-5-20250929")
    response = model.invoke(state["messages"])

    # Check if we should escalate
    if state["retry_count"] >= state["max_retries"]:
        return Command(
            update={
                "messages": state["messages"] + [
                    {
                        "role": "assistant",
                        "content": "I've encountered repeated errors. Please contact support."
                    }
                ]
            },
            goto=END
        )

    # Normal flow - if tool calls, route to tools
    if response.tool_calls:
        return Command(
            update={"messages": state["messages"] + [response]},
            goto="tools"
        )

    # No tool calls, we're done
    return Command(
        update={"messages": state["messages"] + [response]},
        goto=END
    )

def tools_node(state: State) -> Command[Literal["agent"]]:
    """Execute tools and handle errors."""
    last_message = state["messages"][-1]
    tool_calls = last_message.tool_calls

    responses = []
    error_occurred = False

    for tool_call in tool_calls:
        try:
            # Execute the tool
            result = execute_tool(tool_call)
            responses.append({
                "role": "tool",
                "content": str(result),
                "tool_call_id": tool_call["id"]
            })
        except Exception as e:
            # Store error message for LLM to see
            error_occurred = True
            responses.append({
                "role": "tool",
                "content": f"Error: {str(e)}. Please adjust your approach.",
                "tool_call_id": tool_call["id"]
            })

    # Increment retry count if error occurred
    new_retry_count = state["retry_count"] + 1 if error_occurred else state["retry_count"]

    return Command(
        update={
            "messages": state["messages"] + responses,
            "retry_count": new_retry_count
        },
        goto="agent"
    )

def execute_tool(tool_call):
    """Simulate tool execution."""
    # Your actual tool execution logic here
    pass

# Build the graph
workflow = StateGraph(State)

workflow.add_node("agent", agent_node)
workflow.add_node("tools", tools_node)

workflow.add_edge(START, "agent")

graph = workflow.compile()

# Run the graph
result = graph.invoke({
    "messages": [{"role": "user", "content": "Search for Python tutorials"}],
    "retry_count": 0,
    "max_retries": 3
})
```

### TypeScript

```typescript
import { StateGraph, START, END } from "@langchain/langgraph";
import { Command, GraphNode } from "@langchain/langgraph";
import { ChatAnthropic } from "@langchain/anthropic";

interface State {
  messages: Array<any>;
  retryCount: number;
  maxRetries: number;
}

const agentNode: GraphNode<State> = async (state) => {
  // Agent decides what to do next
  const model = new ChatAnthropic({ model: "claude-sonnet-4-5-20250929" });
  const response = await model.invoke(state.messages);

  // Check if we should escalate
  if (state.retryCount >= state.maxRetries) {
    return new Command({
      update: {
        messages: [
          ...state.messages,
          {
            role: "assistant",
            content: "I've encountered repeated errors. Please contact support."
          }
        ]
      },
      goto: END
    });
  }

  // Normal flow - if tool calls, route to tools
  if (response.tool_calls && response.tool_calls.length > 0) {
    return new Command({
      update: { messages: [...state.messages, response] },
      goto: "tools"
    });
  }

  // No tool calls, we're done
  return new Command({
    update: { messages: [...state.messages, response] },
    goto: END
  });
};

const toolsNode: GraphNode<State> = async (state) => {
  // Execute tools and handle errors
  const lastMessage = state.messages[state.messages.length - 1];
  const toolCalls = lastMessage.tool_calls || [];

  const responses = [];
  let errorOccurred = false;

  for (const toolCall of toolCalls) {
    try {
      // Execute the tool
      const result = await executeTool(toolCall);
      responses.push({
        role: "tool",
        content: String(result),
        tool_call_id: toolCall.id
      });
    } catch (error) {
      // Store error message for LLM to see
      errorOccurred = true;
      responses.push({
        role: "tool",
        content: `Error: ${error}. Please adjust your approach.`,
        tool_call_id: toolCall.id
      });
    }
  }

  // Increment retry count if error occurred
  const newRetryCount = errorOccurred ? state.retryCount + 1 : state.retryCount;

  return new Command({
    update: {
      messages: [...state.messages, ...responses],
      retryCount: newRetryCount
    },
    goto: "agent"
  });
};

async function executeTool(toolCall: any): Promise<any> {
  // Your actual tool execution logic here
  throw new Error("Not implemented");
}

// Build the graph
const workflow = new StateGraph<State>({
  channels: {
    messages: { value: null },
    retryCount: { value: 0 },
    maxRetries: { value: 3 }
  }
});

workflow.addNode("agent", agentNode);
workflow.addNode("tools", toolsNode);

workflow.addEdge(START, "agent");

const graph = workflow.compile();

// Run the graph
const result = await graph.invoke({
  messages: [{ role: "user", content: "Search for Python tutorials" }],
  retryCount: 0,
  maxRetries: 3
});
```

## Key Pattern Elements

1. **Error Capture**: Catch exceptions in tool execution
2. **Error Context**: Store error details in state
3. **Loop Back**: Return to agent node with error information
4. **Retry Tracking**: Increment counter when errors occur
5. **Escalation**: Exit after max retries to prevent infinite loops

## Best Practices

1. **Provide clear error messages**: The LLM needs context to adjust
2. **Track retry counts**: Prevent infinite loops
3. **Differentiate error types**: Only recover from errors the LLM can fix
4. **Set reasonable retry limits**: 2-3 retries is usually sufficient
5. **Log errors**: Keep diagnostics for debugging
6. **Combine with retry policies**: Use system retries for transient errors, LLM recovery for logic errors

## Combining Strategies

You can use multiple error handling approaches together:

```python
from langgraph.types import RetryPolicy

# System-level retry for transient errors
workflow.add_node(
    "tools",
    tools_node,
    retry_policy=RetryPolicy(max_attempts=3, initial_interval=1.0)
)

# LLM-level recovery for logic errors
def tools_node(state: State):
    try:
        result = execute_tool(state['tool_call'])
        return Command(update={"result": result}, goto="agent")
    except ValueError as e:  # LLM can fix this
        return Command(update={"result": f"Error: {e}"}, goto="agent")
    except AuthError:  # LLM can't fix this
        raise
```

This gives you:
- **Automatic retries** for network issues (RetryPolicy)
- **LLM recovery** for fixable errors (Command with error state)
- **Fast failure** for unrecoverable errors (raised exceptions)

## See Also

- [Retry Strategies](./retry-strategies.md) - System-level automatic retries
- [Human Escalation](./human-escalation.md) - User intervention for missing information
- [State Management Patterns](../../langgraph-agent-patterns/references/state-management-patterns.md) - Tracking errors in state
