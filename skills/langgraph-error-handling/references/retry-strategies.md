# Retry Strategies in LangGraph

This guide covers retry strategies for handling transient failures in LangGraph nodes, including API configuration, backoff strategies, and common patterns.

## Overview

Retry policies in LangGraph automatically handle transient failures such as network issues, rate limits, and temporary service unavailability. They apply exponential backoff with configurable parameters to nodes that interact with external systems.

## RetryPolicy API Reference

### Python

```python
from langgraph.types import RetryPolicy

RetryPolicy(
    max_attempts=3,           # Maximum retry attempts (default: 3)
    initial_interval=1.0,     # Initial delay in seconds (default: 1.0)
    backoff_factor=2.0,       # Exponential backoff multiplier (default: 2.0)
    retry_on=None            # Exception type, callable, or None for default
)
```

**Parameters:**

- `max_attempts` (int): Total number of attempts including the initial call. Default is 3.
- `initial_interval` (float): Initial delay in seconds before the first retry. Default is 1.0.
- `backoff_factor` (float): Multiplier for exponential backoff. Each retry waits `initial_interval * (backoff_factor ** retry_number)` seconds. Default is 2.0.
- `retry_on` (Exception type, callable, or None): Specifies which exceptions to retry. Default uses `default_retry_on` function.

**Default Retry Behavior (Python):**

By default, the retry policy retries on **most exceptions EXCEPT**:
- `ValueError`
- `TypeError`
- `ArithmeticError`
- `ImportError`
- `LookupError`
- `NameError`
- `SyntaxError`
- `RuntimeError`
- `ReferenceError`
- `StopIteration`
- `StopAsyncIteration`
- `OSError`

For HTTP libraries (`requests`, `httpx`), it **only retries on 5xx status codes**.

### TypeScript

```typescript
import { RetryPolicy } from "@langchain/langgraph";

interface RetryPolicy {
  maxAttempts?: number;        // Maximum retry attempts (default: 3)
  initialInterval?: number;    // Initial delay in seconds (default: 1.0)
  backoffFactor?: number;      // Exponential backoff multiplier (default: 2.0)
  retryOn?: (error: Error) => boolean;  // Function to determine if error should be retried
}
```

**Default Retry Behavior (TypeScript):**

By default, the retry policy retries on **all exceptions EXCEPT**:
- `TypeError`
- `SyntaxError`
- `ReferenceError`

## Applying RetryPolicy to Nodes

### Python: Graph API

```python
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.types import RetryPolicy
from langchain.chat_models import init_chat_model

model = init_chat_model("claude-3-5-sonnet-20240620")

def call_model(state: MessagesState):
    response = model.invoke(state["messages"])
    return {"messages": [response]}

builder = StateGraph(MessagesState)
builder.add_node(
    "model",
    call_model,
    retry_policy=RetryPolicy(max_attempts=5, initial_interval=1.0)
)
builder.add_edge(START, "model")
builder.add_edge("model", END)
graph = builder.compile()
```

### TypeScript: Graph API

```typescript
import { StateGraph, StateSchema, MessagesValue, GraphNode } from "@langchain/langgraph";
import { ChatAnthropic } from "@langchain/anthropic";

const State = new StateSchema({
  messages: MessagesValue,
});

const model = new ChatAnthropic({ model: "claude-3-5-sonnet-20240620" });

const callModel: GraphNode<typeof State> = async (state) => {
  const response = await model.invoke(state.messages);
  return { messages: [response] };
};

const graph = new StateGraph(State)
  .addNode("call_model", callModel, {
    retryPolicy: { maxAttempts: 5, initialInterval: 1.0 }
  })
  .addEdge(START, "call_model")
  .addEdge("call_model", END)
  .compile();
```

### Python: Functional API

```python
from langgraph.func import entrypoint, task
from langgraph.types import RetryPolicy
from langgraph.checkpoint.memory import InMemorySaver

retry_policy = RetryPolicy(max_attempts=3, initial_interval=1.0)

@task(retry_policy=retry_policy)
def get_info():
    # Task implementation
    return "OK"

checkpointer = InMemorySaver()

@entrypoint(checkpointer=checkpointer)
def main(inputs, writer):
    return get_info().result()

config = {"configurable": {"thread_id": "1"}}
main.invoke({'input': 'data'}, config=config)
```

### TypeScript: Functional API

```typescript
import { MemorySaver, entrypoint, task, RetryPolicy } from "@langchain/langgraph";

const retryPolicy: RetryPolicy = {
  maxAttempts: 3,
  initialInterval: 1.0
};

const getInfo = task(
  { name: "getInfo", retry: retryPolicy },
  () => {
    // Task implementation
    return "OK";
  }
);

const checkpointer = new MemorySaver();

const main = entrypoint(
  { checkpointer, name: "main" },
  async (inputs: Record<string, any>) => {
    return await getInfo();
  }
);

const config = { configurable: { thread_id: "1" } };
await main.invoke({ input: "data" }, config);
```

## Custom retry_on Functions

### Python: Retry on Specific Exception

```python
import sqlite3
from langgraph.types import RetryPolicy

# Retry only on sqlite3.OperationalError
builder.add_node(
    "query_database",
    query_database,
    retry_policy=RetryPolicy(retry_on=sqlite3.OperationalError)
)
```

### Python: Retry on Multiple Exceptions

```python
import requests
from langgraph.types import RetryPolicy

# Retry on multiple exception types
builder.add_node(
    "api_call",
    api_call,
    retry_policy=RetryPolicy(
        retry_on=(requests.exceptions.ConnectionError, requests.exceptions.Timeout)
    )
)
```

### Python: Custom Lambda Function

```python
from langgraph.types import RetryPolicy

def should_retry(error: Exception) -> bool:
    # Custom retry logic
    if isinstance(error, ValueError):
        return "rate limit" in str(error).lower()
    return isinstance(error, ConnectionError)

builder.add_node(
    "custom_node",
    custom_function,
    retry_policy=RetryPolicy(retry_on=should_retry)
)
```

### TypeScript: Retry on Specific Error

```typescript
import Database from "better-sqlite3";

const graph = new StateGraph(State)
  .addNode("query_database", queryDatabase, {
    retryPolicy: {
      retryOn: (e: any): boolean => {
        if (e instanceof Database.SqliteError) {
          return e.code === "SQLITE_BUSY";
        }
        return false;
      },
    },
  })
  .compile();
```

### TypeScript: Custom Error Checking

```typescript
const graph = new StateGraph(State)
  .addNode("api_call", apiCallFunction, {
    retryPolicy: {
      maxAttempts: 5,
      retryOn: (error: Error): boolean => {
        // Retry on network errors or rate limits
        return error.message.includes("ECONNREFUSED") ||
               error.message.includes("rate limit");
      },
    },
  })
  .compile();
```

## Backoff Strategies

### Exponential Backoff

LangGraph uses exponential backoff by default. The wait time between retries is calculated as:

```
delay = initial_interval * (backoff_factor ** retry_number)
```

**Example calculation** with `initial_interval=1.0` and `backoff_factor=2.0`:
- Attempt 1: Immediate
- Attempt 2: Wait 1.0 seconds (1.0 * 2^0)
- Attempt 3: Wait 2.0 seconds (1.0 * 2^1)
- Attempt 4: Wait 4.0 seconds (1.0 * 2^2)

### Python: Aggressive Backoff

```python
from langgraph.types import RetryPolicy

# Aggressive backoff for rate-limited APIs
builder.add_node(
    "rate_limited_api",
    api_call,
    retry_policy=RetryPolicy(
        max_attempts=5,
        initial_interval=2.0,
        backoff_factor=3.0  # Triples delay each retry
    )
)
```

**Calculation** with `initial_interval=2.0` and `backoff_factor=3.0`:
- Attempt 1: Immediate
- Attempt 2: Wait 2.0 seconds
- Attempt 3: Wait 6.0 seconds (2.0 * 3^1)
- Attempt 4: Wait 18.0 seconds (2.0 * 3^2)
- Attempt 5: Wait 54.0 seconds (2.0 * 3^3)

### Python: Constant Delay

```python
from langgraph.types import RetryPolicy

# Constant delay (backoff_factor = 1.0)
builder.add_node(
    "constant_retry",
    function,
    retry_policy=RetryPolicy(
        max_attempts=4,
        initial_interval=3.0,
        backoff_factor=1.0  # No exponential growth
    )
)
```

**Calculation**: Every retry waits 3.0 seconds.

### TypeScript: Custom Backoff

```typescript
const graph = new StateGraph(State)
  .addNode("api_call", apiCall, {
    retryPolicy: {
      maxAttempts: 5,
      initialInterval: 1.5,
      backoffFactor: 2.5  // Custom exponential growth
    }
  })
  .compile();
```

## Circuit Breaker Pattern

LangGraph does not have built-in circuit breakers, but you can implement them manually using state tracking.

### Python: Circuit Breaker Implementation

```python
from typing import TypedDict
from datetime import datetime, timedelta
from langgraph.graph import StateGraph, START, END
from langgraph.types import RetryPolicy

class CircuitState(TypedDict):
    consecutive_failures: int
    circuit_open: bool
    last_failure_time: datetime | None
    messages: list

def circuit_breaker_node(state: CircuitState):
    """Check circuit breaker state before executing"""
    if state.get("circuit_open"):
        # Check if enough time has passed to retry
        last_failure = state.get("last_failure_time")
        if last_failure and datetime.now() - last_failure < timedelta(minutes=5):
            return {
                "messages": ["Circuit breaker is open, skipping operation"],
                "circuit_open": True
            }
        # Reset circuit breaker
        return {"circuit_open": False, "consecutive_failures": 0}

    try:
        # Actual operation
        result = perform_risky_operation()
        return {
            "messages": [result],
            "consecutive_failures": 0,
            "circuit_open": False
        }
    except Exception as e:
        failures = state.get("consecutive_failures", 0) + 1
        # Open circuit after 3 consecutive failures
        if failures >= 3:
            return {
                "consecutive_failures": failures,
                "circuit_open": True,
                "last_failure_time": datetime.now(),
                "messages": [f"Circuit breaker opened after {failures} failures"]
            }
        raise  # Let retry policy handle it

def perform_risky_operation():
    # Your actual implementation
    pass

builder = StateGraph(CircuitState)
builder.add_node(
    "circuit_breaker",
    circuit_breaker_node,
    retry_policy=RetryPolicy(max_attempts=3)
)
builder.add_edge(START, "circuit_breaker")
builder.add_edge("circuit_breaker", END)
graph = builder.compile()
```

### TypeScript: Circuit Breaker Implementation

```typescript
interface CircuitState {
  consecutiveFailures: number;
  circuitOpen: boolean;
  lastFailureTime: Date | null;
  messages: string[];
}

const circuitBreakerNode = (state: CircuitState) => {
  if (state.circuitOpen) {
    const lastFailure = state.lastFailureTime;
    if (lastFailure && Date.now() - lastFailure.getTime() < 5 * 60 * 1000) {
      return {
        messages: ["Circuit breaker is open, skipping operation"],
        circuitOpen: true
      };
    }
    // Reset circuit breaker
    return { circuitOpen: false, consecutiveFailures: 0 };
  }

  try {
    const result = performRiskyOperation();
    return {
      messages: [result],
      consecutiveFailures: 0,
      circuitOpen: false
    };
  } catch (error) {
    const failures = (state.consecutiveFailures || 0) + 1;
    if (failures >= 3) {
      return {
        consecutiveFailures: failures,
        circuitOpen: true,
        lastFailureTime: new Date(),
        messages: [`Circuit breaker opened after ${failures} failures`]
      };
    }
    throw error;
  }
};

const graph = new StateGraph(State)
  .addNode("circuit_breaker", circuitBreakerNode, {
    retryPolicy: { maxAttempts: 3 }
  })
  .compile();
```

## Common Patterns

### Pattern 1: LLM Calls with Retry

```python
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.types import RetryPolicy
from langchain.chat_models import init_chat_model

model = init_chat_model("claude-3-5-sonnet-20240620")

def call_llm(state: MessagesState):
    response = model.invoke(state["messages"])
    return {"messages": [response]}

builder = StateGraph(MessagesState)
builder.add_node(
    "llm",
    call_llm,
    retry_policy=RetryPolicy(
        max_attempts=5,
        initial_interval=1.0,
        backoff_factor=2.0
    )
)
builder.add_edge(START, "llm")
builder.add_edge("llm", END)
graph = builder.compile()
```

### Pattern 2: Database Queries

```python
import sqlite3
from langgraph.graph import StateGraph, START, END
from langgraph.types import RetryPolicy
from typing import TypedDict

class DatabaseState(TypedDict):
    query_result: str

def query_database(state: DatabaseState):
    # Database query implementation
    return {"query_result": "data"}

builder = StateGraph(DatabaseState)
builder.add_node(
    "db_query",
    query_database,
    retry_policy=RetryPolicy(
        max_attempts=3,
        retry_on=sqlite3.OperationalError  # Retry only on DB lock errors
    )
)
builder.add_edge(START, "db_query")
builder.add_edge("db_query", END)
graph = builder.compile()
```

### Pattern 3: External API Calls

```python
import requests
from langgraph.graph import StateGraph, START, END
from langgraph.types import RetryPolicy
from typing import TypedDict

class APIState(TypedDict):
    api_response: dict

def call_external_api(state: APIState):
    response = requests.get("https://api.example.com/data")
    response.raise_for_status()
    return {"api_response": response.json()}

def should_retry_api_error(error: Exception) -> bool:
    """Custom retry logic for API errors"""
    if isinstance(error, requests.exceptions.HTTPError):
        # Retry on 5xx and 429 (rate limit)
        return error.response.status_code >= 500 or error.response.status_code == 429
    # Retry on connection errors
    return isinstance(error, (requests.exceptions.ConnectionError, requests.exceptions.Timeout))

builder = StateGraph(APIState)
builder.add_node(
    "api_call",
    call_external_api,
    retry_policy=RetryPolicy(
        max_attempts=5,
        initial_interval=2.0,
        backoff_factor=2.0,
        retry_on=should_retry_api_error
    )
)
builder.add_edge(START, "api_call")
builder.add_edge("api_call", END)
graph = builder.compile()
```

### Pattern 4: Multi-Node Graph with Different Retry Policies

```python
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.types import RetryPolicy
from langchain.chat_models import init_chat_model
import requests

model = init_chat_model("claude-3-5-sonnet-20240620")

def search_documentation(state: MessagesState):
    # Search implementation
    response = requests.get("https://docs.example.com/search")
    return {"messages": [response.text]}

def call_llm(state: MessagesState):
    response = model.invoke(state["messages"])
    return {"messages": [response]}

builder = StateGraph(MessagesState)

# Network-heavy node: aggressive retry
builder.add_node(
    "search",
    search_documentation,
    retry_policy=RetryPolicy(max_attempts=5, initial_interval=1.0)
)

# LLM node: moderate retry
builder.add_node(
    "llm",
    call_llm,
    retry_policy=RetryPolicy(max_attempts=3, initial_interval=2.0)
)

builder.add_edge(START, "search")
builder.add_edge("search", "llm")
builder.add_edge("llm", END)
graph = builder.compile()
```

### TypeScript: Complete Example

```typescript
import { StateGraph, StateSchema, MessagesValue } from "@langchain/langgraph";
import { ChatAnthropic } from "@langchain/anthropic";
import axios from "axios";

const State = new StateSchema({ messages: MessagesValue });

const searchDocs = async (state: typeof State.Type) => {
  const response = await axios.get("https://docs.example.com/search");
  return { messages: [response.data] };
};

const callLLM = async (state: typeof State.Type) => {
  const model = new ChatAnthropic({ model: "claude-3-5-sonnet-20240620" });
  const response = await model.invoke(state.messages);
  return { messages: [response] };
};

const graph = new StateGraph(State)
  .addNode("search", searchDocs, {
    retryPolicy: {
      maxAttempts: 5,
      initialInterval: 1.0,
      retryOn: (error) => {
        // Retry on network errors and 5xx
        return error.message.includes("ECONNREFUSED") ||
               (error.response && error.response.status >= 500);
      }
    }
  })
  .addNode("llm", callLLM, {
    retryPolicy: { maxAttempts: 3, initialInterval: 2.0 }
  })
  .addEdge(START, "search")
  .addEdge("search", "llm")
  .addEdge("llm", END)
  .compile();
```

## Best Practices

1. **Match retry attempts to failure likelihood**: Use higher `max_attempts` for known flaky services.
2. **Use specific exception types**: Avoid retrying on non-transient errors like `ValueError` or `TypeError`.
3. **Adjust backoff for rate limits**: Use higher `initial_interval` and `backoff_factor` for rate-limited APIs.
4. **Implement circuit breakers**: For critical services, combine retry policies with circuit breaker patterns.
5. **Monitor retry metrics**: Track retry counts and failure rates to identify problematic nodes.
6. **Consider timeout policies**: Combine retry policies with appropriate timeout configurations.
