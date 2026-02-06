# Error Types and Classification

Comprehensive catalog of error types for LangGraph applications, based on the official 4-category error handling system.

## Error Classification System

LangGraph provides a structured approach to error handling based on who can fix the error:

| Error Type | Who Fixes | Strategy | When to Use |
|------------|-----------|----------|-------------|
| **Transient** | System (automatic) | `RetryPolicy` | Network timeouts, rate limits, temporary API failures |
| **LLM-recoverable** | LLM | Store in state, loop back | Tool failures, parsing errors, validation errors |
| **User-fixable** | Human | `interrupt()` | Missing required input, ambiguous instructions, approvals |
| **Unexpected** | Developer | Let bubble up | Unknown bugs, programming errors, auth failures |

## Transient Errors

Errors that usually resolve on retry. Use `RetryPolicy` to handle automatically.

### Network Library Errors

| Exception | Source Library | Description | Notes |
|-----------|---------------|-------------|-------|
| `TimeoutException` | httpx | Request exceeded timeout | Common with slow APIs |
| `ConnectTimeout` | httpx | Failed to establish connection | Network connectivity |
| `ReadTimeout` | httpx | Server didn't respond in time | Server overload |
| `ConnectError` | httpx | Connection establishment failed | DNS, routing issues |
| `ConnectionError` | requests | Network connection failed | General connectivity |
| `Timeout` | requests | Request timeout | requests library timeout |
| `ReadTimeout` | requests | Reading response timed out | Server delays |
| `ServerDisconnectedError` | aiohttp | Server closed connection | Async connection issues |
| `ClientConnectionError` | aiohttp | Client-side connection error | Async network issues |
| `ClientConnectorError` | aiohttp | Connection establishment failed | Async DNS/routing |

### LLM Provider Errors

| Exception | Provider | Description | Retry Recommended |
|-----------|----------|-------------|-------------------|
| `RateLimitError` | openai, anthropic | Hit rate limit | Yes (with backoff) |
| `APITimeoutError` | openai, anthropic | API request timeout | Yes |
| `InternalServerError` | openai, anthropic | Provider 5xx error | Yes |
| `APIConnectionError` | openai, anthropic | Connection to API failed | Yes |
| `ServiceUnavailableError` | openai, anthropic | Service temporarily down | Yes |

### HTTP Status Codes (Retryable)

| Status Code | Meaning | RetryPolicy Behavior |
|-------------|---------|---------------------|
| 500 | Internal Server Error | Retry |
| 502 | Bad Gateway | Retry |
| 503 | Service Unavailable | Retry |
| 504 | Gateway Timeout | Retry |

**Note**: For exceptions from popular HTTP libraries (`requests`, `httpx`), default `RetryPolicy` behavior only retries 5xx status codes. 4xx errors, including 429, are not retried unless you customize `retry_on`.

## LLM-Recoverable Errors

Errors the LLM can potentially fix if it sees the error message in state.

### Parsing & Validation Errors

| Exception | Source Library | Description | Recovery Strategy |
|-----------|---------------|-------------|-------------------|
| `OutputParserException` | langchain_core.output_parsers | LLM output doesn't match expected format | Show error to LLM, let it retry |
| `ValidationError` | pydantic | Schema validation failed | Return validation details to LLM |
| `JSONDecodeError` | json | Invalid JSON output | Prompt LLM to fix JSON |
| `ToolInputParsingException` | langchain_core.tools | Tool arguments invalid | LLM sees error, adjusts args |

### Tool Execution Errors

| Exception | Context | Description | Recovery Strategy |
|-----------|---------|-------------|-------------------|
| `ToolException` | langchain_core.tools | Tool execution failed | Return error message to LLM |
| Custom tool errors | Any tool | Tool-specific failures | Use `ToolNode(handle_tool_errors=True)` |
| API errors (4xx except 429) | External APIs | Bad request, not found | LLM adjusts parameters |

### Structured Output Errors

| Error Code | Description | Recovery |
|------------|-------------|----------|
| `OUTPUT_PARSING_FAILURE` | Failed to parse LLM structured output | LLM retries with corrected format |
| `INVALID_TOOL_RESULTS` | Tool returned invalid format | LLM adjusts tool call |
| `StructuredOutputValidationError` | Schema validation failed | Error message shown to LLM |

## User-Fixable Errors

Errors requiring human intervention or input.

### Common Scenarios

| Scenario | Example | Implementation |
|----------|---------|----------------|
| **Missing required info** | Customer ID, account number | `interrupt()` with request details |
| **Ambiguous request** | Multiple matching records | Pause, ask user to clarify |
| **Sensitive operations** | Delete data, send email | Require explicit approval |
| **External dependencies** | Payment authorization | Wait for user action |
| **Insufficient permissions** | User lacks access rights | Prompt user to get access |

### Implementation Pattern

```python
from langgraph.types import interrupt, Command

def requires_customer_id(state):
    if not state.get('customer_id'):
        user_input = interrupt({
            "message": "Customer ID needed",
            "request": "Please provide the customer account ID"
        })
        return Command(
            update={"customer_id": user_input['customer_id']},
            goto="requires_customer_id"  # Resume here
        )
    # Continue processing
```

### Interrupt Response

When `interrupt()` is called, the graph returns:

```json
{
  "__interrupt__": [{
    "value": {"message": "Customer ID needed", ...},
    "resumable": true,
    "ns": ["node_name:..."],
    "when": "during"
  }]
}
```

Resume with: `Command(resume=user_input)` or in SDK: `client.runs.wait(thread_id, assistant_id, command={"resume": data})`

## Unexpected Errors (Fatal)

Errors that should bubble up for developer attention.

### Python Built-in Exceptions (Not Retried by Default)

| Exception | Category | Retry Default | Reason |
|-----------|----------|---------------|--------|
| `ValueError` | Logic error | **No** | Programming mistake |
| `TypeError` | Type mismatch | **No** | Code bug |
| `AttributeError` | Missing attribute | **No** | Code bug |
| `KeyError` | Missing dict key | **No** | Logic error |
| `IndexError` | List index out of range | **No** | Logic error |
| `NameError` | Undefined variable | **No** | Code bug |
| `SyntaxError` | Invalid Python syntax | **No** | Code bug |
| `ImportError` | Import failed | **No** | Dependency issue |
| `RuntimeError` | Generic runtime error | **No** | Varies |
| `ArithmeticError` | Math operation failed | **No** | Logic error |
| `LookupError` | Lookup failed | **No** | Logic error |
| `ReferenceError` | Weak reference invalid | **No** | Memory issue |
| `StopIteration` | Iterator exhausted | **No** | Control flow |
| `StopAsyncIteration` | Async iterator exhausted | **No** | Control flow |
| `OSError` | OS-level failure | **No** | System issue |

### JavaScript Built-in Exceptions (Not Retried by Default)

| Exception | Reason |
|-----------|--------|
| `TypeError` | Code bug |
| `SyntaxError` | Code bug |
| `ReferenceError` | Code bug |

### Authentication & Authorization Errors

| Exception | Source | Should Retry? | Reason |
|-----------|--------|---------------|--------|
| `AuthenticationError` | LLM providers | **No** | Invalid API key |
| `PermissionDeniedError` | LLM providers | **No** | Insufficient permissions |
| `InvalidRequestError` | LLM providers | **No** | Malformed request |
| MODEL_AUTHENTICATION | LangGraph error code | **No** | Auth failure |

### LangGraph-Specific Errors

| Error Code | Description | Should Retry? |
|------------|-------------|---------------|
| `GRAPH_RECURSION_LIMIT` | Max recursion depth exceeded | No - design issue |
| `INVALID_GRAPH_NODE_RETURN_VALUE` | Node returned invalid value | No - code bug |
| `INVALID_CONCURRENT_GRAPH_UPDATE` | Concurrent update conflict | No - logic error |
| `MISSING_CHECKPOINTER` | Checkpointer required but not set | No - config error |
| `MULTIPLE_SUBGRAPHS` | Multiple subgraphs detected | No - design issue |
| `INVALID_CHAT_HISTORY` | Invalid message format | No - data error |
| `INVALID_PROMPT_INPUT` | Invalid prompt input | No - validation error |
| `MESSAGE_COERCION_FAILURE` | Failed to coerce message | No - format error |

## Superstep Transaction Behavior

LangGraph executes nodes within **supersteps** - transactional units that affect error handling:

### Key Behaviors

| Behavior | Description | Impact |
|----------|-------------|--------|
| **Transactional** | If any parallel branch fails, entire superstep rolls back | No partial state updates |
| **Checkpointer saves successes** | With checkpointer, successful nodes don't re-run | Avoids redundant work |
| **Only failing branches retry** | RetryPolicy only retries failed nodes | Efficient error recovery |
| **Parallel execution** | Branches execute in parallel within superstep | Better performance |

### Example

```python
# Superstep with 3 parallel nodes
workflow.add_edge(START, ["node_a", "node_b", "node_c"])

# Scenario:
# - node_a: succeeds
# - node_b: fails (retryable error)
# - node_c: succeeds

# Without RetryPolicy:
# → Entire superstep fails
# → No state updates applied
# → Graph execution stops

# With RetryPolicy on node_b:
# → node_b retries (node_a, node_c don't re-run)
# → If retry succeeds, superstep completes
# → All state updates applied together

# With Checkpointer + RetryPolicy:
# → Successful nodes (a, c) checkpointed
# → On resume, only node_b re-executes
# → Avoids repeating successful work
```

### Resume After Failure

```python
from langgraph.checkpoint.memory import MemorySaver

checkpointer = MemorySaver()
graph = workflow.compile(checkpointer=checkpointer)

config = {"configurable": {"thread_id": "abc123"}}

# First call: node_b fails
try:
    graph.invoke(input_data, config)
except Exception:
    pass  # Checkpoint saved

# Fix the issue (e.g., API comes back online)
# Resume from checkpoint - only failed nodes retry
result = graph.invoke(None, config)  # Resumes from checkpoint
```

## RetryPolicy Configuration

### Default Behavior

```python
from langgraph.types import RetryPolicy

# Common explicit configuration
RetryPolicy(
    max_attempts=3,              # Total: 1 initial + 2 retries
    initial_interval=1.0,        # 1s before first retry
    backoff_factor=2.0,          # Exponential backoff
    retry_on=default_retry_on    # See "Not Retried by Default" above
)
```

### Custom Configuration

```python
# Retry only specific errors
workflow.add_node(
    "api_call",
    api_node,
    retry_policy=RetryPolicy(
        max_attempts=5,
        retry_on=lambda e: isinstance(e, (TimeoutError, RateLimitError))
    )
)

# Aggressive retry for critical operations
workflow.add_node(
    "database_query",
    db_node,
    retry_policy=RetryPolicy(
        max_attempts=10,
        initial_interval=0.1,
        backoff_factor=1.5,
        retry_on=sqlite3.OperationalError
    )
)
```

## Best Practices Summary

| Error Type | Do | Don't |
|------------|----|----|
| **Transient** | Use RetryPolicy with exponential backoff | Retry indefinitely without backoff |
| **LLM-recoverable** | Store error in state, loop to LLM node | Silently catch and hide from LLM |
| **User-fixable** | Use `interrupt()` with clear message | Block indefinitely without feedback |
| **Unexpected** | Let bubble up, log for debugging | Catch all exceptions blindly |

### Error Handling Checklist

- [ ] Apply `RetryPolicy` to nodes calling external APIs
- [ ] Configure `handle_tool_errors=True` on ToolNode for LLM recovery
- [ ] Use `interrupt()` for operations requiring human input
- [ ] Let programming errors bubble up (don't catch ValueError, TypeError)
- [ ] Use checkpointer for production to preserve progress
- [ ] Monitor logs for unexpected errors
- [ ] Test retry behavior with network failures
- [ ] Document expected error scenarios in code comments
