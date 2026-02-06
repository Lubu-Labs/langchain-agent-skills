# Human Escalation with Interrupts

Human-in-the-loop (HITL) error escalation allows LangGraph agents to pause execution and request human intervention when they encounter situations they cannot handle autonomously. This pattern combines error handling with the `interrupt()` function to create robust, production-ready agents.

## When to Escalate to Humans

Not all errors should be escalated to humans. Use human escalation for:

### User-Fixable Errors
- Missing required information (account IDs, order numbers, user preferences)
- Ambiguous instructions that need clarification
- Authorization or permission issues requiring user action
- Configuration errors that need user input

### Sensitive Operations
- Financial transactions above certain thresholds
- Data deletion or destructive operations
- Access to sensitive information
- Actions with compliance implications

### Unrecoverable Errors
- API failures where user needs to provide alternative approach
- Resource constraints requiring user decision (e.g., budget exceeded)
- Multi-step failures where LLM cannot determine next action

**Do NOT escalate:**
- Transient errors (network issues, rate limits) - use retry policies
- LLM-recoverable errors (tool failures, parsing issues) - loop back to LLM
- Unexpected system errors - let them bubble up for debugging

## The `interrupt()` Function

The `interrupt()` function pauses graph execution and waits for external input before continuing.

### Python

```python
from langgraph.types import interrupt

def approval_node(state: State):
    # Pause and ask for approval - accepts any JSON-serializable value
    user_input = interrupt({
        "question": "Approve this action?",
        "details": state["action_details"]
    })

    # When resumed, user_input contains the resume value
    return {"status": "approved" if user_input else "rejected"}
```

### TypeScript

```typescript
import { interrupt } from "@langchain/langgraph";

async function approvalNode(state: State) {
    // Pause and ask for approval
    const userInput = interrupt({
        question: "Approve this action?",
        details: state.actionDetails
    });

    // When resumed, userInput contains the resume value
    return { status: userInput ? "approved" : "rejected" };
}
```

### Requirements

**CRITICAL**: `interrupt()` requires a checkpointer to work:

```python
from langgraph.checkpoint.memory import MemorySaver

# Use MemorySaver for development
checkpointer = MemorySaver()

# Use persistent checkpointer in production
# from langgraph.checkpoint.postgres import AsyncPostgresSaver
# checkpointer = AsyncPostgresSaver(...)

graph = builder.compile(checkpointer=checkpointer)
```

### Key Behaviors

1. **Accepts any JSON-serializable value** - Pass objects, strings, lists, etc.
2. **Pauses execution** - Graph stops at the interrupt point
3. **Saves state** - Checkpointer persists graph state for later resume
4. **Returns resume value** - The value passed to `Command(resume=...)` becomes the return value
5. **Surfaces as `__interrupt__`** - Interrupt payload appears in the result for the caller

## Approve/Reject Pattern

The most common HITL pattern: pause before a critical action and route based on approval.

### Python

```python
from typing import Literal
from langgraph.types import interrupt, Command

def approval_node(state: State) -> Command[Literal["proceed", "cancel"]]:
    # Present action details for review
    is_approved = interrupt({
        "question": "Do you want to proceed with this action?",
        "details": state["action_details"],
        "estimated_cost": state.get("cost")
    })

    # Route based on the response
    if is_approved:
        return Command(goto="proceed")
    else:
        return Command(goto="cancel")
```

**Resume execution:**

```python
config = {"configurable": {"thread_id": "approval-123"}}

# Initial invocation - runs until interrupt
result = graph.invoke(
    {"action_details": "Delete 1000 records", "status": "pending"},
    config=config
)

# Check for interrupt
if result.get("__interrupt__"):
    print(result["__interrupt__"])  # See the payload

    # Resume with approval
    result = graph.invoke(Command(resume=True), config=config)

    # Or resume with rejection
    # result = graph.invoke(Command(resume=False), config=config)
```

### TypeScript

```typescript
import { Command, interrupt } from "@langchain/langgraph";

async function approvalNode(state: State) {
    // Present action details for review
    const isApproved = interrupt({
        question: "Do you want to proceed with this action?",
        details: state.actionDetails,
        estimatedCost: state.cost
    });

    // Route based on the response
    return new Command({
        goto: isApproved ? "proceed" : "cancel"
    });
}
```

**Resume execution:**

```typescript
const config = { configurable: { thread_id: "approval-123" } };

// Initial invocation - runs until interrupt
let result = await graph.invoke(
    { actionDetails: "Delete 1000 records", status: "pending" },
    config
);

// Check for interrupt
if (result.__interrupt__) {
    console.log(result.__interrupt__); // See the payload

    // Resume with approval
    result = await graph.invoke(
        new Command({ resume: true }),
        config
    );

    // Or resume with rejection
    // result = await graph.invoke(new Command({ resume: false }), config);
}
```

## HITL Middleware for Tool Approval

For agent tools, use built-in HITL middleware instead of manual `interrupt()` calls.

### Python

```python
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langgraph.checkpoint.memory import MemorySaver

agent = create_agent(
    model="gpt-4.1",
    tools=[write_file, execute_sql, read_data],
    middleware=[
        HumanInTheLoopMiddleware(
            interrupt_on={
                # All decisions allowed (approve, edit, reject)
                "write_file": True,

                # Only approve or reject (no editing)
                "execute_sql": {"allowed_decisions": ["approve", "reject"]},

                # Auto-approve (no interrupt)
                "read_data": False,
            },
            description_prefix="Tool execution pending approval"
        )
    ],
    checkpointer=MemorySaver()
)
```

### TypeScript

```typescript
import { createAgent, humanInTheLoopMiddleware } from "langchain";
import { MemorySaver } from "@langchain/langgraph";

const agent = createAgent({
    model: "gpt-4.1",
    tools: [writeFile, executeSql, readData],
    middleware: [
        humanInTheLoopMiddleware({
            interruptOn: {
                // All decisions allowed
                "writeFile": true,

                // Only approve or reject
                "executeSql": { allowedDecisions: ["approve", "reject"] },

                // Auto-approve
                "readData": false,
            }
        })
    ],
    checkpointer: new MemorySaver()
});
```

### Handling Tool Interrupts

The `__interrupt__` field contains `action_requests` and `review_configs`:

```python
from langgraph.types import Command

config = {"configurable": {"thread_id": "some_id"}}

# Run until interrupt
result = agent.invoke(
    {"messages": [{"role": "user", "content": "Delete old database records"}]},
    config=config
)

# Check for interrupt
if result.get("__interrupt__"):
    interrupt_data = result["__interrupt__"][0].value

    # action_requests: list of tools to review
    for action in interrupt_data["action_requests"]:
        print(f"Tool: {action['name']}")
        print(f"Args: {action['arguments']}")
        print(f"Description: {action['description']}")

    # review_configs: allowed decisions per tool
    for config_item in interrupt_data["review_configs"]:
        print(f"Tool: {config_item['action_name']}")
        print(f"Allowed: {config_item['allowed_decisions']}")
```

### Decision Types

Resume with one of three decision types:

**Approve** - Execute tool as-is:
```python
agent.invoke(
    Command(resume={"decisions": [{"type": "approve"}]}),
    config=config
)
```

**Edit** - Modify tool arguments before execution:
```python
agent.invoke(
    Command(resume={
        "decisions": [{
            "type": "edit",
            "edited_action": {
                "name": "execute_sql",
                "args": {"query": "DELETE FROM records WHERE age > 90"}
            }
        }]
    }),
    config=config
)
```

**Reject** - Cancel tool execution with feedback:
```python
agent.invoke(
    Command(resume={
        "decisions": [{
            "type": "reject",
            "message": "Cannot delete records without WHERE clause. Add a condition."
        }]
    }),
    config=config
)
```

### Multiple Tool Decisions

When a **single interrupt** contains multiple tool calls, provide decisions in the same order as `action_requests`:

```python
# Three tools waiting for approval
decisions = [
    {"type": "approve"},              # First tool
    {"type": "edit", "edited_action": {...}},  # Second tool
    {"type": "reject", "message": "..."}       # Third tool
]

agent.invoke(
    Command(resume={"decisions": decisions}),
    config=config
)
```

When a run returns **multiple interrupt events**, resume by interrupt ID:

```python
from langgraph.types import Command

resume = {}
for interrupt_ in result["__interrupt__"]:
    # Build per-interrupt decisions payload
    resume[interrupt_.id] = {"decisions": [{"type": "approve"}]}

agent.invoke(Command(resume=resume), config=config)
```

## Critical Rules for `interrupt()`

### 1. Do NOT Reorder Interrupts

Interrupts are matched by index. Keep them in consistent order across executions.

**BAD - Conditional skip changes order:**
```python
def node_a(state: State):
    name = interrupt("What's your name?")

    # On first run: might skip this interrupt
    # On resume: might not skip it - INDEX MISMATCH!
    if state.get("needs_age"):
        age = interrupt("What's your age?")

    city = interrupt("What's your city?")
    return {"name": name, "city": city}
```

**GOOD - Consistent order:**
```python
def node_a(state: State):
    name = interrupt("What's your name?")

    # Always call the same interrupts in the same order
    age = interrupt("What's your age?") if state.get("needs_age") else None

    city = interrupt("What's your city?")
    return {"name": name, "age": age, "city": city}
```

### 2. Do NOT Loop Interrupts Non-Deterministically

**BAD - Dynamic list changes between runs:**
```python
def node_a(state: State):
    results = []
    # List length might change between initial run and resume!
    for item in state.get("dynamic_list", []):
        result = interrupt(f"Approve {item}?")
        results.append(result)
    return {"results": results}
```

**GOOD - Fixed list at node entry:**
```python
def node_a(state: State):
    # Capture list at start - deterministic across resume
    items = state.get("items_to_approve", [])
    results = []
    for item in items:
        result = interrupt(f"Approve {item}?")
        results.append(result)
    return {"results": results}
```

### 3. Do NOT Wrap in Try/Except

`interrupt()` throws a special exception to pause execution. Don't catch it!

**BAD - Catches interrupt exception:**
```python
def node_a(state: State):
    try:
        name = interrupt("What's your name?")
    except Exception as e:
        print(e)  # This catches the interrupt exception!
    return state
```

**GOOD - Separate interrupt from error handling:**
```python
def node_a(state: State):
    # Interrupt first
    name = interrupt("What's your name?")

    # Then handle errors separately
    try:
        data = fetch_data(name)
    except Exception as e:
        print(e)

    return state
```

**ACCEPTABLE - Catch specific exceptions:**
```python
def node_a(state: State):
    try:
        name = interrupt("What's your name?")
        data = fetch_data(name)
    except NetworkException as e:
        # Specific exception won't catch interrupt
        print(e)
    return state
```

### 4. Always Place Interrupt BEFORE Other Operations

Place interrupts at the start of nodes to avoid partial execution before pausing.

**BAD - Operations before interrupt:**
```python
def risky_node(state: State):
    # These operations run before interrupt - might leave system in bad state!
    update_database(state["data"])
    send_notification(state["user"])

    # If user rejects, database already updated!
    approved = interrupt("Approve changes?")

    if not approved:
        rollback_database()  # Too late!
```

**GOOD - Interrupt before operations:**
```python
def safe_node(state: State):
    # Ask first
    approved = interrupt({
        "question": "Approve changes?",
        "details": state["data"]
    })

    if approved:
        # Only proceed after approval
        update_database(state["data"])
        send_notification(state["user"])

    return state
```

## Error Escalation Pattern

Combine error handling with interrupts to escalate unrecoverable errors to humans.

### Python

```python
from langgraph.types import interrupt, Command
from typing import Literal

def risky_operation_node(state: State) -> Command[Literal["retry", "abort"]]:
    try:
        # Attempt the operation
        result = execute_api_call(state["api_params"])
        return Command(
            update={"result": result, "status": "success"},
            goto="next_step"
        )

    except RateLimitError as e:
        # Transient error - let system retry
        return Command(
            update={"error": str(e), "status": "rate_limited"},
            goto="retry"
        )

    except AuthenticationError as e:
        # User-fixable error - escalate to human
        user_guidance = interrupt({
            "error_type": "authentication_failed",
            "error_message": str(e),
            "question": "API authentication failed. Please provide action:",
            "options": [
                "Retry with new credentials",
                "Skip this operation",
                "Abort workflow"
            ],
            "context": {
                "api_endpoint": state["api_params"]["endpoint"],
                "attempted_at": state["timestamp"]
            }
        })

        # Process user guidance
        if user_guidance["action"] == "retry":
            return Command(
                update={
                    "api_params": user_guidance.get("new_params", state["api_params"]),
                    "status": "retrying"
                },
                goto="retry"
            )
        elif user_guidance["action"] == "skip":
            return Command(
                update={"status": "skipped"},
                goto="next_step"
            )
        else:
            return Command(
                update={"status": "aborted"},
                goto="abort"
            )

    except ValidationError as e:
        # LLM-recoverable error - loop back with error context
        return Command(
            update={
                "error": f"Validation failed: {str(e)}",
                "status": "validation_error"
            },
            goto="agent"  # Let LLM see error and adjust
        )

    except Exception as e:
        # Unexpected error - let it bubble up
        raise
```

### TypeScript

```typescript
import { Command, interrupt } from "@langchain/langgraph";

async function riskyOperationNode(state: State): Promise<Command> {
    try {
        // Attempt the operation
        const result = await executeApiCall(state.apiParams);
        return new Command({
            update: { result, status: "success" },
            goto: "nextStep"
        });
    }
    catch (error) {
        // Check error type and handle appropriately
        if (error instanceof RateLimitError) {
            // Transient error - let system retry
            return new Command({
                update: { error: error.message, status: "rate_limited" },
                goto: "retry"
            });
        }
        else if (error instanceof AuthenticationError) {
            // User-fixable error - escalate to human
            const userGuidance = interrupt({
                errorType: "authentication_failed",
                errorMessage: error.message,
                question: "API authentication failed. Please provide action:",
                options: [
                    "Retry with new credentials",
                    "Skip this operation",
                    "Abort workflow"
                ],
                context: {
                    apiEndpoint: state.apiParams.endpoint,
                    attemptedAt: state.timestamp
                }
            });

            // Process user guidance
            if (userGuidance.action === "retry") {
                return new Command({
                    update: {
                        apiParams: userGuidance.newParams || state.apiParams,
                        status: "retrying"
                    },
                    goto: "retry"
                });
            }
            else if (userGuidance.action === "skip") {
                return new Command({
                    update: { status: "skipped" },
                    goto: "nextStep"
                });
            }
            else {
                return new Command({
                    update: { status: "aborted" },
                    goto: "abort"
                });
            }
        }
        else if (error instanceof ValidationError) {
            // LLM-recoverable error - loop back with error context
            return new Command({
                update: {
                    error: `Validation failed: ${error.message}`,
                    status: "validation_error"
                },
                goto: "agent"  // Let LLM see error and adjust
            });
        }
        else {
            // Unexpected error - let it bubble up
            throw error;
        }
    }
}
```

### Complete Error Escalation Example

```python
from typing import Literal, TypedDict, Optional
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command, interrupt


class TransferState(TypedDict):
    amount: float
    account: str
    status: Optional[Literal["pending", "approved", "failed", "completed"]]
    error: Optional[str]
    retry_count: int


def validate_transfer(state: TransferState) -> Command[Literal["process", "escalate"]]:
    """Validate transfer and escalate on errors."""
    try:
        # Validation logic
        if state["amount"] > 10000:
            raise ValueError("Amount exceeds limit")
        if not state["account"].startswith("ACC"):
            raise ValueError("Invalid account format")

        return Command(goto="process")

    except ValueError as e:
        # User-fixable validation error - escalate
        user_decision = interrupt({
            "error_type": "validation_error",
            "error_message": str(e),
            "question": "Transfer validation failed. How to proceed?",
            "transfer_details": {
                "amount": state["amount"],
                "account": state["account"]
            },
            "options": ["Fix and retry", "Cancel transfer"]
        })

        if user_decision["action"] == "retry":
            # User provided corrected data
            return Command(
                update={
                    "amount": user_decision.get("amount", state["amount"]),
                    "account": user_decision.get("account", state["account"]),
                    "retry_count": state["retry_count"] + 1
                },
                goto="validate"  # Try validation again
            )
        else:
            return Command(
                update={"status": "failed", "error": "Cancelled by user"},
                goto=END
            )


def process_transfer(state: TransferState) -> TransferState:
    """Execute the transfer."""
    # Process transfer logic
    return {"status": "completed"}


# Build graph
builder = StateGraph(TransferState)
builder.add_node("validate", validate_transfer)
builder.add_node("process", process_transfer)
builder.add_edge(START, "validate")
builder.add_edge("process", END)

checkpointer = MemorySaver()
graph = builder.compile(checkpointer=checkpointer)

# Execute with error escalation
config = {"configurable": {"thread_id": "transfer-789"}}

result = graph.invoke(
    {"amount": 15000, "account": "12345", "status": "pending", "retry_count": 0},
    config=config
)

# Check for interrupt
if result.get("__interrupt__"):
    print("Validation failed, awaiting user input...")
    print(result["__interrupt__"][0].value)

    # Simulate user providing corrected data
    result = graph.invoke(
        Command(resume={
            "action": "retry",
            "amount": 9500,
            "account": "ACC-12345"
        }),
        config=config
    )

print(f"Final status: {result['status']}")
```

## Best Practices

1. **Always use a checkpointer** - Interrupts require persistence
2. **Keep interrupts deterministic** - Same order, same conditions
3. **Interrupt early** - Before operations, not after
4. **Provide rich context** - Include all information users need to decide
5. **Handle resume values** - Validate user input after resume
6. **Use HITL middleware for tools** - Don't reinvent tool approval
7. **Separate error types** - Route to appropriate handler (system/LLM/human)
8. **Test resume paths** - Verify behavior after interrupts
9. **Use persistent checkpointer in production** - MemorySaver is for development only
10. **Document interrupt contracts** - What values are expected on resume
