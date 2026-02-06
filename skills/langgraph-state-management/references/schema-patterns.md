# State Schema Patterns

Common state schema patterns for LangGraph applications with examples for different use cases.

## Chat Application State

### Basic Chat State

```python
from langgraph.graph import MessagesState

class ChatState(MessagesState):
    """State for basic chat application."""
```

**TypeScript:**
```typescript
import { StateSchema, MessagesValue } from "@langchain/langgraph";

const ChatState = new StateSchema({
  messages: MessagesValue,
});
```

### Chat with Session Management

```python
class SessionChatState(MessagesState):
    """Chat state with session tracking."""
    session_id: str
    user_id: str
    context: dict
    last_activity: float
```

### Multi-Turn Conversation State

```python
class ConversationState(MessagesState):
    """State for multi-turn conversations with context."""
    conversation_id: str
    turn_count: int
    context_summary: str
    topics: list[str]
```

## Research Agent State

### Basic Research State

```python
class ResearchState(TypedDict):
    """State for research agent."""
    query: str
    search_results: list[dict]
    processed_sources: list[str]
    summary: str
    citations: list[str]
```

### Multi-Source Research State

```python
def extend_list(left: list, right: list) -> list:
    """Extend list reducer."""
    return left + right

class MultiSourceResearchState(TypedDict):
    """Research from multiple sources."""
    query: str

    # Sources
    web_results: Annotated[list[dict], extend_list]
    academic_results: Annotated[list[dict], extend_list]
    news_results: Annotated[list[dict], extend_list]

    # Processing
    processed_count: int
    total_sources: int

    # Output
    synthesis: str
    confidence: float
```

### Research with Validation State

```python
class ValidatedResearchState(TypedDict):
    """Research state with fact-checking."""
    query: str

    # Research phase
    raw_results: list[dict]
    facts: list[dict]

    # Validation phase
    validated_facts: list[dict]
    contradictions: list[dict]
    confidence_scores: dict[str, float]

    # Output
    final_report: str
```

## Workflow Orchestration State

### Task Workflow State

```python
from typing import Literal

class TaskWorkflowState(TypedDict):
    """State for task-based workflow."""
    task: str
    status: Literal["pending", "in_progress", "completed", "failed"]
    current_step: str
    steps_completed: list[str]
    results: dict[str, Any]
    errors: list[str]
```

### Approval Workflow State

```python
class ApprovalWorkflowState(TypedDict):
    """Workflow with human approval steps."""
    request: dict

    # Processing
    draft: str
    status: Literal["draft", "pending_approval", "approved", "rejected"]

    # Approval tracking
    approver: str
    approval_timestamp: float
    rejection_reason: str | None

    # Retry
    revision_count: int
    max_revisions: int
```

### Pipeline State

```python
class PipelineState(TypedDict):
    """Multi-stage pipeline state."""
    input_data: dict

    # Stage outputs
    stage1_output: dict
    stage2_output: dict
    stage3_output: dict

    # Tracking
    current_stage: int
    stages_completed: list[str]
    stage_errors: dict[str, str]

    # Final
    final_output: dict
    pipeline_success: bool
```

## Tool-Calling Agent State

### Basic Tool-Calling State

```python
from langgraph.graph import MessagesState

class ToolCallingState(MessagesState):
    """State for agent with tools."""

    # Tool tracking
    tool_calls_made: list[str]
    tool_results: dict[str, Any]

    # Control
    should_continue: bool
    max_iterations: int
    current_iteration: int
```

### Advanced Tool State

```python
class AdvancedToolState(MessagesState):
    """Tool-calling with error handling and retries."""

    # Tool execution
    pending_tools: list[dict]
    completed_tools: list[dict]
    failed_tools: list[dict]

    # Retry logic
    retry_count: dict[str, int]
    max_retries: int

    # Results
    tool_outputs: dict[str, Any]
    final_answer: str
```

## RAG (Retrieval-Augmented Generation) State

### Basic RAG State

```python
class RAGState(TypedDict):
    """State for RAG application."""
    query: str

    # Retrieval
    retrieved_docs: list[dict]
    relevance_scores: list[float]

    # Generation
    context: str
    response: str
    sources: list[str]
```

### Multi-Index RAG State

```python
class MultiIndexRAGState(TypedDict):
    """RAG with multiple vector stores."""
    query: str

    # Retrieval from multiple indexes
    docs_index1: list[dict]
    docs_index2: list[dict]
    docs_index3: list[dict]

    # Reranking
    reranked_docs: list[dict]

    # Generation
    context: str
    response: str
    confidence: float
```

### Conversational RAG State

```python
class ConversationalRAGState(MessagesState):
    """RAG with conversation history."""

    # Context
    conversation_context: str

    # Retrieval
    query_rephrased: str
    retrieved_docs: list[dict]

    # Generation
    response: str
    sources: list[str]
```

## Data Processing State

### Batch Processing State

```python
class BatchProcessingState(TypedDict):
    """State for batch data processing."""
    items: list[dict]

    # Processing
    processed_items: Annotated[list[dict], extend_list]
    failed_items: Annotated[list[dict], extend_list]

    # Progress
    total_count: int
    processed_count: int
    failed_count: int

    # Results
    results: dict
    processing_complete: bool
```

### ETL Pipeline State

```python
class ETLState(TypedDict):
    """Extract-Transform-Load pipeline."""
    # Extract
    source_data: list[dict]
    extraction_timestamp: float

    # Transform
    transformed_data: list[dict]
    transformation_rules: dict
    validation_errors: list[str]

    # Load
    load_status: Literal["pending", "loading", "completed", "failed"]
    records_loaded: int
    load_errors: list[str]
```

## Code Generation State

### Code Generation State

```python
class CodeGenState(TypedDict):
    """State for code generation agent."""
    task_description: str
    language: str

    # Generation
    generated_code: str
    test_cases: list[str]

    # Validation
    syntax_valid: bool
    tests_passed: bool

    # Iteration
    revision_count: int
    feedback: str
```

### Code Review State

```python
class CodeReviewState(TypedDict):
    """State for code review agent."""
    code: str
    language: str

    # Analysis
    issues: list[dict]
    suggestions: list[dict]
    complexity_score: float

    # Review
    review_comments: str
    approval_status: Literal["approved", "changes_requested", "rejected"]
```

## Error Recovery State

### Retry State

```python
class RetryState(TypedDict):
    """State with retry logic."""
    task: str

    # Execution
    result: Any | None
    error: str | None

    # Retry tracking
    attempt_count: int
    max_attempts: int
    last_error: str

    # Status
    status: Literal["pending", "running", "success", "failed"]
```

### Fallback State

```python
class FallbackState(TypedDict):
    """State with fallback strategies."""
    task: str

    # Primary approach
    primary_result: Any | None
    primary_error: str | None

    # Fallback approaches
    fallback_attempted: list[str]
    fallback_result: Any | None

    # Final
    final_result: Any
    strategy_used: str
```

## Complex Multi-Agent State

### Supervisor State

```python
class SupervisorState(MessagesState):
    """Supervisor-subagent pattern state."""

    # Routing
    next: Literal["agent1", "agent2", "agent3", "FINISH"]
    current_agent: str

    # Tracking
    agent_history: list[str]
    iteration_count: int

    # Shared context
    context: dict
```

### Router State

```python
class RouterState(MessagesState):
    """Router pattern state."""

    # Routing decision
    route: Literal["category1", "category2", "category3"]
    confidence: float

    # Execution
    result: Any
```

### Orchestrator-Worker State

```python
class OrchestratorState(TypedDict):
    """Orchestrator-worker pattern state."""
    task: str

    # Decomposition
    subtasks: list[dict]

    # Execution
    results: Annotated[list[str], extend_list]

    # Progress
    completed_count: int
    total_count: int

    # Aggregation
    final_result: str
```

## Custom Reducer Patterns

### List Extension

```python
def extend_list(left: list, right: list) -> list:
    """Extend list without duplicates."""
    return left + right

class ListState(TypedDict):
    """State with list extension."""
    items: Annotated[list[str], extend_list]
```

### Dictionary Merging

```python
def merge_dicts(left: dict, right: dict) -> dict:
    """Merge dictionaries with right overriding left."""
    return {**left, **right}

class DictState(TypedDict):
    """State with dictionary merging."""
    context: Annotated[dict, merge_dicts]
```

### Set Union

```python
def union_sets(left: set, right: set) -> set:
    """Union of sets."""
    return left | right

class SetState(TypedDict):
    """State with set union."""
    tags: Annotated[set[str], union_sets]
```

### Conditional Update

```python
def conditional_update(left: Any, right: Any | None) -> Any:
    """Update only if right is not None."""
    return right if right is not None else left

class ConditionalState(TypedDict):
    """State with conditional updates."""
    status: Annotated[str, conditional_update]
```

## State Schema Best Practices

### 1. Use TypedDict for Type Safety

```python
# ✅ Good: TypedDict provides type safety
class MyState(TypedDict):
    field: str

# ❌ Bad: Plain dict loses type information
my_state = {"field": "value"}
```

### 2. Document State Fields

```python
class DocumentedState(TypedDict):
    """Well-documented state schema."""

    # Input
    query: str  # User's search query

    # Processing
    results: list[dict]  # Search results from API

    # Output
    response: str  # Generated response
```

### 3. Use Literal for Routing Fields

```python
# ✅ Good: Literal provides type safety for routing
class GoodState(TypedDict):
    next: Literal["agent1", "agent2", "FINISH"]

# ❌ Bad: String without type constraint
class BadState(TypedDict):
    next: str  # Could be anything
```

### 4. Choose Appropriate Reducers

```python
# ✅ Good: Use MessagesState for chat history
class ChatState(MessagesState):
    pass

# ❌ Bad: No reducer, loses previous messages
messages: list
```

### 5. Include Metadata for Debugging

```python
class DebugState(TypedDict):
    """State with debugging metadata."""
    # Core fields
    task: str
    result: str

    # Debugging metadata
    start_time: float
    end_time: float
    iteration_count: int
    errors: list[str]
```

### 6. Keep State Flat When Possible

```python
# ✅ Good: Flat structure
class FlatState(TypedDict):
    user_id: str
    user_name: str
    user_email: str

# ⚠️  Consider: Nested structure (use sparingly)
class NestedState(TypedDict):
    user: dict  # Less type-safe
```

### 7. Use Optional for Nullable Fields

```python
from typing import Optional

class OptionalFieldState(TypedDict):
    required_field: str
    optional_field: Optional[str]  # Can be None
```

## Additional Resources

- LangGraph State Documentation: https://docs.langchain.com/oss/python/langgraph/graph-api
- Message Reducers: https://docs.langchain.com/oss/python/langgraph/use-graph-api
