# Orchestrator-Worker Pattern

The orchestrator-worker pattern distributes independent tasks across multiple worker agents that execute in parallel, then aggregates their results. Best for parallelizable workloads.

## When to Use

- Tasks can be decomposed into independent subtasks
- Subtasks can execute simultaneously
- Results need aggregation or synthesis
- Performance/throughput is critical

## Architecture

```
Request → Orchestrator (split) → Worker 1 ┐
                               → Worker 2 ├→ Aggregator → Result
                               → Worker 3 ┘
```

## Implementation

### Python with Send API

```python
from typing import TypedDict, Annotated
import operator
from langgraph.graph import StateGraph, START, END, Send

class OrchestratorState(TypedDict):
    """State for orchestrator pattern."""
    task: str
    subtasks: list[dict]
    results: Annotated[list[dict], operator.add]
    final_result: str

class WorkerState(TypedDict):
    """State for individual worker."""
    subtask: dict

def orchestrator_node(state: OrchestratorState) -> dict:
    """Split task into subtasks and dispatch to workers."""
    task = state["task"]

    # Decompose into subtasks (could use LLM)
    subtasks = [
        {"id": 1, "query": "Research topic A"},
        {"id": 2, "query": "Research topic B"},
        {"id": 3, "query": "Research topic C"}
    ]

    return {"subtasks": subtasks}

def worker_node(state: WorkerState) -> dict:
    """Process individual subtask."""
    subtask = state["subtask"]

    # Process subtask
    result = f"Result for {subtask['query']}"

    return {"results": [{"result": result, "status": "success"}]}

def aggregator_node(state: OrchestratorState) -> dict:
    """Aggregate worker results."""
    results = state["results"]

    # Combine results (could use LLM for synthesis)
    final_result = "\n".join(r["result"] for r in results)

    return {"final_result": final_result}

# Graph assembly with parallel execution
def create_orchestrator_graph():
    """Build orchestrator-worker graph with parallel execution."""

    def dispatch_workers(state: OrchestratorState):
        """Create Send objects for parallel worker execution."""
        return [
            Send("worker", {"subtask": subtask})
            for subtask in state["subtasks"]
        ]

    graph = StateGraph(OrchestratorState)

    graph.add_node("orchestrator", orchestrator_node)
    graph.add_node("worker", worker_node)
    graph.add_node("aggregator", aggregator_node)

    graph.add_edge(START, "orchestrator")

    # Dispatch to workers in parallel
    graph.add_conditional_edges(
        "orchestrator",
        dispatch_workers
    )

    # Workers send results to aggregator
    graph.add_edge("worker", "aggregator")
    graph.add_edge("aggregator", END)

    return graph.compile()
```

### JavaScript

```typescript
import { StateGraph, START, END, Send } from "@langchain/langgraph";
import { Annotation } from "@langchain/langgraph";

const OrchestratorStateAnnotation = Annotation.Root({
  task: Annotation<string>(),
  subtasks: Annotation<Array<any>>({
    reducer: (_, right) => right,
    default: () => [],
  }),
  results: Annotation<Array<{ result: string; status?: string }>>({
    reducer: (left, right) => left.concat(right),
    default: () => [],
  }),
  finalResult: Annotation<string>(),
});

function orchestratorNode(state: any): Partial<any> {
  const task = state.task;

  const subtasks = [
    { id: 1, query: "Research topic A" },
    { id: 2, query: "Research topic B" },
    { id: 3, query: "Research topic C" }
  ];

  return { subtasks };
}

function workerNode(state: any): Partial<any> {
  const subtask = state.subtask;
  const result = `Result for ${subtask.query}`;

  return { results: [{ result, status: "success" }] };
}

function aggregatorNode(state: any): Partial<any> {
  const results = state.results;
  const finalResult = results.map((r: any) => r.result).join("\n");

  return { finalResult };
}

export function createOrchestratorGraph() {
  function dispatchWorkers(state: any) {
    return state.subtasks.map((subtask: any) =>
      new Send("worker", { subtask })
    );
  }

  const graph = new StateGraph(OrchestratorStateAnnotation)
    .addNode("orchestrator", orchestratorNode)
    .addNode("worker", workerNode)
    .addNode("aggregator", aggregatorNode)

    .addEdge(START, "orchestrator")

    .addConditionalEdges("orchestrator", dispatchWorkers)

    .addEdge("worker", "aggregator")
    .addEdge("aggregator", END);

  return graph.compile();
}
```

## Task Decomposition Strategies

### LLM-Based Decomposition

```python
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

class TaskDecomposition(BaseModel):
    """Structured task decomposition."""
    subtasks: list[str]
    dependencies: dict[str, list[str]]

def decompose_task(task: str) -> TaskDecomposition:
    """Use LLM to break down task."""
    model = ChatOpenAI(model="gpt-4")
    structured = model.with_structured_output(TaskDecomposition)

    prompt = f"""Break down this task into independent subtasks that can run in parallel:
    Task: {task}

    Identify:
    1. Subtasks that can execute simultaneously
    2. Any dependencies between subtasks
    """

    return structured.invoke(prompt)
```

### Rule-Based Decomposition

```python
def decompose_research_task(topic: str) -> list[dict]:
    """Decompose research task into subtasks."""
    return [
        {"type": "web_search", "query": f"{topic} overview"},
        {"type": "academic_search", "query": f"{topic} research papers"},
        {"type": "news_search", "query": f"{topic} recent news"},
        {"type": "expert_search", "query": f"{topic} experts"}
    ]
```

## Result Aggregation

### Simple Concatenation

```python
def simple_aggregator(results: list[dict]) -> str:
    """Concatenate results."""
    return "\n\n".join(f"## Result {i+1}\n{r['result']}" for i, r in enumerate(results))
```

### LLM-Based Synthesis

```python
def llm_aggregator(results: list[dict], original_task: str) -> str:
    """Use LLM to synthesize results."""
    model = ChatOpenAI(model="gpt-4")

    prompt = f"""Synthesize these research results into a coherent answer:

    Original question: {original_task}

    Results:
    {chr(10).join(f"{i+1}. {r['result']}" for i, r in enumerate(results))}

    Provide a comprehensive, well-organized answer.
    """

    return model.invoke(prompt).content
```

### Structured Aggregation

```python
from pydantic import BaseModel

class AggregatedResult(BaseModel):
    """Structured aggregation."""
    summary: str
    key_findings: list[str]
    sources: list[str]
    confidence: float

def structured_aggregator(results: list[dict]) -> AggregatedResult:
    """Aggregate into structured format."""
    model = ChatOpenAI(model="gpt-4")
    structured = model.with_structured_output(AggregatedResult)

    return structured.invoke({
        "results": results,
        "instruction": "Synthesize these results"
    })
```

## Handling Worker Failures

### Retry Strategy

```python
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10)
)
def resilient_worker(state: WorkerState) -> dict:
    """Worker with automatic retry."""
    try:
        result = process_subtask(state["subtask"])
        return {"results": [{"result": result, "status": "success"}]}
    except Exception as e:
        return {"results": [{"result": str(e), "status": "failed"}]}
```

### Fallback Handling

```python
def robust_aggregator(state: OrchestratorState) -> dict:
    """Handle failed workers."""
    results = state.get("results", [])

    # Filter out failed results
    successful_results = [
        r for r in results
        if isinstance(r, dict) and r.get("status") == "success"
    ]

    if len(successful_results) == 0:
        return {"final_result": "All workers failed", "status": "error"}

    # Aggregate successful results
    final_result = aggregate_results([r["result"] for r in successful_results])

    # Include warning about failures
    if len(successful_results) < len(results):
        final_result += f"\n\nWarning: {len(results) - len(successful_results)} workers failed"

    return {"final_result": final_result, "status": "partial_success"}
```

## Best Practices

### 1. Resource Management

Limit concurrent workers to avoid overwhelming resources:

```python
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import RetryPolicy

def create_orchestrator_graph():
    graph = StateGraph(OrchestratorState)

    # Configure with resource limits
    graph.add_node(
        "worker",
        worker_node,
        retry_policy=RetryPolicy(max_attempts=3)
    )

    # Add checkpointer for state persistence
    checkpointer = MemorySaver()

    return graph.compile(checkpointer=checkpointer)
```

### 2. Progress Tracking

Track worker progress in state:

```python
from typing import TypedDict, Annotated
import operator

class OrchestratorState(TypedDict):
    task: str
    subtasks: list[dict]
    completed: int
    total: int
    results: Annotated[list[dict], operator.add]

def track_progress(state: OrchestratorState) -> dict:
    """Update progress tracking."""
    return {
        "completed": len(state.get("results", [])),
        "total": len(state.get("subtasks", []))
    }
```

### 3. Timeout Handling

Set timeouts for workers:

```python
import asyncio
from concurrent.futures import TimeoutError

async def worker_with_timeout(state: WorkerState, timeout: int = 30) -> dict:
    """Worker with timeout."""
    try:
        result = await asyncio.wait_for(
            process_subtask_async(state["subtask"]),
            timeout=timeout
        )
        return {"results": [{"result": result, "status": "success"}]}
    except TimeoutError:
        return {"results": [{"result": "Task timed out", "status": "timeout"}]}
```

## When to Use vs Alternatives

**Use orchestrator-worker when:**
- Tasks are independent and parallelizable
- Throughput is important
- Results need synthesis

**Use supervisor pattern instead when:**
- Tasks are sequential or dependent
- Agents need to collaborate
- Dynamic routing is needed

**Use map-reduce pattern instead when:**
- Processing large datasets
- Uniform processing across items
- No complex aggregation logic needed

## Performance Considerations

**Advantages:**
- Parallel execution reduces total time
- Scales horizontally
- High throughput

**Challenges:**
- All workers consume resources simultaneously
- Aggregation waits for slowest worker
- Memory usage grows with worker count
- Cost scales with parallelism

**Optimizations:**
- Batch workers if too many subtasks
- Stream results to aggregator incrementally
- Use cheaper models for workers
- Implement worker pooling

## Testing

```python
def test_orchestrator_pattern():
    """Test orchestrator-worker pattern."""
    graph = create_orchestrator_graph()

    # Test with sample task
    result = graph.invoke({
        "task": "Research artificial intelligence"
    })

    assert "final_result" in result
    assert len(result["results"]) == len(result["subtasks"])

def test_worker_isolation():
    """Ensure workers don't interfere."""
    # Workers should process independently
    state1 = {"subtask": {"id": 1, "query": "Task 1"}}
    state2 = {"subtask": {"id": 2, "query": "Task 2"}}

    result1 = worker_node(state1)
    result2 = worker_node(state2)

    assert result1["results"][0]["result"] != result2["results"][0]["result"]
```

## Additional Resources

- LangGraph Map-Reduce: https://docs.langchain.com/oss/python/langgraph/use-graph-api#map-reduce-and-the-send-api
- Send API Documentation: https://reference.langchain.com/python/langgraph/types/?h=send#langgraph.types.Send
