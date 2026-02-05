# Handoffs Pattern

Agent handoffs allow one agent to transfer control to another agent, passing context and partial results. This creates collaborative workflows where agents work sequentially on the same task.

## When to Use

- Tasks require multiple specialized skills in sequence
- Each agent builds on previous agent's work
- Context must be preserved across handoffs
- Clear transfer points exist in the workflow

## Basic Handoff Implementation

### Python

```python
from typing import TypedDict, Literal
from langgraph.graph import StateGraph, START, END

class HandoffState(TypedDict):
    """State with handoff support."""
    messages: list
    current_agent: str
    next_agent: Literal["researcher", "writer", "editor", "FINISH"]
    context: dict  # Preserved across handoffs

def researcher_agent(state: HandoffState) -> dict:
    """Research agent that hands off to writer."""
    # Do research work
    research_results = perform_research(state["messages"])

    # Decide if ready to hand off
    if research_is_complete(research_results):
        return {
            "messages": [{"role": "assistant", "content": research_results}],
            "current_agent": "researcher",
            "next_agent": "writer",
            "context": {"research_data": research_results}
        }
    else:
        # Continue researching
        return {"next_agent": "researcher"}

def writer_agent(state: HandoffState) -> dict:
    """Writer receives handoff from researcher."""
    # Access context from previous agent
    research_data = state["context"].get("research_data", "")

    # Create content based on research
    content = write_content(research_data)

    # Hand off to editor
    return {
        "messages": [{"role": "assistant", "content": content}],
        "current_agent": "writer",
        "next_agent": "editor",
        "context": {"draft": content, "research_data": research_data}
    }

def editor_agent(state: HandoffState) -> dict:
    """Editor reviews and finalizes."""
    draft = state["context"].get("draft", "")

    # Review and edit
    final_content = edit_content(draft)

    return {
        "messages": [{"role": "assistant", "content": final_content}],
        "current_agent": "editor",
        "next_agent": "FINISH"
    }

# Build graph
def create_handoff_graph():
    graph = StateGraph(HandoffState)

    graph.add_node("researcher", researcher_agent)
    graph.add_node("writer", writer_agent)
    graph.add_node("editor", editor_agent)

    graph.add_edge(START, "researcher")

    # Routing based on next_agent field
    graph.add_conditional_edges(
        "researcher",
        lambda s: s["next_agent"],
        {
            "researcher": "researcher",  # Continue researching
            "writer": "writer",
            "FINISH": END
        }
    )

    graph.add_conditional_edges(
        "writer",
        lambda s: s["next_agent"],
        {
            "editor": "editor",
            "researcher": "researcher",  # Can hand back
            "FINISH": END
        }
    )

    graph.add_conditional_edges(
        "editor",
        lambda s: s["next_agent"],
        {
            "writer": "writer",  # Request revisions
            "FINISH": END
        }
    )

    return graph.compile()
```

### TypeScript

```typescript
import { StateGraph, START, END } from "@langchain/langgraph";
import { Annotation } from "@langchain/langgraph";

const HandoffStateAnnotation = Annotation.Root({
  messages: Annotation<any[]>({
    reducer: (left, right) => left.concat(right),
    default: () => [],
  }),
  currentAgent: Annotation<string>(),
  nextAgent: Annotation<"researcher" | "writer" | "editor" | "FINISH">(),
  context: Annotation<Record<string, any>>({
    reducer: (left, right) => ({ ...left, ...right }),
    default: () => ({}),
  }),
});

async function researcherAgent(state: any): Promise<Partial<any>> {
  const researchResults = await performResearch(state.messages);

  if (isResearchComplete(researchResults)) {
    return {
      messages: [{ role: "assistant", content: researchResults }],
      currentAgent: "researcher",
      nextAgent: "writer",
      context: { researchData: researchResults }
    };
  }

  return { nextAgent: "researcher" };
}

export function createHandoffGraph() {
  const graph = new StateGraph(HandoffStateAnnotation)
    .addNode("researcher", researcherAgent)
    .addNode("writer", writerAgent)
    .addNode("editor", editorAgent)

    .addEdge(START, "researcher")

    .addConditionalEdges(
      "researcher",
      (state) => state.nextAgent,
      {
        researcher: "researcher",
        writer: "writer",
        FINISH: END
      }
    )

    .addConditionalEdges(
      "writer",
      (state) => state.nextAgent,
      {
        editor: "editor",
        researcher: "researcher",
        FINISH: END
      }
    )

    .addConditionalEdges(
      "editor",
      (state) => state.nextAgent,
      {
        writer: "writer",
        FINISH: END
      }
    );

  return graph.compile();
}
```

## Handoff Strategies

### 1. Explicit Handoff

Agent explicitly declares handoff:

```python
def agent_with_explicit_handoff(state: HandoffState) -> dict:
    """Agent explicitly decides when to hand off."""
    result = process_task(state)

    # Check completion criteria
    if is_task_complete(result):
        return {
            "next_agent": "next_specialist",
            "context": {"completed_work": result}
        }

    return {"next_agent": state["current_agent"]}  # Continue
```

### 2. Conditional Handoff

Handoff based on conditions:

```python
def conditional_handoff(state: HandoffState) -> dict:
    """Hand off based on state conditions."""
    if state["context"].get("needs_review"):
        return {"next_agent": "reviewer"}
    elif state["context"].get("needs_research"):
        return {"next_agent": "researcher"}
    else:
        return {"next_agent": "FINISH"}
```

### 3. Circular Handoff

Agents can hand back to previous agents:

```python
def editor_with_feedback(state: HandoffState) -> dict:
    """Editor can send back for revisions."""
    draft = state["context"]["draft"]

    issues = check_quality(draft)

    if issues:
        return {
            "next_agent": "writer",
            "context": {"revision_notes": issues}
        }

    return {"next_agent": "FINISH"}
```

## Context Preservation

### Structured Context

```python
from pydantic import BaseModel

class WorkflowContext(BaseModel):
    """Structured context for handoffs."""
    research_findings: list[str] = []
    draft_versions: list[str] = []
    review_notes: list[str] = []
    metadata: dict = {}

class HandoffState(TypedDict):
    messages: list
    current_agent: str
    next_agent: str
    context: WorkflowContext

def researcher_with_context(state: HandoffState) -> dict:
    """Researcher adds to structured context."""
    context = state["context"]
    findings = do_research()

    context.research_findings.extend(findings)
    context.metadata["research_complete"] = True

    return {
        "next_agent": "writer",
        "context": context
    }
```

### Context Summarization

Prevent context from growing too large:

```python
def summarize_context(context: dict) -> dict:
    """Summarize context before handoff."""
    if len(context.get("messages", [])) > 10:
        # Summarize old messages
        summary = create_summary(context["messages"][:5])
        context["messages"] = [
            {"role": "system", "content": f"Previous context: {summary}"},
            *context["messages"][5:]
        ]

    return context

def agent_with_summarization(state: HandoffState) -> dict:
    """Agent that summarizes context before handoff."""
    result = process_task(state)

    # Summarize before handing off
    context = summarize_context(state["context"])
    context["latest_result"] = result

    return {
        "next_agent": "next_agent",
        "context": context
    }
```

## Best Practices

### 1. Clear Handoff Criteria

Define explicit criteria for handoffs:

```python
def should_handoff_to_writer(state: HandoffState) -> bool:
    """Check if ready to hand off to writer."""
    context = state["context"]

    return (
        context.get("research_complete", False) and
        len(context.get("sources", [])) >= 3 and
        context.get("quality_check_passed", False)
    )

def researcher_agent(state: HandoffState) -> dict:
    if should_handoff_to_writer(state):
        return {"next_agent": "writer"}
    return {"next_agent": "researcher"}
```

### 2. Handoff Validation

Validate state before handoff:

```python
def validate_handoff(from_agent: str, to_agent: str, context: dict) -> bool:
    """Validate handoff is appropriate."""
    required_fields = {
        "writer": ["research_data"],
        "editor": ["draft"],
        "reviewer": ["draft", "research_data"]
    }

    if to_agent in required_fields:
        for field in required_fields[to_agent]:
            if field not in context:
                raise ValueError(f"Handoff to {to_agent} missing required field: {field}")

    return True

def safe_handoff_agent(state: HandoffState) -> dict:
    """Agent with handoff validation."""
    next_agent = determine_next_agent(state)

    # Validate before handoff
    validate_handoff(
        state["current_agent"],
        next_agent,
        state["context"]
    )

    return {"next_agent": next_agent}
```

### 3. Loop Prevention

Prevent infinite handoff loops:

```python
def track_handoffs(state: HandoffState) -> dict:
    """Track handoff history to prevent loops."""
    history = state["context"].get("handoff_history", [])
    history.append(state["current_agent"])

    # Check for loops (same agent visited 3 times)
    from collections import Counter
    counts = Counter(history)

    if any(count >= 3 for count in counts.values()):
        # Force completion
        return {"next_agent": "FINISH", "messages": ["Loop detected, completing task"]}

    state["context"]["handoff_history"] = history
    return determine_next_agent_normal(state)
```

## Handoff vs Supervisor

**Handoffs:**
- Direct agent-to-agent transfer
- Agents decide when to hand off
- More autonomous agents
- Simpler coordination

**Supervisor:**
- Central coordinator
- Supervisor decides routing
- More control
- Complex coordination logic

**When to use handoffs:**
- Linear or predictable workflow
- Agents have clear completion criteria
- Less coordination overhead needed

**When to use supervisor:**
- Dynamic routing needed
- Complex decision logic
- Centralized control desired

## Testing Handoffs

```python
def test_handoff_sequence():
    """Test handoff sequence works correctly."""
    graph = create_handoff_graph()

    result = graph.invoke({
        "messages": [{"role": "user", "content": "Write about AI"}],
        "current_agent": "",
        "next_agent": "researcher",
        "context": {}
    })

    # Verify expected end state
    assert result["current_agent"] == "editor"
    assert result["next_agent"] == "FINISH"

def test_handoff_context():
    """Test context preservation across handoffs."""
    graph = create_handoff_graph()

    result = graph.invoke({
        "messages": [{"role": "user", "content": "Research topic"}],
        "current_agent": "",
        "next_agent": "researcher",
        "context": {}
    })

    # Verify context preserved
    assert "research_data" in result["context"]
    assert "draft" in result["context"]
```

## Additional Resources

- LangGraph Human-in-the-Loop: https://docs.langchain.com/oss/python/langchain/human-in-the-loop
- Conditional Edges: https://docs.langchain.com/oss/python/langgraph/graph-api
