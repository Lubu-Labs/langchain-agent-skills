# Supervisor-Subagent Pattern

The supervisor-subagent pattern uses a central coordinator (supervisor) to analyze tasks and delegate work to specialized subagents. This is a common multi-agent pattern for complex workflows.

## When to Use

Use supervisor-subagent when:
- You have multiple specialized agents with distinct capabilities
- Tasks require dynamic routing based on context
- You need centralized coordination and decision-making
- Subagents need to communicate through a central authority

## Architecture

```
User Request → Supervisor → Subagent 1
                ↓            ↓
              Subagent 2     Back to Supervisor
                ↓            ↓
              Subagent 3     Decision (continue/finish)
```

## Core Components

### 1. State Schema

The state must support message accumulation and routing decisions:

**Python:**
```python
from typing import TypedDict, Literal, Annotated
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage

class SupervisorState(TypedDict):
    """State for supervisor pattern.

    Attributes:
        messages: Chat history with add_messages reducer
        next: Next agent to route to
        current_agent: Currently active agent
        task_complete: Whether the task is complete
    """
    messages: Annotated[list[BaseMessage], add_messages]
    next: Literal["researcher", "writer", "reviewer", "FINISH"]
    current_agent: str
    task_complete: bool
```

**JavaScript:**
```typescript
import { BaseMessage } from "@langchain/core/messages";
import { Annotation } from "@langchain/langgraph";

export const SupervisorStateAnnotation = Annotation.Root({
  messages: Annotation<BaseMessage[]>({
    reducer: (left, right) => left.concat(right),
    default: () => [],
  }),
  next: Annotation<"researcher" | "writer" | "reviewer" | "FINISH">({
    reducer: (_, right) => right,
  }),
  currentAgent: Annotation<string>({
    reducer: (_, right) => right,
  }),
  taskComplete: Annotation<boolean>({
    reducer: (_, right) => right,
    default: () => false,
  }),
});
```

**Key features:**
- `messages`: Accumulates conversation history with `add_messages` reducer
- `next`: Stores routing decision (which agent or FINISH)
- `current_agent`: Tracks which agent last acted
- `task_complete`: Boolean flag for termination

### 2. Supervisor Node

The supervisor analyzes the current state and decides which subagent should act next:

**Python:**
```python
from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI

model = ChatOpenAI(model="gpt-4")

SUPERVISOR_PROMPT = """You are a supervisor managing a team of agents.

Available agents:
- researcher: Searches for information and gathers facts
- writer: Creates written content based on research
- reviewer: Reviews and improves written content

Analyze the conversation and decide which agent should act next.
If the task is complete, respond with "FINISH".

Consider:
1. What has been accomplished so far
2. What still needs to be done
3. Which agent is best suited for the next step
"""

def supervisor_node(state: SupervisorState) -> dict:
    """Supervisor decides which agent acts next."""
    messages = state["messages"]

    prompt = ChatPromptTemplate.from_messages([
        ("system", SUPERVISOR_PROMPT),
        MessagesPlaceholder(variable_name="messages"),
        ("system", "Who should act next? Respond with: researcher, writer, reviewer, or FINISH")
    ])

    chain = prompt | model
    response = chain.invoke({"messages": messages})

    # Extract routing decision from LLM response
    next_agent = response.content.strip()

    # Validate routing decision
    valid_options = ["researcher", "writer", "reviewer", "FINISH"]
    if next_agent not in valid_options:
        next_agent = "FINISH"  # Default to finish if invalid

    return {
        "next": next_agent,
        "current_agent": "supervisor"
    }
```

**JavaScript:**
```typescript
import { ChatOpenAI } from "@langchain/openai";
import { ChatPromptTemplate, MessagesPlaceholder } from "@langchain/core/prompts";

const model = new ChatOpenAI({ model: "gpt-4" });

const SUPERVISOR_PROMPT = `You are a supervisor managing a team of agents.

Available agents:
- researcher: Searches for information and gathers facts
- writer: Creates written content based on research
- reviewer: Reviews and improves written content

Analyze the conversation and decide which agent should act next.
If the task is complete, respond with "FINISH".`;

export async function supervisorNode(state: any): Promise<Partial<any>> {
  const messages = state.messages;

  const prompt = ChatPromptTemplate.fromMessages([
    ["system", SUPERVISOR_PROMPT],
    new MessagesPlaceholder("messages"),
    ["system", "Who should act next? Respond with: researcher, writer, reviewer, or FINISH"]
  ]);

  const chain = prompt.pipe(model);
  const response = await chain.invoke({ messages });

  let nextAgent = response.content.trim();

  // Validate routing decision
  const validOptions = ["researcher", "writer", "reviewer", "FINISH"];
  if (!validOptions.includes(nextAgent)) {
    nextAgent = "FINISH";
  }

  return {
    next: nextAgent,
    currentAgent: "supervisor"
  };
}
```

**Key implementation details:**
- Uses LLM to make routing decisions based on context
- Validates routing decision against valid options
- Defaults to FINISH for invalid routes
- Can be enhanced with structured output for more reliable routing

### 3. Subagent Nodes

Each subagent is a specialized node that performs specific tasks:

**Python:**
```python
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.tools.tavily_search import TavilySearchResults

# Research agent with search tool
search_tool = TavilySearchResults(max_results=3)

def researcher_node(state: SupervisorState) -> dict:
    """Research agent searches for information."""
    messages = state["messages"]

    # Get the latest user query
    latest_query = messages[-1].content

    # Perform search
    search_results = search_tool.invoke(latest_query)

    # Format results
    result_text = "Research findings:\n"
    for i, result in enumerate(search_results, 1):
        result_text += f"\n{i}. {result['content']}\n"

    return {
        "messages": [HumanMessage(content=result_text)],
        "current_agent": "researcher"
    }


# Reuse the same model defined in the supervisor section
def writer_node(state: SupervisorState) -> dict:
    """Writer creates content based on research."""
    messages = state["messages"]

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a skilled writer. Create well-written content based on the research provided."),
        MessagesPlaceholder(variable_name="messages"),
        ("system", "Write your content now:")
    ])

    chain = prompt | model
    response = chain.invoke({"messages": messages})

    return {
        "messages": [HumanMessage(content=response.content)],
        "current_agent": "writer"
    }


def reviewer_node(state: SupervisorState) -> dict:
    """Reviewer checks and improves content."""
    messages = state["messages"]

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an editor. Review the content and suggest improvements."),
        MessagesPlaceholder(variable_name="messages"),
        ("system", "Provide your review:")
    ])

    chain = prompt | model
    response = chain.invoke({"messages": messages})

    return {
        "messages": [HumanMessage(content=response.content)],
        "current_agent": "reviewer"
    }
```

**JavaScript:**
```typescript
import { HumanMessage } from "@langchain/core/messages";
import { ChatPromptTemplate, MessagesPlaceholder } from "@langchain/core/prompts";
import { TavilySearchResults } from "@langchain/community/tools/tavily_search";

const searchTool = new TavilySearchResults({ maxResults: 3 });

export async function researcherNode(state: any): Promise<Partial<any>> {
  const messages = state.messages;
  const latestQuery = messages[messages.length - 1].content;

  const searchResults = await searchTool.invoke(latestQuery);

  let resultText = "Research findings:\n";
  searchResults.forEach((result: any, i: number) => {
    resultText += `\n${i + 1}. ${result.content}\n`;
  });

  return {
    messages: [new HumanMessage(resultText)],
    currentAgent: "researcher"
  };
}

export async function writerNode(state: any): Promise<Partial<any>> {
  const messages = state.messages;

  const prompt = ChatPromptTemplate.fromMessages([
    ["system", "You are a skilled writer. Create well-written content based on the research provided."],
    new MessagesPlaceholder("messages"),
    ["system", "Write your content now:"]
  ]);

  const chain = prompt.pipe(model);
  const response = await chain.invoke({ messages });

  return {
    messages: [new HumanMessage(response.content)],
    currentAgent: "writer"
  };
}
```

### 4. Graph Assembly

Wire together supervisor and subagents with proper routing:

**Python:**
```python
from langgraph.graph import StateGraph, END

def create_supervisor_graph() -> StateGraph:
    """Assemble supervisor-subagent graph."""
    graph = StateGraph(SupervisorState)

    # Add nodes
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("researcher", researcher_node)
    graph.add_node("writer", writer_node)
    graph.add_node("reviewer", reviewer_node)

    # Set entry point
    graph.set_entry_point("supervisor")

    # Add conditional edges from supervisor
    graph.add_conditional_edges(
        "supervisor",
        lambda state: state["next"],
        {
            "researcher": "researcher",
            "writer": "writer",
            "reviewer": "reviewer",
            "FINISH": END
        }
    )

    # All subagents route back to supervisor
    graph.add_edge("researcher", "supervisor")
    graph.add_edge("writer", "supervisor")
    graph.add_edge("reviewer", "supervisor")

    return graph.compile()
```

**JavaScript:**
```typescript
import { StateGraph, START, END } from "@langchain/langgraph";

export function createSupervisorGraph() {
  const graph = new StateGraph(SupervisorStateAnnotation)
    .addNode("supervisor", supervisorNode)
    .addNode("researcher", researcherNode)
    .addNode("writer", writerNode)
    .addNode("reviewer", reviewerNode)

    .addEdge(START, "supervisor")

    .addConditionalEdges(
      "supervisor",
      (state) => state.next,
      {
        researcher: "researcher",
        writer: "writer",
        reviewer: "reviewer",
        FINISH: END
      }
    )

    .addEdge("researcher", "supervisor")
    .addEdge("writer", "supervisor")
    .addEdge("reviewer", "supervisor");

  return graph.compile();
}
```

## Best Practices

### 1. Clear Agent Responsibilities

Define clear, non-overlapping responsibilities for each subagent:

```python
SUPERVISOR_PROMPT = """Available agents and their capabilities:

- researcher:
  * Searches web for current information
  * Gathers facts, statistics, citations
  * Provides source references

- writer:
  * Creates articles, blog posts, documentation
  * Organizes information coherently
  * Applies appropriate tone and style

- reviewer:
  * Checks factual accuracy
  * Improves clarity and grammar
  * Ensures consistency with requirements
"""
```

### 2. Structured Routing

Use structured output for more reliable routing:

```python
from pydantic import BaseModel
from typing import Literal

class RouteDecision(BaseModel):
    """Structured routing decision."""
    next_agent: Literal["researcher", "writer", "reviewer", "FINISH"]
    reasoning: str

def supervisor_node(state: SupervisorState) -> dict:
    """Supervisor with structured output."""
    messages = state["messages"]

    # Use structured output for routing
    structured_model = model.with_structured_output(RouteDecision)

    decision = structured_model.invoke([
        SystemMessage(content=SUPERVISOR_PROMPT),
        *messages,
        SystemMessage(content="Decide which agent should act next.")
    ])

    return {
        "next": decision.next_agent,
        "current_agent": "supervisor"
    }
```

### 3. Loop Prevention

Add loop detection to prevent infinite cycles:

```python
def supervisor_node(state: SupervisorState) -> dict:
    """Supervisor with loop detection."""
    messages = state["messages"]

    # Check for repeated patterns
    recent_agents = [
        msg.name for msg in messages[-6:]
        if hasattr(msg, 'name')
    ]

    # If same agent called 3+ times in a row, force different route
    if len(recent_agents) >= 3 and len(set(recent_agents[-3:])) == 1:
        return {
            "next": "FINISH",
            "current_agent": "supervisor",
            "messages": [SystemMessage(content="Detected loop, completing task.")]
        }

    # Normal routing logic...
```

### 4. Context Management

Keep context size manageable by summarizing history:

```python
def supervisor_node(state: SupervisorState) -> dict:
    """Supervisor with context summarization."""
    messages = state["messages"]

    # If context gets too long, summarize older messages
    if len(messages) > 20:
        # Summarize first N messages
        summary_prompt = f"Summarize these messages concisely: {messages[:10]}"
        summary = model.invoke(summary_prompt).content

        # Keep recent messages + summary
        messages = [
            SystemMessage(content=f"Previous context: {summary}"),
            *messages[10:]
        ]

    # Continue with routing...
```

## Common Variations

### Parallel Execution

Execute multiple subagents in parallel:

```python
from langgraph.graph import StateGraph, END

def create_parallel_supervisor_graph():
    """Supervisor that can dispatch multiple agents in parallel."""
    graph = StateGraph(SupervisorState)

    graph.add_node("supervisor", supervisor_node)
    graph.add_node("researcher", researcher_node)
    graph.add_node("fact_checker", fact_checker_node)

    # Both can run in parallel
    graph.add_edge("supervisor", "researcher")
    graph.add_edge("supervisor", "fact_checker")

    # Use Send for parallel execution
    # (Requires modified state to track parallel results)

    return graph.compile()
```

### Hierarchical Supervision

Nest supervisors for complex workflows:

```python
# Top-level supervisor
def top_supervisor_node(state):
    """Delegates to sub-supervisors."""
    # Route to research_team or writing_team
    pass

# Research team supervisor
def research_supervisor_node(state):
    """Manages research subagents."""
    # Route to web_searcher, document_reader, etc.
    pass
```

## Debugging Tips

### 1. Trace Agent Decisions

Log supervisor routing decisions:

```python
def supervisor_node(state: SupervisorState) -> dict:
    """Supervisor with decision logging."""
    decision = get_routing_decision(state)

    print(f"Supervisor decision: {decision}")
    print(f"Current state: {len(state['messages'])} messages")

    return {"next": decision, "current_agent": "supervisor"}
```

### 2. Visualize Flow

Use LangSmith tracing to visualize the agent flow:

```python
import os
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_API_KEY"] = "<your-api-key>"

result = graph.invoke({"messages": [HumanMessage(content="Research AI")]})
```

### 3. Test Routing Logic

Unit test supervisor routing decisions:

```python
def test_supervisor_routing():
    """Test supervisor routes correctly."""
    # Test state with research needed
    state = {
        "messages": [HumanMessage(content="Tell me about AI")],
        "next": "",
        "current_agent": "",
        "task_complete": False
    }

    result = supervisor_node(state)
    assert result["next"] == "researcher"

    # Test state with writing needed
    state["messages"].append(HumanMessage(content="Research results: ..."))
    result = supervisor_node(state)
    assert result["next"] == "writer"
```

## Performance Considerations

### Token Usage

Supervisor patterns use more tokens due to:
- LLM calls for routing decisions
- Full context passed to supervisor each iteration
- Message accumulation over time

Mitigation strategies:
- Use cheaper models for routing (e.g., GPT-4o-mini)
- Summarize context when it grows large
- Cache supervisor prompts
- Use structured output to reduce parsing

### Latency

Sequential routing adds latency:
- Each supervisor decision requires an LLM call
- Subagents wait for supervisor decisions
- No parallel execution by default

Optimizations:
- Use faster models for supervisor
- Implement parallel execution where possible
- Cache common routing patterns
- Consider router pattern for simpler routing

## Migration Path

Converting a simple agent to supervisor pattern:

**Before (simple agent):**
```python
def agent(state):
    # Do research
    # Do writing
    # Do review
    return result
```

**After (supervisor pattern):**
```python
# Split into specialized nodes
def researcher(state): ...
def writer(state): ...
def reviewer(state): ...

# Add supervisor for coordination
def supervisor(state):
    # Decide which specialized agent to call
    return routing_decision

# Assemble graph
graph = create_supervisor_graph()
```

## When Not to Use

Avoid supervisor pattern when:
- **Simple sequential flow**: Use basic graph edges instead
- **Deterministic routing**: Use router pattern instead
- **No coordination needed**: Use parallel execution
- **High latency sensitivity**: Consider simpler patterns

## Additional Resources

- Multi-Agent Tutorials (Python): https://docs.langchain.com/oss/python/langchain/multi-agent/subagents-personal-assistant
- Multi-Agent Tutorials (JavaScript): https://docs.langchain.com/oss/javascript/langchain/multi-agent/subagents-personal-assistant
- LangGraph Graph API: https://docs.langchain.com/oss/python/langgraph/graph-api
- State Management Guide: See state-management-patterns.md
