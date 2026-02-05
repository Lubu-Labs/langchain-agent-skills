# Router Pattern

The router pattern uses deterministic or simple conditional logic to route requests to specialized agents without a central supervisor. Routes are decided once at the beginning based on the initial request.

## When to Use

Use router pattern when:
- Routing decision can be made from the initial request
- You don't need coordination between agents
- Each request maps to exactly one agent
- Lower latency is important

## Architecture

```
User Request → Router → Agent A → END
                   ├→ Agent B → END
                   └→ Agent C → END
```

Key difference from supervisor: **one-time routing, no feedback loop**.

## Implementation

### Python

```python
from typing import TypedDict, Literal
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import BaseMessage, HumanMessage

class RouterState(TypedDict):
    """State for router pattern."""
    messages: list[BaseMessage]
    route: Literal["sales", "support", "billing"]

def router_node(state: RouterState) -> dict:
    """Route based on initial message content."""
    query = state["messages"][0].content.lower()

    # Simple keyword-based routing
    if any(word in query for word in ["buy", "purchase", "price"]):
        route = "sales"
    elif any(word in query for word in ["help", "problem", "issue"]):
        route = "support"
    elif any(word in query for word in ["invoice", "payment", "billing"]):
        route = "billing"
    else:
        route = "support"  # default

    return {"route": route}

# Specialized agent nodes
def sales_agent(state: RouterState) -> dict:
    """Handle sales queries."""
    # Process sales query...
    return {"messages": [HumanMessage(content="Sales response...")]}

def support_agent(state: RouterState) -> dict:
    """Handle support queries."""
    # Process support query...
    return {"messages": [HumanMessage(content="Support response...")]}

def billing_agent(state: RouterState) -> dict:
    """Handle billing queries."""
    # Process billing query...
    return {"messages": [HumanMessage(content="Billing response...")]}

# Build graph
def create_router_graph():
    """Assemble router graph."""
    graph = StateGraph(RouterState)

    # Add nodes
    graph.add_node("router", router_node)
    graph.add_node("sales", sales_agent)
    graph.add_node("support", support_agent)
    graph.add_node("billing", billing_agent)

    # Entry point
    graph.add_edge(START, "router")

    # Conditional routing from router to agents
    graph.add_conditional_edges(
        "router",
        lambda state: state["route"],
        {
            "sales": "sales",
            "support": "support",
            "billing": "billing"
        }
    )

    # All agents go to END
    graph.add_edge("sales", END)
    graph.add_edge("support", END)
    graph.add_edge("billing", END)

    return graph.compile()
```

### TypeScript

```typescript
import { StateGraph, START, END, Annotation, messagesStateReducer } from "@langchain/langgraph";
import { BaseMessage } from "@langchain/core/messages";

const RouterStateAnnotation = Annotation.Root({
  messages: Annotation<BaseMessage[]>({
    reducer: messagesStateReducer,
    default: () => [],
  }),
  route: Annotation<"sales" | "support" | "billing">({
    reducer: (_, right) => right,
  }),
});

function routerNode(state: any): Partial<any> {
  const query = state.messages[0].content.toLowerCase();

  let route: "sales" | "support" | "billing";

  if (["buy", "purchase", "price"].some(word => query.includes(word))) {
    route = "sales";
  } else if (["help", "problem", "issue"].some(word => query.includes(word))) {
    route = "support";
  } else if (["invoice", "payment", "billing"].some(word => query.includes(word))) {
    route = "billing";
  } else {
    route = "support";
  }

  return { route };
}

export function createRouterGraph() {
  const graph = new StateGraph(RouterStateAnnotation)
    .addNode("router", routerNode)
    .addNode("sales", salesAgent)
    .addNode("support", supportAgent)
    .addNode("billing", billingAgent)

    .addEdge(START, "router")

    .addConditionalEdges(
      "router",
      (state) => state.route,
      {
        sales: "sales",
        support: "support",
        billing: "billing"
      }
    )

    .addEdge("sales", END)
    .addEdge("support", END)
    .addEdge("billing", END);

  return graph.compile();
}
```

## Routing Strategies

### 1. Keyword-Based

Simple string matching:

```python
def keyword_router(query: str) -> str:
    """Route based on keywords."""
    keywords_map = {
        "sales": ["buy", "purchase", "price", "demo"],
        "support": ["help", "problem", "error", "broken"],
        "billing": ["invoice", "payment", "refund", "charge"]
    }

    query_lower = query.lower()
    for category, keywords in keywords_map.items():
        if any(kw in query_lower for kw in keywords):
            return category

    return "support"  # default
```

### 2. LLM-Based Classification

Use LLM for semantic routing:

```python
from langchain_openai import ChatOpenAI
from pydantic import BaseModel
from typing import Literal

class RouteClassification(BaseModel):
    """Routing classification."""
    category: Literal["sales", "support", "billing"]
    confidence: float

def llm_router(query: str) -> str:
    """Route using LLM classification."""
    model = ChatOpenAI(model="gpt-4o-mini")
    structured = model.with_structured_output(RouteClassification)

    result = structured.invoke(f"""Classify this query into sales, support, or billing:
    Query: {query}
    """)

    return result.category
```

### 3. Semantic Similarity

Use embeddings for semantic routing:

```python
from langchain_openai import OpenAIEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

embeddings = OpenAIEmbeddings()

# Define category examples
CATEGORY_EXAMPLES = {
    "sales": "I want to buy your product",
    "support": "I need help with a technical issue",
    "billing": "Question about my invoice"
}

# Pre-compute category embeddings
category_embeddings = {
    cat: embeddings.embed_query(example)
    for cat, example in CATEGORY_EXAMPLES.items()
}

def semantic_router(query: str) -> str:
    """Route based on semantic similarity."""
    query_embedding = embeddings.embed_query(query)

    # Compute similarities
    similarities = {
        cat: cosine_similarity([query_embedding], [emb])[0][0]
        for cat, emb in category_embeddings.items()
    }

    # Return category with highest similarity
    return max(similarities, key=similarities.get)
```

### 4. Intent Classification Model

Use a zero-shot classifier for routing:

```python
from transformers import pipeline

classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

def model_router(query: str) -> str:
    """Route using zero-shot classification."""
    result = classifier(query, candidate_labels=["sales", "support", "billing"])
    return result["labels"][0]
```

## Best Practices

### 1. Default Route

Always provide a default route for unclassified requests:

```python
def router_node(state: RouterState) -> dict:
    """Router with safe default."""
    route = classify_query(state["messages"][0].content)

    # Validate route
    valid_routes = ["sales", "support", "billing"]
    if route not in valid_routes:
        route = "support"  # Safe default

    return {"route": route}
```

### 2. Confidence Thresholds

Use confidence scores to handle ambiguous cases:

```python
def router_with_confidence(state: RouterState) -> dict:
    """Router with confidence checking."""
    query = state["messages"][0].content
    route, confidence = classify_with_confidence(query)

    # If low confidence, route to human or general agent
    if confidence < 0.7:
        route = "general_agent"

    return {"route": route}
```

### 3. Multi-Label Routing

Handle queries that match multiple categories:

```python
def multi_label_router(state: RouterState) -> dict:
    """Handle multi-category queries."""
    query = state["messages"][0].content
    categories = get_all_matching_categories(query)

    if len(categories) > 1:
        # Route to specialized multi-category handler
        return {"route": "multi_category_agent", "categories": categories}
    elif len(categories) == 1:
        return {"route": categories[0]}
    else:
        return {"route": "support"}
```

## When to Upgrade to Supervisor

Consider switching to supervisor pattern when:
- Agents need to collaborate or hand off to each other
- Initial routing isn't sufficient for complex queries
- You need dynamic re-routing based on intermediate results
- Context needs to be shared across multiple agents

Migration example:

**Router (before):**
```
Query → Router → Agent → END
```

**Supervisor (after):**
```
Query → Supervisor → Agent1 → Supervisor → Agent2 → END
```

## Performance Characteristics

**Advantages:**
- Lower latency (one routing decision)
- Simpler state management
- Lower token usage
- Easier to debug

**Disadvantages:**
- No agent collaboration
- Can't handle complex multi-step workflows
- Initial routing decision is final
- Limited flexibility

## Testing

Test routing logic separately from agents:

```python
def test_router_logic():
    """Test routing decisions."""
    test_cases = [
        ("I want to buy a product", "sales"),
        ("Help! My app crashed", "support"),
        ("Where's my invoice?", "billing"),
        ("General question", "support"),  # default
    ]

    for query, expected_route in test_cases:
        state = {"messages": [HumanMessage(content=query)]}
        result = router_node(state)
        assert result["route"] == expected_route, f"Failed for: {query}"
```

## Hybrid Patterns

Combine router with other patterns:

### Router + Supervisor

```python
# Route to different supervisor teams
def create_hybrid_graph():
    graph = StateGraph(HybridState)

    # Initial router
    graph.add_node("router", router_node)

    # Each route leads to a supervisor
    graph.add_node("sales_supervisor", sales_supervisor)
    graph.add_node("support_supervisor", support_supervisor)

    graph.add_conditional_edges(
        "router",
        lambda s: s["route"],
        {
            "sales": "sales_supervisor",
            "support": "support_supervisor"
        }
    )

    # Each supervisor manages its own subagents...
```

### Router + Tools

```python
# Route based on required tools
def tool_router(state: RouterState) -> dict:
    """Route to agent with appropriate tools."""
    query = state["messages"][0].content

    if "database" in query or "query" in query:
        return {"route": "sql_agent"}  # Has DB tools
    elif "search" in query or "web" in query:
        return {"route": "search_agent"}  # Has search tools
    else:
        return {"route": "general_agent"}
```

## Additional Resources

- LangGraph Graph API (Python): https://docs.langchain.com/oss/python/langgraph/graph-api
- LangGraph Graph API (JavaScript): https://docs.langchain.com/oss/javascript/langgraph/graph-api
