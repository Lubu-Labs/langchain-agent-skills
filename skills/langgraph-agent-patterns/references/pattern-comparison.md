# Multi-Agent Pattern Comparison

Quick reference for choosing the right pattern for your use case.

## Pattern Overview Table

| Pattern | Coordination | Routing | Best For | Complexity |
|---------|-------------|---------|----------|------------|
| **Supervisor** | Central coordinator | Dynamic, LLM-based | Complex workflows, agent collaboration | High |
| **Router** | None | One-time, deterministic | Simple categorization, independent tasks | Low |
| **Orchestrator-Worker** | Orchestrator + Aggregator | Task decomposition | Parallel execution, throughput | Medium |
| **Handoffs** | Agent-to-agent | Sequential, explicit | Linear workflows, build-on-previous | Medium |

## Decision Tree

```
Start: Do you need multiple agents?
├─ No → Use simple single-agent pattern
└─ Yes → Continue

    Can tasks be done in parallel?
    ├─ Yes → Orchestrator-Worker Pattern
    └─ No → Continue

        Is routing decision deterministic?
        ├─ Yes → Router Pattern
        └─ No → Continue

            Do agents need to collaborate?
            ├─ Yes → Supervisor Pattern
            └─ No → Handoffs Pattern
```

## Detailed Comparison

### Supervisor Pattern

**Architecture:**
```
User → Supervisor ⇄ Agent1
              ⇄ Agent2
              ⇄ Agent3
         ↓
       END
```

**Characteristics:**
- **Coordination:** Central supervisor makes all routing decisions
- **Routing:** Dynamic, context-aware, LLM-based
- **Communication:** All agents communicate through supervisor
- **Flexibility:** High - can adapt to changing requirements
- **Cost:** Higher (LLM calls for routing)
- **Latency:** Higher (sequential routing decisions)

**Use When:**
- Workflow is complex and context-dependent
- Agents need to collaborate or share information
- Routing logic is too complex for simple rules
- You need centralized control and visibility

**Avoid When:**
- Simple linear workflow
- Performance/latency is critical
- Routing is deterministic
- Budget is constrained

**Example Use Cases:**
- Research + writing + review pipeline
- Customer support with escalation
- Multi-step analysis tasks
- Complex decision-making workflows

### Router Pattern

**Architecture:**
```
User → Router → Agent1 → END
           ├→ Agent2 → END
           └→ Agent3 → END
```

**Characteristics:**
- **Coordination:** None (one-shot routing)
- **Routing:** Deterministic, rule-based or simple classification
- **Communication:** No inter-agent communication
- **Flexibility:** Low - fixed routing logic
- **Cost:** Lower (minimal LLM usage)
- **Latency:** Lower (single routing decision)

**Use When:**
- Routing can be determined from initial request
- Tasks are independent
- Low latency is important
- Clear categorization exists

**Avoid When:**
- Agents need to collaborate
- Dynamic routing needed
- Multi-step workflows required

**Example Use Cases:**
- Customer support ticketing (sales/support/billing)
- Content categorization
- Request triage
- Intent-based routing

### Orchestrator-Worker Pattern

**Architecture:**
```
User → Orchestrator → Worker1 ┐
                  → Worker2  ├→ Aggregator → END
                  → Worker3 ┘
```

**Characteristics:**
- **Coordination:** Orchestrator for task decomposition, Aggregator for synthesis
- **Routing:** Task decomposition into parallel subtasks
- **Communication:** Workers are independent, results aggregated
- **Flexibility:** Medium - handles parallelizable tasks
- **Cost:** Higher (multiple parallel LLM calls)
- **Latency:** Lower (parallel execution)

**Use When:**
- Tasks can be decomposed into independent subtasks
- Subtasks can run in parallel
- Throughput is important
- Results need synthesis

**Avoid When:**
- Tasks are sequential
- Dependencies between subtasks
- Resource constraints (memory, API limits)

**Example Use Cases:**
- Multi-source research (web, academic, news)
- Parallel document processing
- Distributed data analysis
- Concurrent API calls

### Handoffs Pattern

**Architecture:**
```
User → Agent1 → Agent2 → Agent3 → END
       (with context passing)
```

**Characteristics:**
- **Coordination:** Agent-to-agent handoffs
- **Routing:** Sequential with explicit handoff points
- **Communication:** Direct agent-to-agent with context preservation
- **Flexibility:** Medium - agents control handoffs
- **Cost:** Medium (sequential LLM calls)
- **Latency:** Medium (sequential execution)

**Use When:**
- Clear sequence of specialized tasks
- Each agent builds on previous work
- Context must be preserved
- Agents have distinct responsibilities

**Avoid When:**
- Parallel execution needed
- Complex routing logic required
- No clear sequence exists

**Example Use Cases:**
- Research → Write → Edit pipeline
- Data collection → Analysis → Reporting
- Planning → Execution → Review
- Draft → Review → Approval workflows

## Performance Comparison

### Latency

**Fastest to Slowest:**
1. **Router** - Single routing decision
2. **Orchestrator-Worker** - Parallel execution
3. **Handoffs** - Sequential but direct
4. **Supervisor** - Sequential with routing overhead

### Token Usage

**Most Efficient to Least:**
1. **Router** - Minimal LLM calls
2. **Handoffs** - One call per agent
3. **Orchestrator-Worker** - Decomposition + aggregation + workers
4. **Supervisor** - Routing decisions + agent work

### Complexity

**Simplest to Most Complex:**
1. **Router** - Simple routing logic
2. **Handoffs** - Sequential flow
3. **Orchestrator-Worker** - Parallel coordination
4. **Supervisor** - Dynamic routing + coordination

## Cost Considerations

### Token Cost Formula

**Router:**
```
Cost = routing_decision + agent_execution
```

**Supervisor:**
```
Cost = Σ(routing_decision_i) + Σ(agent_execution_i)
where i = number of routing iterations
```

**Orchestrator-Worker:**
```
Cost = decomposition + (worker_execution × num_workers) + aggregation
```

**Handoffs:**
```
Cost = Σ(agent_execution_i) for each agent in sequence
```

### Cost Optimization Strategies

**Router:**
- Use keyword matching instead of LLM for simple cases
- Cache routing decisions for similar queries

**Supervisor:**
- Use cheaper models for routing (gpt-4o-mini)
- Implement routing caches
- Add early termination logic

**Orchestrator-Worker:**
- Limit number of parallel workers
- Use cheaper models for workers
- Implement result streaming

**Handoffs:**
- Summarize context before handoff
- Use cheaper models where appropriate
- Skip unnecessary handoffs

## Scaling Characteristics

### Horizontal Scaling

**Best to Worst:**
1. **Orchestrator-Worker** - Natural parallelization
2. **Router** - Independent agent execution
3. **Handoffs** - Can parallelize within agents
4. **Supervisor** - Sequential bottleneck

### Vertical Scaling (Complexity)

**How Each Pattern Scales with Complexity:**

**Router:**
- Scales poorly - routing logic becomes complex
- Consider upgrading to Supervisor for complex cases

**Supervisor:**
- Scales well - LLM handles complexity
- May need hierarchical supervisors for very complex cases

**Orchestrator-Worker:**
- Scales well - add more workers
- Aggregation may become bottleneck

**Handoffs:**
- Scales moderately - add more agents in sequence
- Long chains may lose context

## Debugging Difficulty

**Easiest to Hardest:**

1. **Router** - Single decision point, clear flow
2. **Handoffs** - Linear flow, easy to trace
3. **Orchestrator-Worker** - Parallel execution can be confusing
4. **Supervisor** - Dynamic routing hard to predict

### Debugging Strategies by Pattern

**Router:**
- Log routing decision with input query
- Test routing logic with unit tests

**Supervisor:**
- Use LangSmith to visualize routing decisions
- Log supervisor reasoning at each step
- Add routing decision validation

**Orchestrator-Worker:**
- Track worker progress and failures
- Log aggregation inputs
- Monitor parallel execution

**Handoffs:**
- Log handoff points and context
- Visualize handoff sequence
- Track context size

## Migration Paths

### Upgrading Patterns

**Router → Supervisor:**
When routing becomes too complex:

```python
# Before: Simple router
def route(query):
    if "sales" in query:
        return "sales_agent"
    ...

# After: Supervisor with dynamic routing
def supervisor(state):
    decision = llm.invoke(f"Route this query: {state['query']}")
    return {"next": decision.content}
```

**Handoffs → Supervisor:**
When you need dynamic routing:

```python
# Before: Fixed handoff sequence
Agent1 → Agent2 → Agent3

# After: Supervisor decides routing
Supervisor ⇄ Agent1/Agent2/Agent3
```

**Supervisor → Orchestrator-Worker:**
When tasks become parallelizable:

```python
# Before: Sequential supervision
Supervisor → Agent1 → Supervisor → Agent2 → ...

# After: Parallel execution
Orchestrator → [Agent1, Agent2, Agent3] → Aggregator
```

### Downgrading Patterns

**Supervisor → Router:**
When routing becomes predictable:

```python
# If supervisor always routes the same way,
# replace with deterministic router
```

## Hybrid Patterns

### Router + Supervisor

Route to different supervisor teams:

```
Router → Sales Supervisor ⇄ Sales Agents
     ↓
     Support Supervisor ⇄ Support Agents
```

**Use When:**
- Different domains need different coordination
- Top-level categorization is simple
- Domain-specific complexity exists

### Supervisor + Orchestrator

Supervisor dispatches to orchestrators:

```
Supervisor → Research Orchestrator → Workers → Aggregator
         ↓
         Writing Agent
```

**Use When:**
- Some tasks are parallelizable, others sequential
- Mix of coordination strategies needed

### Hierarchical Supervisor

Supervisors manage other supervisors:

```
Top Supervisor → Research Supervisor → Research Agents
             ↓
             Writing Supervisor → Writing Agents
```

**Use When:**
- Very complex workflows
- Multiple coordination levels
- Team-based organization

## Summary Table

| Feature | Supervisor | Router | Orchestrator | Handoffs |
|---------|-----------|--------|--------------|----------|
| **Latency** | High | Low | Medium | Medium |
| **Cost** | High | Low | Medium | Medium |
| **Flexibility** | High | Low | Medium | Medium |
| **Scalability** | Medium | High | High | Medium |
| **Complexity** | High | Low | Medium | Medium |
| **Debugging** | Hard | Easy | Medium | Easy |
| **Collaboration** | Yes | No | No | Yes |
| **Parallel** | No | Yes | Yes | No |

## Quick Decision Guide

**Choose Supervisor when:**
- Need dynamic, context-aware routing
- Agents must collaborate
- Workflow is complex and unpredictable
- Centralized control is important

**Choose Router when:**
- Simple categorization task
- Routes are deterministic
- Low latency required
- Independent agent execution

**Choose Orchestrator-Worker when:**
- Tasks are parallelizable
- High throughput needed
- Results need synthesis
- Resources allow parallel execution

**Choose Handoffs when:**
- Clear sequential workflow
- Each step builds on previous
- Context preservation critical
- Simple coordination sufficient

## Additional Resources

- Pattern implementations: See assets/examples/
- State design for patterns: See state-management-patterns.md
- Individual pattern details: See supervisor-subagent.md, router-pattern.md, orchestrator-worker.md, handoffs.md
