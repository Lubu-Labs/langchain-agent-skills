# Todo Patterns for Deep Agents

Detailed todo structures and code examples for common task types.

> Compatibility note: Deep Agents is currently pre-1.0, so minor-version upgrades may change APIs. Re-validate examples when upgrading Deep Agents.

## Research Task Pattern

**Context**: Use this pattern for information gathering and synthesis tasks.

**Workflow**: gather → synthesize → report

**Todo Structure**:
```json
{
  "todos": [
    {"content": "Search for relevant documentation/papers", "status": "pending"},
    {"content": "Read and extract key findings", "status": "pending"},
    {"content": "Analyze patterns and identify themes", "status": "pending"},
    {"content": "Synthesize findings into summary", "status": "pending"}
  ]
}
```

**Code Example**:
```python
from deepagents import create_deep_agent
import os
from typing import Literal
from tavily import TavilyClient

tavily_client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])

def internet_search(
    query: str,
    max_results: int = 5,
    topic: Literal["general", "news", "finance"] = "general",
    include_raw_content: bool = False,
):
    """Run a web search."""
    return tavily_client.search(
        query,
        max_results=max_results,
        include_raw_content=include_raw_content,
        topic=topic,
    )

# Create agent with TodoListMiddleware (included by default)
agent = create_deep_agent(
    model="anthropic:claude-sonnet-4-5-20250929",
    tools=[internet_search],
    system_prompt="""
You are a research assistant. For multi-step research tasks:
1. Create a plan with write_todos
2. Ask the user: "Does this plan look good?"
3. Once approved, execute each step and update status
4. Update todo statuses with write_todos as work progresses
"""
)

# User: "Research recent advances in LLM agents"
# Agent response:
# - Calls write_todos with 4-step research plan
# - Asks: "Does this plan look good?"
# - User approves
# - Executes step by step, updating status
```

**Common Pitfalls**:
- Too granular: "Search Google", "Click first link", "Read paragraph 1" → Too many steps
- Too vague: "Do research" → Unclear completion criteria
- No synthesis step → Missing the final deliverable

---

## Coding Task Pattern

**Context**: Use this pattern for feature implementation and refactoring.

**Workflow**: design → implement → test → refactor

**Todo Structure**:
```json
{
  "todos": [
    {"content": "Design API endpoints and data flow", "status": "pending"},
    {"content": "Implement core functionality", "status": "pending"},
    {"content": "Write unit and integration tests", "status": "pending"},
    {"content": "Test end-to-end workflow", "status": "pending"}
  ]
}
```

**Code Example**:
```python
from deepagents import create_deep_agent

# Create agent with filesystem access (via FilesystemMiddleware)
agent = create_deep_agent(
    model="anthropic:claude-sonnet-4-5-20250929",
    # Built-in tools (write_todos + filesystem + subagents) are available by default.
    tools=[],
    system_prompt="""
You are a coding assistant. For implementation tasks:
1. Create a plan with write_todos (design, implement, test)
2. Show the plan to the user for approval
3. Execute each step:
   - Design: Plan the structure and API
   - Implement: Write the code using filesystem tools
   - Test: Write tests and verify functionality
4. Update todo status as you complete each step
"""
)

# User: "Implement user authentication with JWT"
# Agent response:
# - Calls write_todos with design/implement/test/verify plan
# - Asks: "Does this plan look good?"
# - User approves
# - Executes: designs auth flow, implements endpoints, writes tests
```

**Integration with Filesystem Tools**:
```python
# Agent workflow with filesystem + todos:
# 1. write_todos: Create plan
# 2. write_file: Create auth.py
# 3. write_todos: Update first task to "completed"
# 4. write_file: Create tests/test_auth.py
# 5. write_todos: Update second task to "completed"
# etc.
```

**Common Pitfalls**:
- Skipping design step → Jumping straight to coding without plan
- No test coverage → Forgetting to add tests to the plan
- Missing refactor step → Leaving technical debt

---

## Analysis Task Pattern

**Context**: Use this pattern for data processing and analysis workflows.

**Workflow**: collect → process → analyze → visualize

**Todo Structure**:
```json
{
  "todos": [
    {"content": "Collect data from sources", "status": "pending"},
    {"content": "Process and clean data", "status": "pending"},
    {"content": "Analyze patterns and trends", "status": "pending"},
    {"content": "Visualize results and create report", "status": "pending"}
  ]
}
```

**Code Example**:
```python
from deepagents import create_deep_agent

# Analysis tools
def fetch_data_tool(source: str) -> dict:
    """Fetch data from source."""
    return {"source": source, "records": []}

def analyze_tool(data: dict) -> dict:
    """Analyze data for patterns."""
    return {"record_count": len(data.get("records", []))}

# Create agent
agent = create_deep_agent(
    model="anthropic:claude-sonnet-4-5-20250929",
    tools=[fetch_data_tool, analyze_tool],
    system_prompt="""
You are a data analyst. For analysis tasks:
1. Create a plan with write_todos (collect, process, analyze, visualize)
2. Ask user approval
3. Execute data pipeline step by step
4. Update status after each stage completes
"""
)

# User: "Analyze LangSmith trace patterns from last week"
# Agent response:
# - Calls write_todos with data pipeline plan
# - Asks: "Does this plan look good?"
# - Executes: fetches traces, processes data, analyzes patterns, creates report
```

**Checkpoint Patterns**:
```python
# For long-running analysis, save intermediate results:
# 1. Collect data → write_file: raw_data.json
# 2. Process data → write_file: processed_data.json
# 3. Analyze → write_file: analysis_results.json
# 4. Visualize → write_file: report.md

# This allows resuming if interrupted
```

**Common Pitfalls**:
- No data validation → Processing corrupt data
- Skipping visualization → Hard to understand results
- Missing checkpoints → Losing progress on failures

---

## Document Processing Pattern

**Context**: Use this pattern for batch document operations (parsing, extraction, transformation).

**Workflow**: read → extract → transform → output

**Todo Structure**:
```json
{
  "todos": [
    {"content": "Read input documents", "status": "pending"},
    {"content": "Extract key information", "status": "pending"},
    {"content": "Transform to target format", "status": "pending"},
    {"content": "Output results and verify", "status": "pending"}
  ]
}
```

**Code Example**:
```python
from deepagents import create_deep_agent

# Document processing tools
def read_pdf_tool(path: str) -> str:
    """Extract text from PDF."""
    return f"Extracted text from: {path}"

def extract_entities_tool(text: str) -> list[dict[str, str]]:
    """Extract entities from text."""
    return [{"type": "example", "value": text[:20]}]

# Create agent
agent = create_deep_agent(
    model="anthropic:claude-sonnet-4-5-20250929",
    tools=[read_pdf_tool, extract_entities_tool],
    system_prompt="""
You are a document processor. For batch processing:
1. Create a plan with write_todos (read, extract, transform, output)
2. Show plan to user
3. Process documents in batches
4. Update status after each batch completes
"""
)

# User: "Extract names and dates from all PDFs in /documents"
# Agent response:
# - Calls write_todos with processing pipeline
# - Asks: "Does this plan look good?"
# - Processes: reads PDFs, extracts entities, transforms to JSON, outputs results
```

**Batch Processing Considerations**:
```python
# For large batches, track progress within todos:
{
  "todos": [
    {"content": "Read input documents (0/100)", "status": "in_progress"},
    {"content": "Extract entities", "status": "pending"},
    {"content": "Transform to JSON", "status": "pending"},
    {"content": "Output results", "status": "pending"}
  ]
}

# Update content as batches complete:
# "Read input documents (25/100)" → "Read input documents (50/100)"
```

**Common Pitfalls**:
- No error handling → One bad document breaks entire pipeline
- Missing verification → Not checking output quality
- Memory issues → Loading all documents at once

---

## Advanced Patterns

### Adaptive Replanning

**When to use**: Task complexity changes mid-execution (e.g., unexpected errors, new requirements).

**Pattern**: Update todo list based on intermediate results.

**Example**:
```python
# write_todos payload (initial):
{
  "todos": [
    {"content": "Search for API docs", "status": "pending"},
    {"content": "Implement integration", "status": "pending"},
  ]
}

# write_todos payload (after discovering deprecation):
{
  "todos": [
    {"content": "Search for API docs", "status": "completed"},
    {"content": "Research alternative APIs", "status": "pending"},  # New todo
    {"content": "Implement integration with new API", "status": "pending"},  # Updated
  ]
}
```

**Guidance**:
- Ask user before major replanning: "The original API is deprecated. Should I research alternatives?"
- Keep original completed todos for context
- Add new pending todos for changed requirements

### Nested Task Hierarchies

**When to use**: Complex tasks with subtasks (e.g., multi-feature implementation).

**Pattern**: Use subagents for subtask execution, parent todos for high-level tracking.

**Example**:
```python
# write_todos payload for parent agent (high-level):
{
  "todos": [
    {"content": "Implement authentication feature", "status": "in_progress"},
    {"content": "Implement dashboard feature", "status": "pending"},
    {"content": "Integration testing", "status": "pending"},
  ]
}

# Subagent for authentication (detailed):
# - Design auth flow
# - Implement login endpoint
# - Write tests
# - Deploy and verify

# Parent marks "Implement authentication feature" completed once subagent finishes
```

**Guidance**:
- Use parent todos for user-facing milestones (3-5 items)
- Delegate details to subagents (can have their own todos)
- Parent updates status when subagent completes

### Conditional Todos

**When to use**: Tasks with branching logic (e.g., if-then scenarios).

**Pattern**: Create initial todos, update based on results.

**Example**:
```python
# write_todos payload (initial):
{
  "todos": [
    {"content": "Check if API is available", "status": "pending"},
    {"content": "If available: integrate directly", "status": "pending"},
    {"content": "If unavailable: implement fallback", "status": "pending"},
  ]
}

# write_todos payload (after checking: API unavailable):
{
  "todos": [
    {"content": "Check if API is available", "status": "completed"},
    {"content": "Implement fallback scraper", "status": "in_progress"},  # Active path
    {"content": "Test fallback accuracy", "status": "pending"},
    # Note: "integrate directly" todo is removed (not needed)
  ]
}
```

**Guidance**:
- Start with conditional todos to show all paths
- Update to keep only active path once decision is made
- Document decision in todo content or separate note

---

## Anti-Patterns

### Too Granular (100+ todos)

**Problem**: Over-planning with micro-tasks ("Open file", "Read line 1", "Read line 2"...)

**Why it fails**:
- Overwhelming to track
- Constant status updates slow down execution
- User loses the big picture

**Fix**: Group micro-tasks into milestones (3-6 high-level todos).

### Too Vague (unclear completion)

**Problem**: "Do research", "Fix bugs", "Improve code"

**Why it fails**:
- No clear completion criteria
- Agent doesn't know when to mark "completed"
- User can't verify progress

**Fix**: Make todos specific and measurable:
- "Search LangChain docs for Deep Agents overview" (clear completion)
- "Fix authentication bug in login.py line 42" (specific)
- "Refactor user model to use TypedDict schema" (measurable)

### Missing Status Updates

**Problem**: Creating todos but never updating status.

**Why it fails**:
- User can't see progress
- Agent loses track of what's done
- Defeats the purpose of planning

**Fix**: Update status promptly:
- Start task → `"status": "in_progress"`
- Finish task → `"status": "completed"`
- Re-send the full current todo list with `write_todos` after meaningful progress

### Ignoring User Approval

**Problem**: Creating todos and executing immediately without asking.

**Why it fails**:
- User might disagree with the plan
- Wastes time on wrong approach
- Violates user autonomy

**Fix**: Always ask: "Does this plan look good?" before starting execution.
