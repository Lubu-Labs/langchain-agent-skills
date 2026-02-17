# Todo-Driven Research Agent Example

Working example demonstrating effective use of `write_todos` for multi-step research tasks.

This example uses simulated tools for deterministic behavior; swap in real tools for production usage.

## What This Example Demonstrates

- **Creating todos**: Breaking down complex tasks into 3-6 actionable steps
- **User approval workflow**: Asking "Does this plan look good?" before executing
- **Status management**: Updating todos from `pending` → `in_progress` → `completed`
- **Progress tracking**: Re-sending updated todo lists with `write_todos`
- **Integration**: Combining todos with tools (search, read, synthesize)

## Setup

### 1. Install Dependencies

```bash
# Enter the example project
cd skills/deepagents-planning-todos/assets/examples/todo-driven-agent

# Create virtual environment
uv venv --python=3.12

# Sync dependencies
uv sync
```

### 2. Configure Environment

```bash
# Copy environment template
cp .env.example .env

# Edit .env and add your API keys
# ANTHROPIC_API_KEY=your-key-here
```

### 3. Run the Example

```bash
uv run agent.py
```

## Expected Output

### Step 1: Agent Creates Todo List

The agent will call `write_todos` with a research plan:

```
Planning research with todos:
  ⏳ [pending] Search for Deep Agents documentation
  ⏳ [pending] Read and extract key findings
  ⏳ [pending] Analyze patterns and themes
  ⏳ [pending] Synthesize into summary
```

### Step 2: Agent Asks for Approval

```
Does this plan look good?
```

**User response**: Type "yes" or "proceed"

### Step 3: Agent Executes Todos

For each todo, the agent will:
1. Update status to `in_progress`
2. Execute the task (search, read, analyze, etc.)
3. Update status to `completed`

### Step 4: Agent Completes

Once all todos are `completed`, the agent finishes and presents the final summary.

## Key Patterns to Notice

### System Prompt Guidance

The agent's system prompt includes specific instructions:
- Create todos for complex tasks (3-6 steps)
- Ask user approval before starting
- Update status promptly as work progresses
- Use `write_todos` again whenever statuses change

### Todo Lifecycle

```python
# Initial: All pending
{"content": "Search for documentation", "status": "pending"}

# Starting work
{"content": "Search for documentation", "status": "in_progress"}

# Finished
{"content": "Search for documentation", "status": "completed"}
```

### Granularity

The example uses 4 todos (search → read → analyze → synthesize), which is ideal for:
- Clear progress tracking
- User understanding
- Not overwhelming

## Customization

### Modify the Research Task

Edit the user message in `agent.py`:

```python
user_message = "Summarize [YOUR TOPIC HERE] using a todo-driven workflow."
```

### Change Todo Structure

Update the system prompt to use different patterns:
- **Coding**: design → implement → test
- **Analysis**: collect → process → analyze → visualize
- **Document processing**: read → extract → transform → output

See `references/todo-patterns.md` for more patterns.

### Add Real Tools

Replace simulated tools with real ones:
- Add a Tavily-based `internet_search` tool (official Deep Agents quickstart pattern)
- Pass `tools=[internet_search]` to `create_deep_agent`
- Set `TAVILY_API_KEY` in your environment

### Enable LangSmith Tracing

Set these values in `.env`:

```bash
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=your-langsmith-api-key
LANGSMITH_PROJECT=todo-driven-agent
```

Then visualize todos:

```bash
# Export trace from LangSmith (download JSON)
# Then run:
uv run ../../scripts/visualize_todos.py trace.json
```

## Troubleshooting

**Agent creates todos but doesn't follow them**:
- Verify system prompt includes instructions to update status
- Check that `TodoListMiddleware` is included (default in `create_deep_agent`)

**Agent doesn't ask for approval**:
- Add explicit instruction in system prompt: "Ask user: 'Does this plan look good?'"

**Todos stuck in `in_progress`**:
- Ensure agent updates status to `completed` after finishing each task
- Check system prompt for status update guidance

## Resources

- **Skill documentation**: `skills/deepagents-planning-todos/SKILL.md`
- **Detailed patterns**: `skills/deepagents-planning-todos/references/todo-patterns.md`
- **Deep Agents docs**: https://docs.langchain.com/oss/python/deepagents/overview
- **Deep Agents quickstart**: https://docs.langchain.com/oss/python/deepagents/quickstart
- **Deep Agents customization**: https://docs.langchain.com/oss/python/deepagents/customization
- **LangSmith tracing for Deep Agents**: https://docs.langchain.com/langsmith/trace-deep-agents
