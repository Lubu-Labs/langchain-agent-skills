# Advanced Filtering and Querying

Practical patterns for filtering LangSmith runs using the Python and TypeScript SDKs.

## Filter Syntax

LangSmith supports a functional trace query syntax for filtering runs.
Examples below follow the official trace query syntax and `list_runs`/`listRuns` usage in LangSmith docs.

### Basic Operators

```python
# Equality
'eq(status, "success")'
'neq(status, "error")'

# Comparison
'gt(total_tokens, 10000)'   # Greater than
'gte(total_tokens, 10000)'  # Greater than or equal
'lt(latency, 1000)'         # Less than
'lte(latency, 1000)'        # Less than or equal

# Contains / substring
'has(tags, "production")'   # Tag contains
'search(name, "my-agent")'  # Name contains substring
'in(metadata_key, ["session_id","conversation_id","thread_id"])'  # Match any value in a list
```

### Logical Operators

```python
# AND - all conditions must be true (documented in trace query syntax)
'and(eq(status, "success"), gt(total_tokens, 1000))'
```

## Common Query Patterns

### Filter by Metadata

```python
from langsmith import Client

client = Client()

# Single metadata key-value
filter_query = 'and(eq(metadata_key, "job_id"), eq(metadata_value, "abc123"))'
runs = client.list_runs(project_name="my-project", filter=filter_query, is_root=True)

# Multiple metadata fields (JSON containment)
filter_query = 'has(metadata, "{\"environment\":\"production\",\"user_id\":\"user123\"}")'
runs = client.list_runs(project_name="my-project", filter=filter_query, is_root=True)

# In array (thread_id, session_id, conversation_id)
thread_id = "thread-abc"
filter_query = f'and(in(metadata_key, ["session_id","conversation_id","thread_id"]), eq(metadata_value, "{thread_id}"))'
```

### Filter by Status

```python
# Successful runs only
for run in client.list_runs(
    project_name="my-project",
    filter='eq(status, "success")',
    is_root=True
):
    print(f"Success: {run.id}")

# Failed runs
for run in client.list_runs(
    project_name="my-project",
    filter='eq(status, "error")',
    is_root=True
):
    print(f"Error: {run.id}")

# Pending runs
for run in client.list_runs(
    project_name="my-project",
    filter='eq(status, "pending")',
    is_root=True
):
    print(f"In progress: {run.id}")
```

### Filter by Time

```python
from datetime import datetime, timedelta, timezone

# Last 24 hours
start_time = datetime.now(timezone.utc) - timedelta(hours=24)
runs = client.list_runs(
    project_name="my-project",
    start_time=start_time,
    is_root=True
)

# Specific date range
start = datetime(2026, 2, 1, tzinfo=timezone.utc)
end = datetime(2026, 2, 7, tzinfo=timezone.utc)
runs = client.list_runs(
    project_name="my-project",
    start_time=start,
    end_time=end,
    is_root=True
)
```

### Filter by Tags

```python
# Has specific tag
for run in client.list_runs(
    project_name="my-project",
    filter='has(tags, "experiment-v2")',
    is_root=True
):
    print(f"Tagged run: {run.id}")

# Multiple tags (run must have both)
filter_query = 'and(has(tags, "production"), has(tags, "high-priority"))'
runs = client.list_runs(project_name="my-project", filter=filter_query, is_root=True)
```

### Filter by Performance

```python
# High token usage (>100k tokens)
for run in client.list_runs(
    project_name="my-project",
    filter='gt(total_tokens, 100000)',
    is_root=True
):
    print(f"High token run: {run.id}, {run.total_tokens} tokens")

# Slow runs (>10 seconds)
for run in client.list_runs(
    project_name="my-project",
    filter='gt(latency, 10000)',  # Latency in milliseconds
    is_root=True
):
    print(f"Slow run: {run.id}")

# Expensive runs (cost estimation requires metadata)
filter_query = 'gt(total_cost, 0.10)'
```

### Filter by Model

```python
# Specific model
filter_query = 'and(eq(metadata_key, "ls_model_name"), eq(metadata_value, "gpt-4"))'
runs = client.list_runs(project_name="my-project", filter=filter_query, is_root=True)

# Model provider
filter_query = 'and(eq(metadata_key, "ls_provider"), eq(metadata_value, "openai"))'
runs = client.list_runs(project_name="my-project", filter=filter_query, is_root=True)
```

## TypeScript Examples

```typescript
import { Client } from "langsmith";

const client = new Client();

// Filter by metadata
const filterQuery = 'and(eq(metadata_key, "job_id"), eq(metadata_value, "abc123"))';
for await (const run of client.listRuns({
  projectName: "my-project",
  filter: filterQuery,
  isRoot: true,
})) {
  console.log(`Run: ${run.id}`);
}

// Filter by status
for await (const run of client.listRuns({
  projectName: "my-project",
  filter: 'eq(status, "error")',
  isRoot: true,
})) {
  console.log(`Error: ${run.id}`);
}

// Filter by time
const startTime = new Date(Date.now() - 24 * 60 * 60 * 1000);
for await (const run of client.listRuns({
  projectName: "my-project",
  startTime,
  isRoot: true,
})) {
  console.log(`Recent run: ${run.id}`);
}

// Filter by tags
for await (const run of client.listRuns({
  projectName: "my-project",
  filter: 'has(tags, "production")',
  isRoot: true,
})) {
  console.log(`Tagged: ${run.id}`);
}
```

## Advanced Patterns

### Find Outliers

```python
# Find runs with unusually high token usage
# (Requires calculating statistics first)

all_runs = list(client.list_runs(project_name="my-project", is_root=True, limit=1000))
token_counts = [r.total_tokens for r in all_runs if r.total_tokens]

from statistics import mean, stdev

avg = mean(token_counts)
std = stdev(token_counts)
threshold = avg + 2 * std  # 2 standard deviations above mean

# Query for outliers
filter_query = f'gt(total_tokens, {int(threshold)})'
outliers = client.list_runs(project_name="my-project", filter=filter_query, is_root=True)
```

### Batch Processing

```python
from itertools import islice

def process_runs_in_batches(project_name: str, filter_query: str, batch_size: int = 100):
    """Stream and process runs in fixed-size batches."""
    run_iter = client.list_runs(
        project_name=project_name,
        filter=filter_query,
        is_root=True,
    )

    while True:
        batch = list(islice(run_iter, batch_size))
        if not batch:
            break

        for run in batch:
            process_run(run)
```

### Conditional Filtering Based on Run Properties

```python
def get_failing_runs_with_context(project_name: str, since_hours: int = 24):
    """Get recent failing runs with their metadata."""
    start_time = datetime.now(timezone.utc) - timedelta(hours=since_hours)

    # First, get error runs
    error_runs = []
    for run in client.list_runs(
        project_name=project_name,
        filter='eq(status, "error")',
        start_time=start_time,
        is_root=True
    ):
        error_runs.append({
            "id": str(run.id),
            "name": run.name,
            "error": run.error,
            "metadata": run.metadata,
            "start_time": run.start_time.isoformat() if run.start_time else None,
        })

    return error_runs
```

## Performance Tips

1. **Use `is_root=True`**: Only fetch top-level runs (traces), not child runs
2. **Limit results**: Use `limit` parameter for large datasets
3. **Filter server-side**: Always prefer filter queries over fetching and filtering locally
4. **Use time ranges**: Narrow down queries with `start_time` and `end_time`
5. **Batch processing**: Process large result sets in chunks

## Troubleshooting

### "No runs found"
- Check project name spelling
- Verify filter syntax (start with a minimal `eq(...)`, then add `and(...)`)
- Check time zone (use `timezone.utc`)

### "Invalid filter"
- Ensure quotes around string values: `eq(status, "success")` not `eq(status, success)`
- Use proper function names: `eq`, `gt`, `has`, etc.
- Check parentheses balance in complex queries

### "Too many results"
- Add time filters: `start_time`, `end_time`
- Use `limit` parameter
- Add more specific filters (status, metadata, etc.)
