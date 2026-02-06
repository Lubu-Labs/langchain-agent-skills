# Analysis Patterns for LangSmith Traces

Proven patterns for analyzing traces at scale, finding insights, and debugging issues.

> Compatibility note: LangSmith run fields are top-level (`status`, `start_time`, `total_tokens`, `prompt_tokens`, `latency`). Some local export workflows also include a normalized `messages` array; snippets below use it when present.

## Pattern 1: Failure Categorization

**Use case**: Understand why agents fail and prioritize fixes

### Categorization Framework

```
Failures
├── Infrastructure (not agent's fault)
│   ├── DaytonaError (sandbox failed)
│   ├── ProcessInterruptedError (manual stop)
│   └── BadRequestError (API error)
├── Resource Limits (may indicate agent issues)
│   ├── GraphRecursionError (hit step limit)
│   └── TimeoutError (exceeded time limit)
└── Task Failures (agent completed but failed task)
    ├── Model Logic Failures
    │   ├── Misunderstanding (misread requirements)
    │   ├── Wrong Strategy (ineffective approach)
    │   ├── Incomplete Solution (partial completion)
    │   ├── Gave Up Early (premature stopping)
    │   ├── No Verification (didn't test)
    │   └── Format Error (right answer, wrong format)
    └── Tool/Execution Failures
        ├── Tool Misuse (wrong tool/args)
        ├── Environment Mismatch (wrong assumptions)
        ├── Non-recovery (repeated same error)
        └── Tool Timeout (tool took too long)
```

### Implementation

```python
def categorize_failure(trace: Dict) -> tuple[str, str]:
    """Categorize a failed trace."""
    messages = trace.get("messages", [])
    status = trace.get("status") or trace.get("metadata", {}).get("status")

    # Check for infrastructure errors first
    if status == "error" and len(messages) < 3:
        return "infrastructure", "quick_failure"

    # Check message content for specific errors
    last_messages = " ".join(str(m.get("content", "")) for m in messages[-5:])

    if "DaytonaError" in last_messages:
        return "infrastructure", "sandbox_error"
    elif "GraphRecursionError" in last_messages:
        return "resource_limit", "recursion_limit"
    elif "TimeoutError" in last_messages or "AgentTimeoutError" in last_messages:
        return "resource_limit", "timeout"

    # Analyze tool usage patterns
    tool_calls = []
    for msg in messages:
        if msg.get("tool_calls"):
            tool_calls.extend([tc.get("name") for tc in msg["tool_calls"]])

    # Check for repeated tool calls (same tool >3 times in a row)
    for i in range(len(tool_calls) - 3):
        if len(set(tool_calls[i:i+3])) == 1:
            return "tool_issue", "stuck_in_loop"

    # Check if agent gave up early
    if len(messages) < 10:
        return "model_logic", "gave_up_early"

    # Default to model logic failure
    return "model_logic", "unknown"
```

## Pattern 2: Compare Passed vs Failed

**Use case**: Find what successful traces do differently

### Statistical Comparison

```python
from statistics import mean

def compare_passed_failed(passed_dir: Path, failed_dir: Path):
    """Compare statistics between passed and failed traces."""

    def analyze_group(directory: Path):
        traces = []
        for trace_file in directory.glob("*.json"):
            with open(trace_file) as f:
                trace = json.load(f)
                messages = trace.get("messages", [])

                traces.append({
                    "message_count": len(messages),
                    "tool_calls": sum(1 for m in messages if m.get("tool_calls")),
                    "tokens": trace.get("total_tokens")
                    or trace.get("metadata", {}).get("token_usage", {}).get("total_tokens", 0),
                })
        return traces

    passed = analyze_group(passed_dir)
    failed = analyze_group(failed_dir)

    print(f"Passed ({len(passed)} traces):")
    print(f"  Avg messages: {mean(t['message_count'] for t in passed):.1f}")
    print(f"  Avg tool calls: {mean(t['tool_calls'] for t in passed):.1f}")
    print(f"  Avg tokens: {mean(t['tokens'] for t in passed if t['tokens'] > 0):.0f}")

    print(f"\nFailed ({len(failed)} traces):")
    print(f"  Avg messages: {mean(t['message_count'] for t in failed):.1f}")
    print(f"  Avg tool calls: {mean(t['tool_calls'] for t in failed):.1f}")
    print(f"  Avg tokens: {mean(t['tokens'] for t in failed if t['tokens'] > 0):.0f}")
```

### Tool Usage Comparison

```python
from collections import Counter

def compare_tool_usage(passed_traces: List[Dict], failed_traces: List[Dict]):
    """Compare which tools are used in passed vs failed traces."""

    def extract_tools(traces):
        tools = []
        for trace in traces:
            for msg in trace.get("messages", []):
                if msg.get("tool_calls"):
                    tools.extend([tc["name"] for tc in msg["tool_calls"]])
        return Counter(tools)

    passed_tools = extract_tools(passed_traces)
    failed_tools = extract_tools(failed_traces)

    # Find tools more common in failures
    all_tools = set(passed_tools.keys()) | set(failed_tools.keys())

    print("Tool usage differences:")
    for tool in sorted(all_tools):
        passed_pct = passed_tools[tool] / sum(passed_tools.values()) * 100 if passed_tools.values() else 0
        failed_pct = failed_tools[tool] / sum(failed_tools.values()) * 100 if failed_tools.values() else 0
        diff = failed_pct - passed_pct

        if abs(diff) > 5:  # >5% difference
            print(f"  {tool}: {passed_pct:.1f}% (passed) vs {failed_pct:.1f}% (failed) [{diff:+.1f}%]")
```

## Pattern 3: Parallel Analysis with Agents

**Use case**: Analyze hundreds of traces efficiently

See `parallel-analysis.md` in the example directory for the complete workflow, or use this condensed version:

### Setup

1. Create manifest listing all traces
2. Split into batches (50-100 traces per batch)
3. Define analysis criteria
4. Launch multiple agents writing to shared file

### Analysis Criteria Example

```markdown
For each trace, determine:
1. Outcome category (infrastructure/resource_limit/model_logic/tool_issue)
2. Subcategory (specific pattern)
3. Summary (one sentence)
4. Root cause (why it happened)
5. Actionable insight (what could help)

Output format (JSONL):
{"trace_id": "...", "category": "...", "subcategory": "...", "summary": "...", "root_cause": "...", "actionable": "..."}
```

### Aggregation Script

```python
import json

findings = []
with open("findings.jsonl") as f:
    for line in f:
        if line.strip():
            findings.append(json.loads(line))

# Group by category
from collections import defaultdict

by_category = defaultdict(list)
for f in findings:
    by_category[f["category"]].append(f)

for cat, items in sorted(by_category.items(), key=lambda x: -len(x[1])):
    print(f"{cat}: {len(items)}")

    # Most common actionable insights in this category
    actions = Counter(item["actionable"] for item in items if item.get("actionable"))
    for action, count in actions.most_common(3):
        print(f"  ({count}x) {action}")
```

## Pattern 4: Temporal Analysis

**Use case**: Understand how agent behavior changes over time

### Track Metrics Over Time

```python
from datetime import datetime
from collections import defaultdict

def analyze_over_time(traces: List[Dict], bucket_hours: int = 1):
    """Group traces by time buckets and analyze trends."""
    by_time = defaultdict(list)

    for trace in traces:
        start = trace.get("start_time") or trace.get("metadata", {}).get("start_time")
        if not start:
            continue

        dt = datetime.fromisoformat(start.replace("Z", "+00:00"))
        bucket = dt.replace(minute=0, second=0, microsecond=0)

        by_time[bucket].append(trace)

    # Analyze each bucket
    for bucket in sorted(by_time.keys()):
        traces_in_bucket = by_time[bucket]
        total = len(traces_in_bucket)
        errors = sum(
            1
            for t in traces_in_bucket
            if (t.get("status") or t.get("metadata", {}).get("status")) == "error"
        )

        print(f"{bucket.isoformat()}: {total} traces, {errors} errors ({errors/total*100:.1f}%)")
```

## Pattern 5: Token Efficiency Analysis

**Use case**: Identify token waste and optimize costs

### Find Token-Heavy Traces

```python
def analyze_token_efficiency(traces: List[Dict]):
    """Find traces with high token usage."""

    # Sort by tokens
    by_tokens = sorted(
        traces,
        key=lambda t: t.get("total_tokens")
        or t.get("metadata", {}).get("token_usage", {}).get("total_tokens", 0),
        reverse=True
    )

    print("Top 10 token-heavy traces:")
    for trace in by_tokens[:10]:
        trace_id = trace.get("trace_id")
        tokens = trace.get("total_tokens") or trace.get("metadata", {}).get("token_usage", {}).get("total_tokens", 0)
        messages = len(trace.get("messages", []))

        # Calculate tokens per message
        tpm = tokens / messages if messages > 0 else 0

        print(f"  {trace_id}: {tokens:,} tokens, {messages} msgs ({tpm:.0f} tokens/msg)")
```

### Identify Context Window Issues

```python
def find_context_issues(traces: List[Dict]):
    """Find traces that may be hitting context limits."""

    issues = []
    for trace in traces:
        messages = trace.get("messages", [])
        tokens = trace.get("prompt_tokens") or trace.get("metadata", {}).get("token_usage", {}).get("prompt_tokens", 0)

        # Check for patterns indicating context issues
        if tokens > 100000:  # >100k prompt tokens
            issues.append({
                "trace_id": trace.get("trace_id"),
                "tokens": tokens,
                "messages": len(messages),
                "issue": "very_high_prompt_tokens"
            })

        # Check for repeated information
        if len(messages) > 50:
            # Simple heuristic: check if late messages reference early context
            early_content = " ".join(str(m.get("content", "")) for m in messages[:5])
            late_content = " ".join(str(m.get("content", "")) for m in messages[-5:])

            # If agent is "forgetting" or "losing track"
            if "forgot" in late_content.lower() or "earlier" in late_content.lower():
                issues.append({
                    "trace_id": trace.get("trace_id"),
                    "tokens": tokens,
                    "messages": len(messages),
                    "issue": "possible_context_loss"
                })

    return issues
```

## Pattern 6: Error Pattern Detection

**Use case**: Find systematic errors that affect multiple traces

### Detect Repeated Errors

```python
from collections import Counter

def find_error_patterns(error_traces: List[Dict]):
    """Find common error patterns across traces."""

    # Extract error messages
    error_messages = []
    for trace in error_traces:
        messages = trace.get("messages", [])

        # Look for error indicators in last few messages
        for msg in messages[-5:]:
            content = str(msg.get("content", ""))

            # Extract error patterns
            if "Error:" in content or "Exception:" in content:
                # Extract just the error type
                for line in content.split("\n"):
                    if "Error" in line or "Exception" in line:
                        error_messages.append(line.strip()[:100])  # First 100 chars

    # Count occurrences
    common_errors = Counter(error_messages).most_common(10)

    print("Most common error patterns:")
    for error, count in common_errors:
        print(f"  ({count}x) {error}")
```

## Pattern 7: A/B Testing Agent Versions

**Use case**: Compare two agent versions objectively

### Statistical Comparison

```python
def compare_agent_versions(traces_v1: List[Dict], traces_v2: List[Dict]):
    """Compare two agent versions statistically."""

    def compute_metrics(traces):
        total = len(traces)
        token_values = [
            t.get("total_tokens") or t.get("metadata", {}).get("token_usage", {}).get("total_tokens", 0)
            for t in traces
            if (t.get("total_tokens") or t.get("metadata", {}).get("token_usage", {}).get("total_tokens", 0)) > 0
        ]
        latency_values = [
            t.get("latency") or t.get("metadata", {}).get("duration_ms", 0)
            for t in traces
            if (t.get("latency") or t.get("metadata", {}).get("duration_ms", 0)) > 0
        ]

        return {
            "total": total,
            "success_rate": sum(
                1 for t in traces if (t.get("status") or t.get("metadata", {}).get("status")) == "success"
            ) / total if total > 0 else 0.0,
            "avg_tokens": mean(token_values) if token_values else 0.0,
            "avg_duration": (mean(latency_values) / 1000) if latency_values else 0.0,
        }

    v1_metrics = compute_metrics(traces_v1)
    v2_metrics = compute_metrics(traces_v2)

    print("Version Comparison:")
    print(f"\nV1 (n={v1_metrics['total']}):")
    print(f"  Success rate: {v1_metrics['success_rate']*100:.1f}%")
    print(f"  Avg tokens: {v1_metrics['avg_tokens']:,.0f}")
    print(f"  Avg duration: {v1_metrics['avg_duration']:.1f}s")

    print(f"\nV2 (n={v2_metrics['total']}):")
    print(f"  Success rate: {v2_metrics['success_rate']*100:.1f}%")
    print(f"  Avg tokens: {v2_metrics['avg_tokens']:,.0f}")
    print(f"  Avg duration: {v2_metrics['avg_duration']:.1f}s")

    print(f"\nDifferences:")
    print(f"  Success rate: {(v2_metrics['success_rate'] - v1_metrics['success_rate'])*100:+.1f}%")
    print(f"  Tokens: {v2_metrics['avg_tokens'] - v1_metrics['avg_tokens']:+,.0f}")
    print(f"  Duration: {v2_metrics['avg_duration'] - v1_metrics['avg_duration']:+.1f}s")
```

## Best Practices

1. **Start with high-level stats** before deep-diving individual traces
2. **Compare groups** (passed/failed, v1/v2, before/after) for context
3. **Look for patterns** across multiple traces, not one-off issues
4. **Categorize systematically** using consistent frameworks
5. **Track metrics over time** to spot trends and regressions
6. **Use parallel analysis** for large datasets (>100 traces)
7. **Document findings** as you go - insights are easy to forget
