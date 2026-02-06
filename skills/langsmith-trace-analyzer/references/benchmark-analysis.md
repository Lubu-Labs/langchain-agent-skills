# Benchmark Analysis Guide

Guide for analyzing agent benchmark runs (like Terminal Bench/Harbor) using LangSmith traces.

## Understanding Benchmark Structure

### Terminology

| Term | Definition | Example |
|------|------------|---------|
| **Benchmark** | A suite of tasks evaluating agent capabilities | Terminal Bench, Harbor |
| **Task** | A specific problem to solve | "crack-7z-hash", "build-pov-ray" |
| **Trial** | One attempt at a task | `crack-7z-hash__Lu7oSkD` |
| **Job** | Complete benchmark run (all tasks × N trials) | job_id in metadata |
| **Reward** | Task outcome | 1.0 (pass) or 0.0 (fail) |

### Data Sources

After a benchmark run:
1. **result.json** - Aggregated statistics and outcomes
2. **Trial directories** - Local logs and outputs
3. **LangSmith traces** - Full conversation history

> Compatibility note: LangSmith run status is a top-level field (`status`) in the run data format. Some benchmark export pipelines also mirror status under metadata; examples below support both.

## Download Workflow

### Step 1: Get Job ID from result.json

```python
import json

with open("result.json") as f:
    result = json.load(f)

job_id = result["id"]
print(f"Job ID: {job_id}")
```

### Step 2: Query LangSmith for Traces

```bash
# Using download_traces.py script
uv run skills/langsmith-trace-analyzer/scripts/download_traces.py \
  --project "my-benchmark-project" \
  --filter "and(eq(metadata_key, \"job_id\"), eq(metadata_value, \"${job_id}\"))" \
  --output ./traces \
  --organize
```

### Step 3: Organize by Outcome

The script automatically organizes traces:

```
traces/
├── manifest.json
└── by-outcome/
    ├── passed/           # reward=1.0
    ├── failed/           # reward=0.0
    └── errors/
        ├── GraphRecursionError/
        ├── TimeoutError/
        └── DaytonaError/
```

## Analysis Workflow

### Phase 1: High-Level Statistics

```python
import json
from pathlib import Path
from collections import Counter

traces_dir = Path("./traces/by-outcome")

# Count outcomes
passed = list((traces_dir / "passed").glob("*.json"))
failed = list((traces_dir / "failed").glob("*.json"))
errors = list((traces_dir / "errors").rglob("*.json"))

total = len(passed) + len(failed) + len(errors)

print(f"Benchmark Results:")
print(f"  Total trials: {total}")
print(f"  Passed: {len(passed)} ({len(passed)/total*100:.1f}%)")
print(f"  Failed: {len(failed)} ({len(failed)/total*100:.1f}%)")
print(f"  Errors: {len(errors)} ({len(errors)/total*100:.1f}%)")

# Error breakdown
error_types = Counter()
for error_dir in (traces_dir / "errors").iterdir():
    if error_dir.is_dir():
        error_types[error_dir.name] = len(list(error_dir.glob("*.json")))

print(f"\nError breakdown:")
for error_type, count in error_types.most_common():
    print(f"  {error_type}: {count}")
```

### Phase 2: Per-Task Analysis

```python
def analyze_by_task(traces_dir: Path):
    """Analyze success rate per task."""
    from collections import defaultdict

    by_task = defaultdict(lambda: {"passed": 0, "failed": 0, "errors": 0})

    # Extract task name from trial name: task__id -> task
    def get_task(trial_name):
        return trial_name.rsplit("__", 1)[0]

    for outcome in ["passed", "failed"]:
        for trace_file in (traces_dir / outcome).glob("*.json"):
            task = get_task(trace_file.stem)
            by_task[task][outcome] += 1

    for error_dir in (traces_dir / "errors").rglob("*.json"):
        if error_dir.is_file():
            task = get_task(error_dir.stem)
            by_task[task]["errors"] += 1

    # Report
    print("\nPer-Task Results:")
    for task in sorted(by_task.keys()):
        stats = by_task[task]
        total = stats["passed"] + stats["failed"] + stats["errors"]
        pass_rate = stats["passed"] / total * 100 if total > 0 else 0

        status = "✅" if stats["passed"] == total else "⚠️" if stats["passed"] > 0 else "❌"
        print(f"{status} {task}: {stats['passed']}/{total} passed ({pass_rate:.0f}%)")
```

### Phase 3: Failure Deep Dive

Focus on systematic failures (0/N passed):

```python
def find_systematic_failures(traces_dir: Path):
    """Find tasks that fail consistently across all trials."""
    from collections import defaultdict

    by_task = defaultdict(list)

    # Collect all outcomes by task
    for outcome in ["passed", "failed"]:
        for trace_file in (traces_dir / outcome).glob("*.json"):
            task = trace_file.stem.rsplit("__", 1)[0]
            by_task[task].append(outcome)

    # Find tasks with 0 passes
    systematic_failures = []
    for task, outcomes in by_task.items():
        if "passed" not in outcomes and len(outcomes) >= 2:
            systematic_failures.append({
                "task": task,
                "trials": len(outcomes),
                "outcomes": Counter(outcomes)
            })

    print(f"\nSystematic Failures ({len(systematic_failures)} tasks):")
    for item in sorted(systematic_failures, key=lambda x: -x["trials"]):
        print(f"  {item['task']}: {dict(item['outcomes'])}")

    return systematic_failures
```

## Categorization Framework

### Error Categories for Benchmarks

```python
def categorize_benchmark_failure(trace: Dict) -> tuple[str, str]:
    """
    Categorize benchmark failure into actionable categories.

    Returns (category, subcategory)
    """
    messages = trace.get("messages", [])
    status = trace.get("status") or trace.get("metadata", {}).get("status")
    last_message = str(messages[-1]) if messages else ""

    # Infrastructure errors
    if len(messages) < 3 or status == "error":
        last_content = " ".join(str(m.get("content", "")) for m in messages[-3:])

        if "DaytonaError" in last_content:
            return "infrastructure", "sandbox_failure"
        elif "BadRequestError" in last_content:
            return "infrastructure", "api_error"

    # Resource limit errors
    if "GraphRecursionError" in last_message:
        # Check if agent was stuck in loop or making progress
        tool_calls = []
        for msg in messages:
            if msg.get("tool_calls"):
                tool_calls.extend([tc["name"] for tc in msg["tool_calls"]])

        # If same tool called >5 times consecutively, it's a loop
        for i in range(len(tool_calls) - 5):
            if len(set(tool_calls[i:i+5])) == 1:
                return "resource_limit", "stuck_in_loop"

        return "resource_limit", "recursion_limit_progressing"

    # Task failures - need manual review
    return "task_failure", "needs_review"
```

## Comparison Patterns

### Compare Multiple Benchmark Runs

```python
def compare_benchmark_runs(run1_dir: Path, run2_dir: Path):
    """Compare two benchmark runs (e.g., different models or versions)."""

    def get_stats(run_dir):
        passed = len(list((run_dir / "passed").glob("*.json")))
        failed = len(list((run_dir / "failed").glob("*.json")))
        errors = len(list((run_dir / "errors").rglob("*.json")))
        total = passed + failed + errors

        return {
            "total": total,
            "passed": passed,
            "failed": failed,
            "errors": errors,
            "pass_rate": passed / total if total > 0 else 0
        }

    stats1 = get_stats(run1_dir)
    stats2 = get_stats(run2_dir)

    print("Benchmark Comparison:")
    print(f"\nRun 1:")
    print(f"  Pass rate: {stats1['pass_rate']*100:.1f}% ({stats1['passed']}/{stats1['total']})")
    print(f"  Errors: {stats1['errors']}")

    print(f"\nRun 2:")
    print(f"  Pass rate: {stats2['pass_rate']*100:.1f}% ({stats2['passed']}/{stats2['total']})")
    print(f"  Errors: {stats2['errors']}")

    print(f"\nDifference:")
    print(f"  Pass rate: {(stats2['pass_rate'] - stats1['pass_rate'])*100:+.1f}%")
    print(f"  Net improvement: {stats2['passed'] - stats1['passed']:+d} tasks")
```

### Find Regressions

```python
def find_regressions(baseline_dir: Path, new_dir: Path):
    """Find tasks that passed in baseline but failed in new run."""

    def get_passed_tasks(run_dir):
        tasks = set()
        for trace_file in (run_dir / "passed").glob("*.json"):
            task = trace_file.stem.rsplit("__", 1)[0]
            tasks.add(task)
        return tasks

    baseline_passed = get_passed_tasks(baseline_dir)
    new_passed = get_passed_tasks(new_dir)

    regressions = baseline_passed - new_passed
    improvements = new_passed - baseline_passed

    print(f"Regressions ({len(regressions)} tasks):")
    for task in sorted(regressions):
        print(f"  ❌ {task}")

    print(f"\nImprovements ({len(improvements)} tasks):")
    for task in sorted(improvements):
        print(f"  ✅ {task}")
```

## Report Template

```python
def generate_benchmark_report(traces_dir: Path, output_file: str):
    """Generate a comprehensive benchmark report."""

    # Collect data
    passed = list((traces_dir / "passed").glob("*.json"))
    failed = list((traces_dir / "failed").glob("*.json"))
    errors = list((traces_dir / "errors").rglob("*.json"))
    total = len(passed) + len(failed) + len(errors)

    # Load manifest for metadata
    with open(traces_dir.parent / "manifest.json") as f:
        manifest = json.load(f)

    # Generate report
    report = f"""# Benchmark Analysis Report

Generated: {manifest['created_at']}
Project: {manifest.get('project', 'N/A')}

## Summary

- **Total Trials**: {total}
- **Pass Rate**: {len(passed)/total*100:.1f}% ({len(passed)}/{total})
- **Failures**: {len(failed)} ({len(failed)/total*100:.1f}%)
- **Errors**: {len(errors)} ({len(errors)/total*100:.1f}%)

## Error Breakdown

"""

    # Add error breakdown
    error_types = Counter()
    for error_dir in (traces_dir / "errors").iterdir():
        if error_dir.is_dir():
            count = len(list(error_dir.glob("*.json")))
            error_types[error_dir.name] = count
            report += f"- {error_dir.name}: {count}\n"

    # Per-task analysis
    report += "\n## Per-Task Results\n\n"

    from collections import defaultdict

    by_task = defaultdict(lambda: {"passed": 0, "failed": 0, "errors": 0})

    for outcome in ["passed", "failed"]:
        for trace_file in (traces_dir / outcome).glob("*.json"):
            task = trace_file.stem.rsplit("__", 1)[0]
            by_task[task][outcome] += 1

    # Sort by pass rate
    task_results = []
    for task, stats in by_task.items():
        task_total = sum(stats.values())
        pass_rate = stats["passed"] / task_total if task_total > 0 else 0
        task_results.append((task, stats, pass_rate))

    task_results.sort(key=lambda x: x[2])  # Sort by pass rate

    report += "| Task | Passed | Failed | Errors | Pass Rate |\n"
    report += "|------|--------|--------|--------|----------|\n"

    for task, stats, pass_rate in task_results:
        report += f"| {task} | {stats['passed']} | {stats['failed']} | {stats['errors']} | {pass_rate*100:.0f}% |\n"

    # Write report
    with open(output_file, "w") as f:
        f.write(report)

    print(f"Report written to: {output_file}")
```

## Best Practices

1. **Download all traces first** before analyzing (avoids rate limits)
2. **Organize by outcome** automatically using scripts
3. **Start with high-level stats** before diving into individual traces
4. **Focus on systematic failures** (0/N passed) for highest impact
5. **Compare versions** to measure improvements
6. **Categorize errors** to prioritize fixes (infrastructure vs agent issues)
7. **Track over time** to spot trends and regressions
