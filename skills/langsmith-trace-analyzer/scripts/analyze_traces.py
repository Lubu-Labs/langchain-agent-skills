#!/usr/bin/env python3
"""
Analyze downloaded LangSmith traces to generate insights and reports.

Features:
- Statistical analysis (message counts, tokens, duration)
- Pattern detection (common errors, repeated issues)
- Comparison (passed vs failed traces)
- Custom categorization

Usage:
    # Analyze all traces in directory
    uv run skills/langsmith-trace-analyzer/scripts/analyze_traces.py ./langsmith-traces --output analysis_report.md

    # Focus on failed traces only
    uv run skills/langsmith-trace-analyzer/scripts/analyze_traces.py ./langsmith-traces/by-outcome/failed --output failed_analysis.md

    # Compare passed vs failed
    uv run skills/langsmith-trace-analyzer/scripts/analyze_traces.py ./langsmith-traces --compare --output comparison.md
"""

import argparse
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from statistics import mean, median
from typing import Any, Dict, List, Optional


def load_trace(path: Path) -> Any:
    """Load a single trace JSON file."""
    with open(path) as f:
        return json.load(f)


def _parse_iso_datetime(value: str) -> Optional[datetime]:
    """Parse ISO datetime strings, including trailing Z timezone."""
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _extract_messages(trace: Any) -> List[Dict[str, Any]]:
    """
    Extract messages from supported trace payload shapes.

    Supported forms:
    - {"messages": [...]}
    - {"inputs": {"messages": [...]}}
    - [...]  # raw array output from langsmith-fetch
    """
    if isinstance(trace, list):
        return [m for m in trace if isinstance(m, dict)]

    if not isinstance(trace, dict):
        return []

    messages = trace.get("messages")
    if isinstance(messages, list):
        return [m for m in messages if isinstance(m, dict)]

    inputs = trace.get("inputs")
    if isinstance(inputs, dict):
        input_messages = inputs.get("messages")
        if isinstance(input_messages, list):
            return [m for m in input_messages if isinstance(m, dict)]

    return []


def _extract_metadata(trace: Dict[str, Any]) -> Dict[str, Any]:
    """Extract metadata from common run/trace fields."""
    metadata = trace.get("metadata")
    if isinstance(metadata, dict):
        return metadata

    extra = trace.get("extra")
    if isinstance(extra, dict):
        extra_metadata = extra.get("metadata")
        if isinstance(extra_metadata, dict):
            return extra_metadata

    return {}


def _extract_duration_ms(trace: Dict[str, Any], metadata: Dict[str, Any]) -> float:
    """Extract duration from duration_ms, latency, or start/end timestamps."""
    duration_ms = metadata.get("duration_ms")
    if isinstance(duration_ms, (int, float)):
        return float(duration_ms)

    latency = trace.get("latency")
    if isinstance(latency, (int, float)):
        return float(latency) * 1000

    start_time = _parse_iso_datetime(str(trace.get("start_time", "")))
    end_time = _parse_iso_datetime(str(trace.get("end_time", "")))
    if start_time and end_time:
        return max((end_time - start_time).total_seconds() * 1000, 0.0)

    return 0.0


def _extract_tool_calls(messages: List[Dict[str, Any]]) -> List[str]:
    """Extract tool names from messages across common payload shapes."""
    tool_calls: List[str] = []
    for msg in messages:
        calls = msg.get("tool_calls") or msg.get("additional_kwargs", {}).get("tool_calls", [])
        if not isinstance(calls, list):
            continue
        for call in calls:
            if not isinstance(call, dict):
                continue
            name = call.get("name")
            if not name:
                fn = call.get("function")
                if isinstance(fn, dict):
                    name = fn.get("name")
            if name:
                tool_calls.append(str(name))
    return tool_calls


def _safe_mean(values: List[float]) -> float:
    """Return mean for non-empty lists, else 0."""
    return mean(values) if values else 0.0


def extract_trace_stats(trace: Any) -> Dict:
    """Extract key statistics from a trace."""
    if isinstance(trace, list):
        trace_obj: Dict[str, Any] = {"messages": trace}
    elif isinstance(trace, dict):
        trace_obj = trace
    else:
        raise TypeError(f"Unsupported trace payload type: {type(trace).__name__}")

    messages = _extract_messages(trace_obj)
    metadata = _extract_metadata(trace_obj)
    token_usage = metadata.get("token_usage", {}) if isinstance(metadata.get("token_usage"), dict) else {}

    # Count message types
    msg_types = Counter(
        str(msg.get("role") or msg.get("type") or "unknown")
        for msg in messages
        if isinstance(msg, dict)
    )

    # Extract tool calls
    tool_calls = _extract_tool_calls(messages)

    status = trace_obj.get("status") or metadata.get("status")
    if not status and trace_obj.get("error"):
        status = "error"

    return {
        "trace_id": trace_obj.get("trace_id") or trace_obj.get("id"),
        "status": status or "unknown",
        "message_count": len(messages),
        "user_messages": msg_types.get("user", 0),
        "assistant_messages": msg_types.get("assistant", 0),
        "tool_messages": msg_types.get("tool", 0),
        "tool_calls": tool_calls,
        "tool_call_count": len(tool_calls),
        "unique_tools": len(set(tool_calls)),
        "total_tokens": trace_obj.get("total_tokens", token_usage.get("total_tokens", 0)) or 0,
        "prompt_tokens": trace_obj.get("prompt_tokens", token_usage.get("prompt_tokens", 0)) or 0,
        "completion_tokens": trace_obj.get("completion_tokens", token_usage.get("completion_tokens", 0)) or 0,
        "duration_ms": _extract_duration_ms(trace_obj, metadata),
        "custom_metadata": metadata.get("custom_metadata", {}),
    }


def analyze_traces_in_dir(trace_dir: Path) -> Dict:
    """Analyze all traces in a directory."""
    trace_files = list(trace_dir.rglob("*.json"))

    if not trace_files:
        return {"count": 0, "traces": []}

    stats_list = []
    for trace_file in trace_files:
        if trace_file.name == "manifest.json":
            continue

        try:
            trace = load_trace(trace_file)
            stats = extract_trace_stats(trace)
            stats["file"] = str(trace_file.relative_to(trace_dir))
            stats_list.append(stats)
        except (json.JSONDecodeError, KeyError, TypeError, ValueError, AttributeError) as e:
            print(f"Warning: Failed to analyze {trace_file}: {e}", file=sys.stderr)

    return {
        "count": len(stats_list),
        "traces": stats_list,
    }


def generate_summary_stats(traces: List[Dict]) -> Dict:
    """Generate summary statistics across traces."""
    if not traces:
        return {}

    return {
        "total": len(traces),
        "avg_messages": mean(t["message_count"] for t in traces),
        "median_messages": median(t["message_count"] for t in traces),
        "avg_tool_calls": mean(t["tool_call_count"] for t in traces),
        "avg_tokens": _safe_mean([float(t["total_tokens"]) for t in traces if t["total_tokens"] > 0]),
        "avg_duration_sec": _safe_mean([float(t["duration_ms"]) for t in traces if t["duration_ms"] > 0]) / 1000,
        "total_tokens": sum(t["total_tokens"] for t in traces),
        "total_duration_sec": sum(t["duration_ms"] for t in traces) / 1000,
        "tool_usage": Counter([tool for t in traces for tool in t["tool_calls"]]),
        "unique_tools": len(set(tool for t in traces for tool in t["tool_calls"])),
    }


def find_patterns(traces: List[Dict]) -> Dict:
    """Find common patterns across traces."""
    patterns = {
        "high_message_count": [],
        "high_token_usage": [],
        "many_tool_calls": [],
        "repeated_tools": [],
        "quick_failures": [],
    }

    for trace in traces:
        # High message count (>50 messages)
        if trace["message_count"] > 50:
            patterns["high_message_count"].append({
                "trace_id": trace["trace_id"],
                "message_count": trace["message_count"],
            })

        # High token usage (>100k tokens)
        if trace["total_tokens"] > 100000:
            patterns["high_token_usage"].append({
                "trace_id": trace["trace_id"],
                "tokens": trace["total_tokens"],
            })

        # Many tool calls (>20)
        if trace["tool_call_count"] > 20:
            patterns["many_tool_calls"].append({
                "trace_id": trace["trace_id"],
                "tool_calls": trace["tool_call_count"],
            })

        # Repeated same tool (same tool >5 times)
        tool_counts = Counter(trace["tool_calls"])
        for tool, count in tool_counts.items():
            if count > 5:
                patterns["repeated_tools"].append({
                    "trace_id": trace["trace_id"],
                    "tool": tool,
                    "count": count,
                })

        # Quick failures (<3 messages, failed status)
        if trace["message_count"] < 3 and trace["status"] == "error":
            patterns["quick_failures"].append({
                "trace_id": trace["trace_id"],
                "messages": trace["message_count"],
            })

    return patterns


def compare_groups(group_a: List[Dict], group_b: List[Dict], label_a: str, label_b: str) -> Dict:
    """Compare two groups of traces."""
    stats_a = generate_summary_stats(group_a)
    stats_b = generate_summary_stats(group_b)

    return {
        "group_a": {"label": label_a, "stats": stats_a},
        "group_b": {"label": label_b, "stats": stats_b},
        "differences": {
            "message_count": stats_b.get("avg_messages", 0) - stats_a.get("avg_messages", 0),
            "tool_calls": stats_b.get("avg_tool_calls", 0) - stats_a.get("avg_tool_calls", 0),
            "tokens": stats_b.get("avg_tokens", 0) - stats_a.get("avg_tokens", 0),
            "duration": stats_b.get("avg_duration_sec", 0) - stats_a.get("avg_duration_sec", 0),
        }
    }


def format_markdown_report(analysis: Dict, comparison: Optional[Dict] = None) -> str:
    """Format analysis results as Markdown report."""
    lines = [
        "# LangSmith Trace Analysis Report",
        f"\nGenerated: {analysis['timestamp']}",
        f"\n## Overview\n",
        f"- Total traces analyzed: {analysis['summary']['total']}",
        f"- Average messages per trace: {analysis['summary'].get('avg_messages', 0):.1f}",
        f"- Average tool calls per trace: {analysis['summary'].get('avg_tool_calls', 0):.1f}",
        f"- Total tokens: {analysis['summary'].get('total_tokens', 0):,}",
        f"- Total duration: {analysis['summary'].get('total_duration_sec', 0):,.1f}s",
        f"\n## Tool Usage\n",
    ]

    # Top tools
    if analysis['summary'].get('tool_usage'):
        lines.append("### Most Used Tools\n")
        for tool, count in analysis['summary']['tool_usage'].most_common(10):
            lines.append(f"- `{tool}`: {count} times")

    # Patterns
    patterns = analysis.get('patterns', {})
    if any(patterns.values()):
        lines.append("\n## Patterns Detected\n")

        if patterns.get('high_message_count'):
            lines.append(f"\n### High Message Count ({len(patterns['high_message_count'])} traces)")
            lines.append("Traces with >50 messages:\n")
            for item in patterns['high_message_count'][:5]:
                lines.append(f"- `{item['trace_id']}`: {item['message_count']} messages")

        if patterns.get('repeated_tools'):
            lines.append(f"\n### Repeated Tool Usage ({len(patterns['repeated_tools'])} instances)")
            lines.append("Same tool called >5 times:\n")
            for item in patterns['repeated_tools'][:5]:
                lines.append(f"- `{item['trace_id']}`: `{item['tool']}` called {item['count']} times")

        if patterns.get('quick_failures'):
            lines.append(f"\n### Quick Failures ({len(patterns['quick_failures'])} traces)")
            lines.append("Failed with <3 messages (likely infrastructure errors):\n")
            for item in patterns['quick_failures'][:5]:
                lines.append(f"- `{item['trace_id']}`: {item['messages']} messages")

    # Comparison
    if comparison:
        lines.append("\n## Comparison: Passed vs Failed\n")
        group_a = comparison['group_a']
        group_b = comparison['group_b']

        lines.append(f"| Metric | {group_a['label']} | {group_b['label']} | Difference |")
        lines.append("|--------|---------|---------|------------|")

        lines.append(
            f"| Count | {group_a['stats'].get('total', 0)} | "
            f"{group_b['stats'].get('total', 0)} | - |"
        )
        lines.append(
            f"| Avg Messages | {group_a['stats'].get('avg_messages', 0):.1f} | "
            f"{group_b['stats'].get('avg_messages', 0):.1f} | "
            f"{comparison['differences']['message_count']:+.1f} |"
        )
        lines.append(
            f"| Avg Tool Calls | {group_a['stats'].get('avg_tool_calls', 0):.1f} | "
            f"{group_b['stats'].get('avg_tool_calls', 0):.1f} | "
            f"{comparison['differences']['tool_calls']:+.1f} |"
        )
        lines.append(
            f"| Avg Tokens | {group_a['stats'].get('avg_tokens', 0):,.0f} | "
            f"{group_b['stats'].get('avg_tokens', 0):,.0f} | "
            f"{comparison['differences']['tokens']:+,.0f} |"
        )
        lines.append(
            f"| Avg Duration (s) | {group_a['stats'].get('avg_duration_sec', 0):.1f} | "
            f"{group_b['stats'].get('avg_duration_sec', 0):.1f} | "
            f"{comparison['differences']['duration']:+.1f} |"
        )

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze LangSmith traces",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument("trace_dir", help="Directory containing trace JSON files")
    parser.add_argument("--output", "-o", help="Output file (markdown)")
    parser.add_argument("--compare", action="store_true", help="Compare passed vs failed")
    parser.add_argument("--json", action="store_true", help="Output as JSON instead of markdown")

    args = parser.parse_args()

    trace_dir = Path(args.trace_dir)
    if not trace_dir.exists():
        print(f"Error: Directory not found: {trace_dir}")
        sys.exit(1)

    print(f"Analyzing traces in: {trace_dir}")

    # Analyze traces
    analysis = analyze_traces_in_dir(trace_dir)
    print(f"Found {analysis['count']} traces")

    if analysis['count'] == 0:
        print("No traces to analyze")
        return 0

    results = {
        "timestamp": datetime.now().isoformat(),
        "trace_directory": str(trace_dir),
        "summary": generate_summary_stats(analysis['traces']),
        "patterns": find_patterns(analysis['traces']),
    }

    # Comparison if requested
    comparison = None
    if args.compare:
        passed_dir = trace_dir / "by-outcome" / "passed"
        failed_dir = trace_dir / "by-outcome" / "failed"

        if passed_dir.exists() and failed_dir.exists():
            print("Comparing passed vs failed traces...")
            passed = analyze_traces_in_dir(passed_dir)
            failed = analyze_traces_in_dir(failed_dir)

            comparison = compare_groups(
                passed['traces'],
                failed['traces'],
                f"Passed ({passed['count']})",
                f"Failed ({failed['count']})"
            )
            results["comparison"] = comparison

    # Output results
    if args.json:
        output = json.dumps(results, indent=2)
    else:
        output = format_markdown_report(results, comparison)

    if args.output:
        with open(args.output, "w") as f:
            f.write(output)
        print(f"Report written to: {args.output}")
    else:
        print("\n" + "=" * 60)
        print(output)

    return 0


if __name__ == "__main__":
    sys.exit(main())
