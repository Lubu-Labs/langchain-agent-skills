#!/usr/bin/env python3
"""
Visualize todo progression from LangSmith trace JSON.

Parse LangSmith trace exports and show how todos evolved during agent execution.
Supports text timeline and Mermaid diagram formats.

Usage:
    uv run visualize_todos.py <trace-file.json>
    uv run visualize_todos.py <trace-file.json> --format mermaid
    uv run visualize_todos.py <trace-file.json> --show-timeline
"""

import argparse
import json
import sys
from typing import Any, Dict, List, Optional
from pathlib import Path


# Status emoji mapping
STATUS_EMOJI = {
    "pending": "⏳",
    "in_progress": "🔄",
    "completed": "✅",
}


def parse_trace(file_path: str) -> Any:
    """Load and validate trace JSON file."""
    path = Path(file_path)

    if not path.exists():
        print(f"❌ Error: File not found: {file_path}", file=sys.stderr)
        sys.exit(1)

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data
    except json.JSONDecodeError as e:
        print(f"❌ Error: Invalid JSON in {file_path}: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error reading file: {e}", file=sys.stderr)
        sys.exit(1)


def _looks_like_run(node: Any) -> bool:
    """Heuristic check for LangSmith run/span objects."""
    if not isinstance(node, dict):
        return False
    run_keys = {
        "id",
        "name",
        "run_type",
        "type",
        "inputs",
        "outputs",
        "start_time",
        "dotted_order",
        "parent_run_id",
        "child_run_ids",
        "child_runs",
    }
    return any(key in node for key in run_keys)


def _collect_runs(trace_data: Any) -> List[Dict[str, Any]]:
    """
    Collect runs from nested and flat export formats.

    Supports:
    - Root run with nested child_runs / children
    - Objects containing "runs"
    - Flat lists of run objects
    """
    runs: List[Dict[str, Any]] = []
    seen: set[str] = set()

    def add_run(run: Dict[str, Any]) -> None:
        run_id = run.get("id")
        if isinstance(run_id, str) and run_id:
            key = f"id:{run_id}"
        else:
            key = f"obj:{id(run)}"

        if key in seen:
            return
        seen.add(key)
        runs.append(run)

    def walk(node: Any) -> None:
        if isinstance(node, list):
            for item in node:
                walk(item)
            return

        if not isinstance(node, dict):
            return

        if _looks_like_run(node):
            add_run(node)

        # Common wrapper keys in trace exports.
        for key in (
            "runs",
            "child_runs",
            "children",
            "trace",
            "traces",
            "data",
            "results",
            "items",
        ):
            value = node.get(key)
            if value is not None:
                walk(value)

    walk(trace_data)
    return runs


def _safe_json_loads(value: Any) -> Optional[Any]:
    """Parse JSON strings safely."""
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def _normalize_todos(candidate: Any) -> Optional[List[Dict[str, str]]]:
    """Validate and normalize todo arrays."""
    if not isinstance(candidate, list):
        return None
    if not candidate:
        return None

    todos: List[Dict[str, str]] = []
    for item in candidate:
        if not isinstance(item, dict):
            return None

        content = item.get("content")
        if not isinstance(content, str) or not content.strip():
            return None

        status = item.get("status", "pending")
        if not isinstance(status, str) or not status:
            status = "pending"

        todos.append({"content": content, "status": status})

    return todos


def _extract_todos_from_payload(
    payload: Any, depth: int = 0
) -> Optional[List[Dict[str, str]]]:
    """Extract todos from tool call payload shapes."""
    if depth > 6:
        return None

    parsed = _safe_json_loads(payload)
    if parsed is not None:
        return _extract_todos_from_payload(parsed, depth + 1)

    normalized = _normalize_todos(payload)
    if normalized:
        return normalized

    if isinstance(payload, dict):
        if "todos" in payload:
            normalized = _normalize_todos(payload.get("todos"))
            if normalized:
                return normalized

        # Common wrappers around tool args in traces.
        for key in (
            "input",
            "inputs",
            "args",
            "arguments",
            "kwargs",
            "payload",
            "tool_input",
            "tool_args",
            "data",
        ):
            if key in payload:
                nested = _extract_todos_from_payload(payload.get(key), depth + 1)
                if nested:
                    return nested

        for value in payload.values():
            nested = _extract_todos_from_payload(value, depth + 1)
            if nested:
                return nested

    if isinstance(payload, list):
        for item in payload:
            nested = _extract_todos_from_payload(item, depth + 1)
            if nested:
                return nested

    return None


def _todo_sort_key(call: Dict[str, Any]) -> tuple[int, str]:
    """Prefer dotted_order, then start_time, then run id."""
    dotted_order = call.get("dotted_order")
    if isinstance(dotted_order, str) and dotted_order:
        return (0, dotted_order)

    start_time = call.get("start_time")
    if isinstance(start_time, str) and start_time:
        return (1, start_time)

    return (2, str(call.get("run_id", "")))


def extract_todos(trace_data: Any) -> List[Dict[str, Any]]:
    """Extract all write_todos calls from LangSmith trace exports."""
    todo_calls: List[Dict[str, Any]] = []
    runs = _collect_runs(trace_data)

    for run in runs:
        name = run.get("name")
        run_type = run.get("run_type") or run.get("type")

        if name != "write_todos":
            continue
        if run_type and run_type != "tool":
            continue

        todos = None
        for candidate in (
            run.get("inputs"),
            run.get("input"),
            run.get("outputs"),
            run.get("output"),
        ):
            todos = _extract_todos_from_payload(candidate)
            if todos:
                break

        if not todos:
            continue

        todo_calls.append(
            {
                "todos": todos,
                "run_id": run.get("id", "unknown"),
                "trace_id": run.get("trace_id"),
                "dotted_order": run.get("dotted_order"),
                "start_time": run.get("start_time"),
            }
        )

    todo_calls.sort(key=_todo_sort_key)
    for step, call in enumerate(todo_calls, start=1):
        call["step"] = step

    return todo_calls


def format_text(todo_calls: List[Dict[str, Any]], show_timeline: bool = False) -> str:
    """Format todo calls as text timeline."""
    if not todo_calls:
        return "No write_todos calls found in trace."

    lines = []
    trace_id = todo_calls[0].get("trace_id") or todo_calls[0].get("run_id", "unknown")
    trace_label = trace_id[:8] if isinstance(trace_id, str) else "unknown"
    lines.append(f"Todo Timeline for trace {trace_label}:\n")

    for i, call in enumerate(todo_calls):
        step = call["step"]
        todos = call["todos"]

        if i == 0:
            lines.append(f"Initial Plan (Step {step}):")
        elif i == len(todo_calls) - 1:
            lines.append(f"\nFinal State (Step {step}):")
        elif show_timeline:
            lines.append(f"\nAfter Step {step}:")
        else:
            continue  # Skip intermediate steps unless --show-timeline

        for todo in todos:
            status = todo.get("status", "pending")
            content = todo.get("content", "No content")
            emoji = STATUS_EMOJI.get(status, "❓")
            lines.append(f"  {emoji} [{status}] {content}")

    return "\n".join(lines)


def format_mermaid(todo_calls: List[Dict[str, Any]]) -> str:
    """Format todo calls as Mermaid state diagram."""
    if not todo_calls:
        return "No write_todos calls found in trace."

    lines = ["stateDiagram-v2"]

    for i, call in enumerate(todo_calls):
        step = call["step"]
        todos = call["todos"]

        state_name = f"Step{step}"
        lines.append(f"    {state_name}: Step {step}")

        for j, todo in enumerate(todos):
            status = todo.get("status", "pending")
            content = todo.get("content", "No content")[:40]  # Truncate long content
            content = content.replace('"', "'")
            emoji = STATUS_EMOJI.get(status, "❓")
            lines.append(f"    {state_name} --> Todo{step}_{j}: {emoji} {content}")

        if i < len(todo_calls) - 1:
            next_step = todo_calls[i + 1]["step"]
            lines.append(f"    Step{step} --> Step{next_step}")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Visualize todo progression from LangSmith trace JSON",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  uv run visualize_todos.py trace.json
  uv run visualize_todos.py trace.json --format mermaid
  uv run visualize_todos.py trace.json --show-timeline
        """,
    )

    parser.add_argument("trace_file", help="Path to LangSmith trace JSON file")

    parser.add_argument(
        "--format",
        choices=["text", "mermaid"],
        default="text",
        help="Output format (default: text)",
    )

    parser.add_argument(
        "--show-timeline",
        action="store_true",
        help="Show all intermediate todo states (text format only)",
    )

    args = parser.parse_args()

    # Parse trace file
    trace_data = parse_trace(args.trace_file)

    # Extract todo calls
    todo_calls = extract_todos(trace_data)

    if not todo_calls:
        print("No write_todos calls found in trace.", file=sys.stderr)
        print("Make sure the trace contains write_todos tool calls.", file=sys.stderr)
        sys.exit(0)

    # Format and print output
    if args.format == "mermaid":
        output = format_mermaid(todo_calls)
    else:
        output = format_text(todo_calls, args.show_timeline)

    print(output)


if __name__ == "__main__":
    main()
