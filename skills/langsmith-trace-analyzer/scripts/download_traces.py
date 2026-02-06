#!/usr/bin/env python3
"""
Download traces from LangSmith with flexible filtering and organization.

Features:
- Query by trace ID, job_id, metadata, status, or time range
- Organize by outcome (passed/failed/error) automatically
- Create manifest with trace metadata
- Resume interrupted downloads

Usage:
    # Download specific traces by ID
    uv run skills/langsmith-trace-analyzer/scripts/download_traces.py --trace-ids id1 id2 id3 --output ./traces

    # Download by job_id metadata
    uv run skills/langsmith-trace-analyzer/scripts/download_traces.py --project my-project --filter 'job_id=abc123' --output ./traces

    # Download recent traces
    uv run skills/langsmith-trace-analyzer/scripts/download_traces.py --project my-project --last-hours 24 --limit 50 --output ./traces

    # Download error traces only
    uv run skills/langsmith-trace-analyzer/scripts/download_traces.py --project my-project --status error --output ./traces
"""

import argparse
import json
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


def ensure_langsmith_fetch():
    """Check if langsmith-fetch is installed."""
    try:
        subprocess.run(["langsmith-fetch", "--help"], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Error: langsmith-fetch not installed. Install with:")
        print("  uv pip install langsmith-fetch")
        sys.exit(1)


def ensure_langsmith_sdk():
    """Check if langsmith SDK is installed."""
    try:
        import langsmith
        return True
    except ImportError:
        print("Error: langsmith SDK not installed. Install with:")
        print("  uv pip install langsmith")
        sys.exit(1)


def query_traces(
    project_name: str,
    filter_query: Optional[str] = None,
    status: Optional[str] = None,
    since_hours: Optional[int] = None,
    limit: Optional[int] = None
) -> List[Dict]:
    """Query LangSmith for traces matching criteria."""
    from langsmith import Client

    client = Client()
    filters = []

    if filter_query:
        filters.append(filter_query)

    if status:
        filters.append(f'eq(status, "{status}")')

    filter_str = f'and({", ".join(filters)})' if len(filters) > 1 else filters[0] if filters else None

    kwargs = {
        "project_name": project_name,
        "is_root": True,
    }

    if filter_str:
        kwargs["filter"] = filter_str

    if since_hours is not None:
        kwargs["start_time"] = datetime.now(timezone.utc) - timedelta(hours=since_hours)

    traces = []
    for run in client.list_runs(**kwargs):
        traces.append({
            "trace_id": str(run.id),
            "status": run.status,
            "metadata": run.metadata or {},
            "start_time": run.start_time.isoformat() if run.start_time else None,
            "end_time": run.end_time.isoformat() if run.end_time else None,
        })

        if limit and len(traces) >= limit:
            break

    return traces


def fetch_single_trace(trace_id: str, output_path: Path, retries: int = 3) -> tuple[bool, Optional[str]]:
    """Fetch a single trace using langsmith-fetch CLI."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "langsmith-fetch", "trace", trace_id,
        "--format", "raw",
        "--include-metadata",
        "--file", str(output_path)
    ]
    fallback_cmd = [
        "langsmith-fetch", "trace", trace_id,
        "--format", "raw",
        "--file", str(output_path)
    ]

    for attempt in range(retries):
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            if result.returncode != 0 and (
                "include-metadata" in (result.stderr or "")
                and ("No such option" in (result.stderr or "") or "unknown option" in (result.stderr or "").lower())
            ):
                # Backward/forward compatibility across langsmith-fetch CLI versions.
                result = subprocess.run(fallback_cmd, capture_output=True, text=True, timeout=60)

            if result.returncode == 0 and output_path.exists():
                # Verify valid JSON
                with open(output_path) as f:
                    json.load(f)
                return True, None

            error = result.stderr or "Unknown error"

        except subprocess.TimeoutExpired:
            error = "Timeout"
        except json.JSONDecodeError as e:
            error = f"Invalid JSON: {e}"
        except Exception as e:
            error = str(e)

        if attempt < retries - 1:
            time.sleep(2 ** attempt)

    return False, error


def _extract_trace_metadata(trace_data: Dict[str, Any]) -> Dict[str, Any]:
    """Extract metadata from supported trace payload shapes."""
    metadata = trace_data.get("metadata")
    if isinstance(metadata, dict):
        return metadata

    extra = trace_data.get("extra")
    if isinstance(extra, dict):
        extra_metadata = extra.get("metadata")
        if isinstance(extra_metadata, dict):
            return extra_metadata

    return {}


def _extract_trace_messages(trace_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract messages from trace payload."""
    messages = trace_data.get("messages")
    if isinstance(messages, list):
        return [m for m in messages if isinstance(m, dict)]

    inputs = trace_data.get("inputs")
    if isinstance(inputs, dict):
        input_messages = inputs.get("messages")
        if isinstance(input_messages, list):
            return [m for m in input_messages if isinstance(m, dict)]

    return []


def categorize_trace(trace_data: Dict[str, Any]) -> tuple[str, Optional[str]]:
    """Categorize trace into passed/failed/error with subcategory."""
    metadata = _extract_trace_metadata(trace_data)
    status = trace_data.get("status") or metadata.get("status")
    if not status and trace_data.get("error"):
        status = "error"
    status = str(status or "unknown")

    if status == "success":
        # Check for task outcome if available
        custom_meta = metadata.get("custom_metadata", {}) if isinstance(metadata.get("custom_metadata"), dict) else {}
        reward = custom_meta.get("reward", metadata.get("reward"))
        if reward in (1, 1.0, True):
            return "passed", None
        if reward in (0, 0.0, False):
            return "failed", None
        return "passed", None

    if status == "error":
        # Try to extract error type from error field and messages.
        messages = _extract_trace_messages(trace_data)
        error_blob = str(trace_data.get("error", ""))
        if "GraphRecursionError" in error_blob:
            return "error", "GraphRecursionError"
        if "TimeoutError" in error_blob or "AgentTimeoutError" in error_blob:
            return "error", "TimeoutError"
        if "DaytonaError" in error_blob:
            return "error", "DaytonaError"

        error_type = "unknown"
        for msg in reversed(messages):
            content = str(msg.get("content", ""))
            if "GraphRecursionError" in content:
                error_type = "GraphRecursionError"
                break
            if "TimeoutError" in content or "AgentTimeoutError" in content:
                error_type = "TimeoutError"
                break
            if "DaytonaError" in content:
                error_type = "DaytonaError"
                break

        return "error", error_type

    return "unknown", None


def find_existing_trace_path(output_dir: Path, trace_id: str) -> Optional[Path]:
    """Locate an already-downloaded trace by id."""
    candidates = [
        output_dir / "temp" / f"{trace_id}.json",
        output_dir / f"{trace_id}.json",
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    by_outcome_dir = output_dir / "by-outcome"
    if by_outcome_dir.exists():
        for path in by_outcome_dir.rglob(f"{trace_id}.json"):
            return path

    return None


def main():
    parser = argparse.ArgumentParser(
        description="Download traces from LangSmith with flexible filtering",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument("--project", help="LangSmith project name")
    parser.add_argument("--trace-ids", nargs="+", help="Specific trace IDs to download")
    parser.add_argument("--filter", help="Custom filter query (e.g., 'job_id=abc123')")
    parser.add_argument("--status", choices=["success", "error"], help="Filter by status")
    parser.add_argument("--last-hours", type=int, help="Download traces from last N hours")
    parser.add_argument("--limit", type=int, help="Maximum traces to download")
    parser.add_argument("--output", default="./langsmith-traces", help="Output directory")
    parser.add_argument("--delay", type=float, default=1.0, help="Delay between requests (seconds)")
    parser.add_argument(
        "--organize",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Organize by outcome",
    )

    args = parser.parse_args()

    ensure_langsmith_fetch()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get trace IDs
    if args.trace_ids:
        traces = [{"trace_id": tid, "status": "unknown", "metadata": {}} for tid in args.trace_ids]
        print(f"Downloading {len(traces)} specified traces...")
    else:
        if not args.project:
            print("Error: --project required when not using --trace-ids")
            sys.exit(1)

        ensure_langsmith_sdk()

        # Parse filter if provided
        filter_query = None
        if args.filter:
            if "=" in args.filter:
                key, value = args.filter.split("=", 1)
                filter_query = f'and(eq(metadata_key, "{key}"), eq(metadata_value, "{value}"))'
            else:
                filter_query = args.filter

        print(f"Querying traces from project: {args.project}")
        traces = query_traces(
            args.project,
            filter_query=filter_query,
            status=args.status,
            since_hours=args.last_hours,
            limit=args.limit
        )
        print(f"Found {len(traces)} matching traces")

    if not traces:
        print("No traces to download")
        return 0

    # Download traces
    results = {"success": [], "failed": [], "by_category": defaultdict(int)}

    for i, trace_info in enumerate(traces, 1):
        trace_id = trace_info["trace_id"]

        temp_path = output_dir / "temp" / f"{trace_id}.json" if args.organize else output_dir / f"{trace_id}.json"

        # Skip if already exists anywhere in the output directory structure.
        existing = find_existing_trace_path(output_dir, trace_id)
        if existing is not None:
            relative = existing.relative_to(output_dir)
            print(f"[{i}/{len(traces)}] SKIP: {trace_id} (already exists at {relative})")
            results["success"].append(trace_id)
            continue

        print(f"[{i}/{len(traces)}] Downloading: {trace_id}")

        success, error = fetch_single_trace(trace_id, temp_path)

        if success:
            # Categorize and move to proper location if organizing
            if args.organize:
                with open(temp_path) as f:
                    trace_data = json.load(f)

                category, subcategory = categorize_trace(trace_data)
                results["by_category"][category] += 1

                if subcategory:
                    final_dir = output_dir / "by-outcome" / category / subcategory
                else:
                    final_dir = output_dir / "by-outcome" / category

                final_dir.mkdir(parents=True, exist_ok=True)
                final_path = final_dir / f"{trace_id}.json"

                temp_path.rename(final_path)
                print(f"  -> {final_path.relative_to(output_dir)}")
            else:
                print(f"  -> {temp_path.relative_to(output_dir)}")

            results["success"].append(trace_id)
        else:
            print(f"  FAILED: {error}")
            results["failed"].append({"trace_id": trace_id, "error": error})

        # Rate limiting
        if i < len(traces):
            time.sleep(args.delay)

    # Create manifest
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "project": args.project,
        "filter": args.filter,
        "total_traces": len(traces),
        "downloaded": len(results["success"]),
        "failed": len(results["failed"]),
        "by_category": dict(results["by_category"]),
    }

    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n{'=' * 60}")
    print("DOWNLOAD COMPLETE")
    print(f"{'=' * 60}")
    print(f"Success: {len(results['success'])}")
    print(f"Failed: {len(results['failed'])}")

    if results["by_category"]:
        print("\nBy Category:")
        for cat, count in sorted(results["by_category"].items()):
            print(f"  {cat}: {count}")

    if results["failed"]:
        print("\nFailed downloads:")
        for item in results["failed"][:10]:
            print(f"  - {item['trace_id']}: {item['error']}")

    print(f"\nManifest written to: {manifest_path}")

    return 0 if not results["failed"] else 1


if __name__ == "__main__":
    sys.exit(main())
