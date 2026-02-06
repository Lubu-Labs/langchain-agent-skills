#!/usr/bin/env python3
"""
Inspect LangGraph checkpoints for debugging state issues.

This script uses LangGraph's official checkpointer API (SqliteSaver) to inspect:
- State evolution over time
- Checkpoint metadata
- State field values at specific points
- Checkpoint history by thread

Usage:
    uv run inspect_checkpoints.py <checkpoint_path> [options]

Examples:
    # Inspect latest checkpoints across threads
    uv run inspect_checkpoints.py ./checkpoints.db

    # Inspect specific checkpoint (optionally provide thread-id for direct lookup)
    uv run inspect_checkpoints.py ./checkpoints.db --checkpoint-id abc123 --thread-id thread-1

    # Show checkpoint history for a thread
    uv run inspect_checkpoints.py ./checkpoints.db --history --thread-id thread-1
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from langgraph.checkpoint.sqlite import SqliteSaver


class CheckpointInspector:
    """Inspector for LangGraph SQLite checkpoints."""

    def __init__(self, checkpoint_path: Path):
        self.checkpoint_path = checkpoint_path
        self.db_path = self._resolve_sqlite_db(checkpoint_path)
        self.backend_type = "sqlite" if self.db_path else "unknown"

    @staticmethod
    def _resolve_sqlite_db(path: Path) -> Optional[Path]:
        """Resolve supported checkpoint path formats."""
        if path.is_file():
            return path

        db_file = path / "checkpoints.db"
        if db_file.exists():
            return db_file

        return None

    def _require_sqlite(self) -> None:
        if SqliteSaver is None:
            raise RuntimeError(
                "Missing dependency: install `langgraph` and `langgraph-checkpoint-sqlite` "
                "to inspect SQLite checkpoints."
            )
        if not self.db_path:
            raise RuntimeError("Could not resolve SQLite checkpoint database path.")

    def _build_config(self, thread_id: Optional[str]) -> Optional[Dict[str, Any]]:
        if not thread_id:
            return None
        return {"configurable": {"thread_id": thread_id}}

    def _to_checkpoint_summary(self, item: Any) -> Dict[str, Any]:
        configurable = (item.config or {}).get("configurable", {})
        parent_configurable = (item.parent_config or {}).get("configurable", {})
        checkpoint_obj = item.checkpoint or {}

        return {
            "thread_id": configurable.get("thread_id"),
            "checkpoint_id": configurable.get("checkpoint_id"),
            "parent_checkpoint_id": parent_configurable.get("checkpoint_id"),
            "created_at": checkpoint_obj.get("ts"),
            "metadata": item.metadata or {},
        }

    def list_checkpoints(
        self, thread_id: Optional[str] = None, limit: int = 10
    ) -> List[Dict[str, Any]]:
        """List available checkpoints."""
        self._require_sqlite()
        config = self._build_config(thread_id)

        with SqliteSaver.from_conn_string(str(self.db_path)) as checkpointer:
            tuples = list(checkpointer.list(config, limit=limit))

        return [self._to_checkpoint_summary(item) for item in tuples]

    def _find_checkpoint_tuple(
        self, checkpoint_id: str, thread_id: Optional[str]
    ) -> Optional[Any]:
        """Find a checkpoint tuple by id, optionally scoped to thread_id."""
        self._require_sqlite()

        with SqliteSaver.from_conn_string(str(self.db_path)) as checkpointer:
            if thread_id:
                config = {
                    "configurable": {
                        "thread_id": thread_id,
                        "checkpoint_id": checkpoint_id,
                    }
                }
                return checkpointer.get_tuple(config)

            # Without thread_id, scan all checkpoints for the first matching id.
            for item in checkpointer.list(None, limit=None):
                configurable = (item.config or {}).get("configurable", {})
                if configurable.get("checkpoint_id") == checkpoint_id:
                    return item

        return None

    def get_checkpoint(
        self, checkpoint_id: str, thread_id: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Get specific checkpoint by ID."""
        item = self._find_checkpoint_tuple(checkpoint_id, thread_id)
        if item is None:
            return None

        summary = self._to_checkpoint_summary(item)
        checkpoint_obj = item.checkpoint or {}
        summary["checkpoint"] = checkpoint_obj.get("channel_values", {})
        return summary

    def get_checkpoint_history(
        self, thread_id: str, limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Get checkpoint history for a thread."""
        self._require_sqlite()
        config = {"configurable": {"thread_id": thread_id}}

        with SqliteSaver.from_conn_string(str(self.db_path)) as checkpointer:
            tuples = list(checkpointer.list(config, limit=limit))

        history = []
        for item in tuples:
            summary = self._to_checkpoint_summary(item)
            checkpoint_obj = item.checkpoint or {}
            summary["checkpoint"] = checkpoint_obj.get("channel_values", {})
            history.append(summary)

        return history

    def print_checkpoint_list(self, checkpoints: List[Dict[str, Any]]) -> None:
        """Pretty print checkpoint list."""
        if not checkpoints:
            print("No checkpoints found.")
            return

        print(f"\n{'='*80}")
        print("Checkpoints")
        print(f"{'='*80}\n")

        for i, cp in enumerate(checkpoints, 1):
            print(f"{i}. Checkpoint ID: {cp['checkpoint_id']}")
            print(f"   Thread ID: {cp['thread_id']}")
            print(f"   Parent: {cp['parent_checkpoint_id'] or 'None'}")
            print(f"   Created: {cp['created_at']}")
            if cp.get("metadata"):
                print(f"   Metadata: {json.dumps(cp['metadata'], indent=6)}")
            print()

    def print_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """Pretty print checkpoint details."""
        print(f"\n{'='*80}")
        print("Checkpoint Details")
        print(f"{'='*80}\n")

        print(f"Checkpoint ID: {checkpoint['checkpoint_id']}")
        print(f"Thread ID: {checkpoint['thread_id']}")
        print(f"Parent: {checkpoint['parent_checkpoint_id'] or 'None'}")
        print(f"Created: {checkpoint['created_at']}\n")

        if checkpoint.get("metadata"):
            print("Metadata:")
            print(json.dumps(checkpoint["metadata"], indent=2))
            print()

        if checkpoint.get("checkpoint"):
            print("State:")
            print(json.dumps(checkpoint["checkpoint"], indent=2))
            print()

    def print_history(self, history: List[Dict[str, Any]]) -> None:
        """Pretty print checkpoint history."""
        if not history:
            print("No checkpoint history found.")
            return

        print(f"\n{'='*80}")
        print("Checkpoint History")
        print(f"{'='*80}\n")

        for i, cp in enumerate(history, 1):
            print(f"{i}. Checkpoint ID: {cp['checkpoint_id']}")
            print(f"   Parent: {cp['parent_checkpoint_id'] or 'None'}")
            print(f"   Created: {cp['created_at']}")

            if cp.get("metadata"):
                print(f"   Metadata: {json.dumps(cp['metadata'], indent=6)}")

            if cp.get("checkpoint"):
                print("   State snapshot:")
                state = cp["checkpoint"]
                keys = list(state.keys())[:5]
                for key in keys:
                    value = state[key]
                    if isinstance(value, (list, dict)):
                        print(f"     {key}: {type(value).__name__} (length: {len(value)})")
                    else:
                        value_str = str(value)
                        if len(value_str) > 50:
                            value_str = value_str[:50] + "..."
                        print(f"     {key}: {value_str}")
                if len(keys) < len(state):
                    print(f"     ... and {len(state) - len(keys)} more fields")

            print()


def main():
    parser = argparse.ArgumentParser(
        description="Inspect LangGraph checkpoints for debugging"
    )
    parser.add_argument(
        "checkpoint_path",
        type=Path,
        help="Path to SQLite checkpoint DB, or a directory containing checkpoints.db",
    )
    parser.add_argument(
        "--checkpoint-id",
        help="Specific checkpoint ID to inspect",
    )
    parser.add_argument(
        "--thread-id",
        help="Filter by thread ID",
    )
    parser.add_argument(
        "--history",
        action="store_true",
        help="Show checkpoint history for thread",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Maximum number of checkpoints to list (default: 10)",
    )

    args = parser.parse_args()

    if not args.checkpoint_path.exists():
        print(f"Error: Checkpoint path not found: {args.checkpoint_path}")
        sys.exit(1)

    inspector = CheckpointInspector(args.checkpoint_path)

    if inspector.backend_type == "unknown":
        print(f"Error: Could not detect SQLite checkpoint DB in {args.checkpoint_path}")
        print("Supported input: direct SQLite DB path, or directory with checkpoints.db")
        sys.exit(1)

    try:
        print(f"Using {inspector.backend_type} backend")

        if args.checkpoint_id:
            checkpoint = inspector.get_checkpoint(args.checkpoint_id, args.thread_id)
            if checkpoint:
                inspector.print_checkpoint(checkpoint)
            else:
                print(f"Checkpoint not found: {args.checkpoint_id}")
                sys.exit(1)

        elif args.history:
            if not args.thread_id:
                print("Error: --history requires --thread-id")
                sys.exit(1)

            history = inspector.get_checkpoint_history(args.thread_id, limit=args.limit)
            inspector.print_history(history)

        else:
            checkpoints = inspector.list_checkpoints(args.thread_id, args.limit)
            inspector.print_checkpoint_list(checkpoints)
    except RuntimeError as exc:
        print(f"Error: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
