#!/usr/bin/env python3
"""
Migrate LangGraph checkpoint state values between schema versions.

This script uses LangGraph's official checkpointer API (SqliteSaver) and can:
- Add new fields with default values
- Remove deprecated fields
- Rename fields
- Transform field values
- Validate migrated state structure

Important behavior:
- Migrations are applied to checkpoint `channel_values`.
- The script writes migrated snapshots using `checkpointer.put(...)` for each
  selected checkpoint config.

Usage:
    uv run migrate_state.py <checkpoint_path> <migration_script>

Migration script format:
    Python file with migrate(old_state: dict) -> dict
"""

import argparse
import copy
import importlib.util
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from langgraph.checkpoint.sqlite import SqliteSaver


class StateMigrator:
    """Migrates LangGraph checkpoint state values."""

    def __init__(
        self,
        checkpoint_path: Path,
        migration_fn: Callable[[Dict[str, Any]], Dict[str, Any]],
        dry_run: bool = False,
    ):
        self.checkpoint_path = checkpoint_path
        self.migration_fn = migration_fn
        self.dry_run = dry_run
        self.db_path = self._resolve_sqlite_db(checkpoint_path)
        self.backend_type = "sqlite" if self.db_path else "unknown"

        self.migrated_count = 0
        self.failed_count = 0
        self.errors: List[str] = []

    @staticmethod
    def _resolve_sqlite_db(path: Path) -> Optional[Path]:
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
                "to run checkpoint migrations."
            )
        if not self.db_path:
            raise RuntimeError("Could not resolve SQLite checkpoint database path.")

    def migrate_all(self, thread_id: Optional[str] = None) -> bool:
        """Migrate all checkpoints (or all checkpoints for a thread)."""
        if self.backend_type != "sqlite":
            print(f"Error: Unsupported backend type: {self.backend_type}")
            return False

        self._require_sqlite()
        config = {"configurable": {"thread_id": thread_id}} if thread_id else None

        with SqliteSaver.from_conn_string(str(self.db_path)) as checkpointer:
            checkpoint_tuples = list(checkpointer.list(config, limit=None))

            total = len(checkpoint_tuples)
            print(f"\nFound {total} checkpoint(s) to migrate")
            if self.dry_run:
                print("(Dry run - no changes will be saved)\n")
            else:
                print("(Migrated checkpoint values will be persisted)\n")

            for i, item in enumerate(checkpoint_tuples, 1):
                configurable = (item.config or {}).get("configurable", {})
                item_thread_id = configurable.get("thread_id", "unknown-thread")
                item_checkpoint_id = configurable.get(
                    "checkpoint_id", "unknown-checkpoint"
                )
                print(f"[{i}/{total}] Migrating checkpoint {item_checkpoint_id}...")

                try:
                    checkpoint_obj = copy.deepcopy(item.checkpoint or {})
                    old_state = copy.deepcopy(checkpoint_obj.get("channel_values", {}))

                    if not isinstance(old_state, dict):
                        raise ValueError(
                            "Checkpoint channel_values must be a dict for schema migration."
                        )

                    new_state = self.migration_fn(copy.deepcopy(old_state))
                    if not isinstance(new_state, dict):
                        raise ValueError(
                            f"Migration function must return dict, got {type(new_state)}"
                        )

                    self._print_diff(old_state, new_state)

                    if not self.dry_run and old_state != new_state:
                        checkpoint_obj["channel_values"] = new_state
                        new_versions = checkpoint_obj.get("channel_versions", {})
                        new_config = checkpointer.put(
                            item.config,
                            checkpoint_obj,
                            item.metadata or {},
                            new_versions,
                        )
                        new_checkpoint_id = (
                            (new_config or {})
                            .get("configurable", {})
                            .get("checkpoint_id")
                        )
                        print(
                            f"   Persisted checkpoint: {new_checkpoint_id or 'unknown'}"
                        )

                    self.migrated_count += 1
                    print("   Migrated successfully\n")

                except Exception as exc:
                    self.failed_count += 1
                    error_msg = f"Thread {item_thread_id}, checkpoint {item_checkpoint_id}: {exc}"
                    self.errors.append(error_msg)
                    print(f"   Migration failed: {exc}\n")

        self._print_summary()
        return self.failed_count == 0

    def _print_diff(self, old_state: Dict[str, Any], new_state: Dict[str, Any]) -> None:
        """Print differences between old and new state."""
        old_keys = set(old_state.keys())
        new_keys = set(new_state.keys())

        added = new_keys - old_keys
        if added:
            print(f"   + Added fields: {', '.join(sorted(added))}")

        removed = old_keys - new_keys
        if removed:
            print(f"   - Removed fields: {', '.join(sorted(removed))}")

        common = old_keys & new_keys
        modified = [key for key in sorted(common) if old_state[key] != new_state[key]]
        if modified:
            print(f"   ~ Modified fields: {', '.join(modified)}")

        if not added and not removed and not modified:
            print("   = No changes")

    def _print_summary(self) -> None:
        """Print migration summary."""
        print(f"\n{'=' * 60}")
        print("Migration Summary")
        print(f"{'=' * 60}")
        print(f"Total checkpoints processed: {self.migrated_count + self.failed_count}")
        print(f"Migrated: {self.migrated_count}")
        print(f"Failed: {self.failed_count}")

        if self.errors:
            print("\nErrors:")
            for error in self.errors:
                print(f"  - {error}")

        if self.failed_count == 0:
            if self.dry_run:
                print("\nDry run completed successfully (no changes saved)")
            else:
                print("\nMigration completed successfully")
        else:
            print(f"\nMigration completed with {self.failed_count} error(s)")


def load_migration_script(
    script_path: Path,
) -> Callable[[Dict[str, Any]], Dict[str, Any]]:
    """Load migration function from script."""
    if not script_path.exists():
        print(f"Error: Migration script not found: {script_path}")
        sys.exit(1)

    spec = importlib.util.spec_from_file_location("migration", script_path)
    if spec is None or spec.loader is None:
        print(f"Error: Could not load migration script from {script_path}")
        sys.exit(1)

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if not hasattr(module, "migrate"):
        print("Error: Migration script must define `migrate(old_state: dict) -> dict`.")
        sys.exit(1)

    return module.migrate


def main():
    parser = argparse.ArgumentParser(
        description="Migrate LangGraph checkpoint state values between schema versions"
    )
    parser.add_argument(
        "checkpoint_path",
        type=Path,
        help="Path to SQLite checkpoint DB, or directory containing checkpoints.db",
    )
    parser.add_argument(
        "migration_script",
        type=Path,
        help="Path to migration script with migrate(old_state: dict) -> dict",
    )
    parser.add_argument(
        "--thread-id",
        help="Only migrate checkpoints for this thread",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show changes without writing migrated checkpoints",
    )

    args = parser.parse_args()

    if not args.checkpoint_path.exists():
        print(f"Error: Checkpoint path not found: {args.checkpoint_path}")
        sys.exit(1)

    print(f"Loading migration script: {args.migration_script}")
    migration_fn = load_migration_script(args.migration_script)

    print(f"Checkpoint path: {args.checkpoint_path}")
    if args.thread_id:
        print(f"Thread ID filter: {args.thread_id}")

    migrator = StateMigrator(
        args.checkpoint_path,
        migration_fn,
        dry_run=args.dry_run,
    )

    if migrator.backend_type == "unknown":
        print(f"Error: Could not detect SQLite checkpoint DB in {args.checkpoint_path}")
        print(
            "Supported input: direct SQLite DB path, or directory with checkpoints.db"
        )
        sys.exit(1)

    try:
        success = migrator.migrate_all(args.thread_id)
    except RuntimeError as exc:
        print(f"Error: {exc}")
        sys.exit(1)

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
