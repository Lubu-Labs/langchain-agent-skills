#!/usr/bin/env python3
"""
Validate langgraph.json configuration file.

Usage:
    uv run validate_langgraph_config.py [path/to/langgraph.json]

Fallback:
    python3 validate_langgraph_config.py [path/to/langgraph.json]

If no path is provided, looks for langgraph.json in the current directory.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


class ConfigValidator:
    """Validator for langgraph.json configuration."""

    def __init__(self, config_path: Path):
        self.config_path = config_path
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.config: Optional[Dict[str, Any]] = None

    def validate(self) -> bool:
        """Run all validation checks. Returns True if valid."""
        if not self._load_config():
            return False

        self._validate_required_fields()
        self._validate_dependencies()
        self._validate_graphs()
        self._validate_optional_fields()

        return len(self.errors) == 0

    def _load_config(self) -> bool:
        """Load and parse the configuration file."""
        if not self.config_path.exists():
            self.errors.append(f"Configuration file not found: {self.config_path}")
            return False

        try:
            with open(self.config_path) as f:
                self.config = json.load(f)
            return True
        except json.JSONDecodeError as e:
            self.errors.append(f"Invalid JSON: {e}")
            return False

    def _validate_required_fields(self):
        """Validate required configuration fields."""
        required = ["dependencies", "graphs"]

        for field in required:
            if field not in self.config:
                self.errors.append(f"Missing required field: '{field}'")

    def _validate_dependencies(self):
        """Validate dependencies field."""
        if "dependencies" not in self.config:
            return

        deps = self.config["dependencies"]

        if not isinstance(deps, list):
            self.errors.append("'dependencies' must be an array")
            return

        if len(deps) == 0:
            self.warnings.append("'dependencies' array is empty")

        for i, dep in enumerate(deps):
            if not isinstance(dep, str):
                self.errors.append(f"dependencies[{i}] must be a string")
                continue

            # Check if it's a path reference
            if dep.startswith("./") or dep.startswith("../") or dep == ".":
                dep_path = (self.config_path.parent / dep).resolve()
                if not dep_path.exists():
                    self.warnings.append(
                        f"Dependency path not found: {dep} (resolved to {dep_path})"
                    )
                else:
                    # Check for package files
                    has_package = any([
                        (dep_path / "pyproject.toml").exists(),
                        (dep_path / "setup.py").exists(),
                        (dep_path / "requirements.txt").exists(),
                        (dep_path / "package.json").exists(),
                    ])
                    if not has_package:
                        self.warnings.append(
                            f"Dependency path '{dep}' has no package configuration "
                            "(pyproject.toml, setup.py, requirements.txt, or package.json)"
                        )

    def _validate_graphs(self):
        """Validate graphs field."""
        if "graphs" not in self.config:
            return

        graphs = self.config["graphs"]

        if not isinstance(graphs, dict):
            self.errors.append("'graphs' must be an object/dict")
            return

        if len(graphs) == 0:
            self.errors.append("'graphs' object is empty - at least one graph required")

        for graph_name, graph_path in graphs.items():
            if not isinstance(graph_path, str):
                self.errors.append(f"Graph '{graph_name}' path must be a string")
                continue

            # Validate format: ./path/to/file.py:variable or ./path/to/file.py:make_graph
            if ":" not in graph_path:
                self.errors.append(
                    f"Graph '{graph_name}' path must be in format "
                    "'./path/to/file.py:variable' or './path/to/file.js:variable'"
                )
                continue

            file_path, var_name = graph_path.rsplit(":", 1)

            # Check if file exists
            resolved_path = (self.config_path.parent / file_path).resolve()
            if not resolved_path.exists():
                self.warnings.append(
                    f"Graph '{graph_name}' file not found: {file_path} "
                    f"(resolved to {resolved_path})"
                )

    def _validate_optional_fields(self):
        """Validate optional configuration fields."""
        config = self.config

        # Validate env
        if "env" in config:
            if isinstance(config["env"], str):
                env_path = self.config_path.parent / config["env"]
                if not env_path.exists():
                    self.warnings.append(f".env file not found: {config['env']}")
            elif not isinstance(config["env"], dict):
                self.errors.append("'env' must be a string (path) or object (key-value pairs)")

        # Validate python_version
        if "python_version" in config:
            valid_versions = ["3.11", "3.12", "3.13"]
            if config["python_version"] not in valid_versions:
                self.errors.append(
                    f"'python_version' must be one of {valid_versions}, "
                    f"got '{config['python_version']}'"
                )

        # Validate node_version
        if "node_version" in config:
            if config["node_version"] != "20":
                self.warnings.append(
                    f"'node_version' is {config['node_version']}, "
                    "only version 20 is officially supported"
                )

        # Validate base_image
        if "base_image" in config and not isinstance(config["base_image"], str):
            self.errors.append("'base_image' must be a string")

        # Validate dockerfile_lines
        if "dockerfile_lines" in config:
            if not isinstance(config["dockerfile_lines"], list):
                self.errors.append("'dockerfile_lines' must be an array")
            elif not all(isinstance(line, str) for line in config["dockerfile_lines"]):
                self.errors.append("All 'dockerfile_lines' entries must be strings")

        # Validate checkpointer
        if "checkpointer" in config and not isinstance(config["checkpointer"], dict):
            self.errors.append("'checkpointer' must be an object/dict")

        # Validate http
        if "http" in config:
            if not isinstance(config["http"], dict):
                self.errors.append("'http' must be an object/dict")

        # Validate store
        if "store" in config and not isinstance(config["store"], dict):
            self.errors.append("'store' must be an object/dict")

    def print_results(self):
        """Print validation results."""
        if self.errors:
            print("❌ Validation failed with errors:\n")
            for error in self.errors:
                print(f"  ERROR: {error}")
            print()

        if self.warnings:
            print("⚠️  Warnings:\n")
            for warning in self.warnings:
                print(f"  WARNING: {warning}")
            print()

        if not self.errors and not self.warnings:
            print("✅ Configuration is valid!")
        elif not self.errors:
            print("✅ Configuration is valid (with warnings)")

        return not self.errors


def main():
    parser = argparse.ArgumentParser(
        description="Validate langgraph.json configuration file"
    )
    parser.add_argument(
        "config_path",
        nargs="?",
        default="langgraph.json",
        help="Path to langgraph.json (default: langgraph.json in current directory)"
    )

    args = parser.parse_args()
    config_path = Path(args.config_path).resolve()

    print(f"🔍 Validating: {config_path}\n")

    validator = ConfigValidator(config_path)
    is_valid = validator.validate()
    validator.print_results()

    sys.exit(0 if is_valid else 1)


if __name__ == "__main__":
    main()
