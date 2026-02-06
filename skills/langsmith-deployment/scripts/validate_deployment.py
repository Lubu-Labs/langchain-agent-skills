#!/usr/bin/env python3
"""
Validate langgraph.json deployment configuration.

Validation rules are aligned with current LangGraph CLI / LangSmith Deployment docs,
including support for:
- graphs (required)
- dependencies (required for Python projects, optional for JS projects)
- env as either a string path to .env or an inline mapping
- python_version / node_version / pip_installer compatibility checks
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

ALLOWED_PYTHON_VERSIONS = {"3.11", "3.12", "3.13"}
ALLOWED_PIP_INSTALLERS = {"auto", "pip", "uv"}
ALLOWED_NODE_VERSIONS = {"20"}


class DeploymentValidator:
    """Validate deployment configuration and readiness."""

    def __init__(self, config_path: Path, target: str = "cloud"):
        self.config_path = config_path
        self.target = target  # cloud, hybrid, or standalone
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.config: Optional[Dict[str, Any]] = None

    def validate(self) -> bool:
        """Run all validation checks."""
        print(f"Validating deployment configuration for target: {self.target}")
        print(f"Config file: {self.config_path}\n")

        self.validate_config_exists()
        if self.errors:
            self.report_results()
            return False

        self.validate_config_syntax()
        if self.errors:
            self.report_results()
            return False

        self.validate_required_fields()
        self.validate_graphs()
        self.validate_language_runtime_fields()
        self.validate_dependencies()
        self.validate_environment_variables()
        self.validate_optional_fields()

        if self.target in ["hybrid", "standalone"]:
            self.validate_docker_requirements()

        self.report_results()
        return len(self.errors) == 0

    def validate_config_exists(self):
        if not self.config_path.exists():
            self.errors.append(f"Configuration file not found: {self.config_path}")

    def validate_config_syntax(self):
        try:
            with self.config_path.open("r", encoding="utf-8") as f:
                parsed = json.load(f)
            if not isinstance(parsed, dict):
                self.errors.append("langgraph.json root must be a JSON object")
                return
            self.config = parsed
            print("✓ Valid JSON syntax")
        except json.JSONDecodeError as e:
            self.errors.append(f"Invalid JSON syntax: {e}")

    def is_js_project(self) -> bool:
        assert self.config is not None
        return "node_version" in self.config

    def validate_required_fields(self):
        assert self.config is not None

        if "graphs" not in self.config:
            self.errors.append("Missing required field 'graphs'")
        else:
            print("✓ Field 'graphs' present")

        # Current docs emphasize dependencies as required for Python projects.
        # JS projects can rely on package.json and node_version-based setup.
        if not self.is_js_project():
            if "dependencies" not in self.config:
                self.errors.append(
                    "Missing required field 'dependencies' for Python projects"
                )
            else:
                print("✓ Field 'dependencies' present")
        elif "dependencies" in self.config:
            print("✓ Field 'dependencies' present")

    def validate_graphs(self):
        assert self.config is not None

        graphs = self.config.get("graphs")
        if graphs is None:
            return

        if not isinstance(graphs, dict):
            self.errors.append("'graphs' must be an object mapping graph_id -> module_path:attribute")
            return

        if not graphs:
            self.errors.append("At least one graph must be defined in 'graphs'")
            return

        print(f"\nValidating {len(graphs)} graph(s):")
        for graph_name, graph_path in graphs.items():
            if not isinstance(graph_name, str) or not graph_name:
                self.errors.append(f"Invalid graph key: {graph_name!r}")
                continue

            if not isinstance(graph_path, str):
                self.errors.append(f"Graph '{graph_name}': path must be a string")
                continue

            if ":" not in graph_path:
                self.errors.append(
                    f"Graph '{graph_name}': expected 'module_or_file:variable_or_factory', got '{graph_path}'"
                )
                continue

            module_path, symbol = graph_path.split(":", 1)
            if not module_path.strip() or not symbol.strip():
                self.errors.append(
                    f"Graph '{graph_name}': invalid graph path '{graph_path}'"
                )
                continue

            print(f"  ✓ {graph_name}: {graph_path}")

    def validate_language_runtime_fields(self):
        assert self.config is not None

        python_version = self.config.get("python_version")
        if python_version is not None:
            version_text = str(python_version)
            if version_text not in ALLOWED_PYTHON_VERSIONS:
                self.errors.append(
                    "'python_version' must be one of "
                    f"{sorted(ALLOWED_PYTHON_VERSIONS)} (got {version_text!r})"
                )
            else:
                print(f"✓ python_version: {version_text}")

        pip_installer = self.config.get("pip_installer")
        if pip_installer is not None:
            value = str(pip_installer)
            if value not in ALLOWED_PIP_INSTALLERS:
                self.errors.append(
                    f"'pip_installer' must be one of {sorted(ALLOWED_PIP_INSTALLERS)} (got {value!r})"
                )
            else:
                print(f"✓ pip_installer: {value}")

        node_version = self.config.get("node_version")
        if node_version is not None:
            node_text = str(node_version)
            if node_text not in ALLOWED_NODE_VERSIONS:
                self.warnings.append(
                    f"Uncommon node_version '{node_text}'. Docs currently highlight {sorted(ALLOWED_NODE_VERSIONS)}."
                )
            else:
                print(f"✓ node_version: {node_text}")

        if self.is_js_project() and "dependencies" not in self.config:
            self.warnings.append(
                "No 'dependencies' key found. Ensure package.json is present and complete for JS builds."
            )

    def validate_dependencies(self):
        assert self.config is not None

        if "dependencies" not in self.config:
            return

        deps = self.config["dependencies"]
        if not isinstance(deps, list):
            self.errors.append("'dependencies' must be a list")
            return

        if not deps:
            self.warnings.append("'dependencies' is empty; verify this is intentional.")
            return

        print(f"\nValidating {len(deps)} dependencies:")

        dep_strings = [str(dep) for dep in deps]
        if not self.is_js_project() and not any("langgraph" in d.lower() for d in dep_strings):
            self.warnings.append(
                "No dependency containing 'langgraph' found. Verify runtime dependencies are complete."
            )

        for dep in deps:
            if not isinstance(dep, str):
                self.errors.append(f"Dependency must be a string: {dep!r}")
                continue
            if not dep.strip():
                self.errors.append("Dependency entries cannot be empty strings")
                continue
            print(f"  ✓ {dep}")

    def validate_environment_variables(self):
        assert self.config is not None

        if "env" not in self.config:
            print("\nNo 'env' key present. Environment variables can still be set at deployment time.")
            return

        env_config = self.config["env"]

        if isinstance(env_config, str):
            print(f"\n✓ env points to file: {env_config}")
            env_path = (self.config_path.parent / env_config).resolve()
            if not env_path.exists():
                self.warnings.append(
                    f"env file path '{env_config}' does not exist locally at {env_path}"
                )
            return

        if not isinstance(env_config, dict):
            self.errors.append("'env' must be either a string file path or an object mapping")
            return

        print(f"\nEnvironment variables defined inline: {len(env_config)}")

        common_keys = {
            "OPENAI_API_KEY": "OpenAI",
            "ANTHROPIC_API_KEY": "Anthropic",
            "LANGSMITH_API_KEY": "LangSmith tracing",
        }

        for key, provider in common_keys.items():
            if key in env_config:
                print(f"  ✓ {key} ({provider})")

        for key, value in env_config.items():
            if not isinstance(key, str):
                self.errors.append(f"Environment variable key must be a string: {key!r}")
                continue

            if not isinstance(value, str):
                self.warnings.append(
                    f"Environment variable '{key}' has non-string value; verify this is intentional"
                )
                continue

            looks_sensitive = any(token in key.upper() for token in ["KEY", "TOKEN", "SECRET", "PASSWORD"])
            if looks_sensitive and value and not value.startswith("$") and not value.startswith("<"):
                if len(value) > 8:
                    self.warnings.append(
                        f"Environment variable '{key}' appears to contain an inline secret. "
                        "Prefer injecting secrets at deploy time."
                    )

    def validate_optional_fields(self):
        assert self.config is not None

        auth = self.config.get("auth")
        if auth is not None:
            if not isinstance(auth, dict) and not isinstance(auth, str):
                self.warnings.append("'auth' is present but not in an expected format")

        api_version = self.config.get("api_version")
        if api_version is not None and not isinstance(api_version, str):
            self.warnings.append("'api_version' is present but not a string")

    def validate_docker_requirements(self):
        print(f"\nValidating Docker requirements for {self.target} deployment:")

        try:
            result = subprocess.run(
                ["docker", "--version"],
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode == 0:
                print(f"  ✓ Docker installed: {result.stdout.strip()}")
            else:
                self.warnings.append("Docker command failed. Verify Docker is installed and accessible.")
        except FileNotFoundError:
            self.warnings.append("Docker is not installed. Required for many hybrid/standalone flows.")

        dockerfile = self.config_path.parent / "Dockerfile"
        if dockerfile.exists():
            print("  ✓ Custom Dockerfile found")
        else:
            print("  Note: No custom Dockerfile found (CLI defaults may be used)")

    def report_results(self):
        print("\n" + "=" * 60)
        print("VALIDATION RESULTS")
        print("=" * 60)

        if self.warnings:
            print(f"\nWarnings ({len(self.warnings)}):")
            for warning in self.warnings:
                print(f"  - {warning}")

        if self.errors:
            print(f"\nErrors ({len(self.errors)}):")
            for error in self.errors:
                print(f"  - {error}")
            print("\nValidation FAILED")
        else:
            print("\nValidation PASSED")
            if self.warnings:
                print("(with warnings - review recommended)")

        print("=" * 60)


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate LangGraph deployment configuration")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("langgraph.json"),
        help="Path to langgraph.json (default: langgraph.json)",
    )
    parser.add_argument(
        "--target",
        choices=["cloud", "hybrid", "standalone"],
        default="cloud",
        help="Deployment target (default: cloud)",
    )

    args = parser.parse_args()

    validator = DeploymentValidator(config_path=args.config, target=args.target)

    try:
        success = validator.validate()
        return 0 if success else 1
    except Exception as e:
        print(f"\nValidation error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
