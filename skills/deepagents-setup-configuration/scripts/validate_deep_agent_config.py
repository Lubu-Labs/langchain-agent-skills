#!/usr/bin/env python3
"""
Validate Deep Agent configuration by parsing agent files.

Usage:
    uv run validate_deep_agent_config.py <agent_file_path>

Fallback:
    python3 validate_deep_agent_config.py <agent_file_path>

Validates:
- Model parameter exists and is valid
- Tools are properly formatted
- Middleware configuration is correct
- Backend configuration is valid
- System prompt is provided
"""

import argparse
import ast
import re
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional


class DeepAgentValidator:
    """Validator for Deep Agent configuration."""

    # Known valid model prefixes
    VALID_MODEL_PREFIXES = [
        "openai:",
        "anthropic:",
        "google_genai:",
        "google:",
        "bedrock:",
        "azure_openai:",
        "mistralai:",
        "ollama:",
        "groq:",
        "cohere:",
        "xai:",
        "gpt-",
        "claude-",
        "gemini-",
        "command-",
        "mistral-",
    ]

    def __init__(self, agent_path: Path):
        self.agent_path = agent_path
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.content: Optional[str] = None
        self.found_agents: List[Dict[str, Any]] = []

    def validate(self) -> bool:
        """Run all validation checks. Returns True if valid."""
        if not self._load_file():
            return False

        if self.agent_path.suffix == ".py":
            self._validate_python()
        elif self.agent_path.suffix in {".js", ".ts"}:
            self._validate_javascript()
        else:
            self.errors.append(
                f"Unsupported file type: {self.agent_path.suffix}. "
                "Expected .py, .js, or .ts"
            )
            return False

        return len(self.errors) == 0

    def _load_file(self) -> bool:
        """Load and read the agent file."""
        if not self.agent_path.exists():
            self.errors.append(f"File not found: {self.agent_path}")
            return False

        try:
            self.content = self.agent_path.read_text()
            return True
        except Exception as e:
            self.errors.append(f"Failed to read file: {e}")
            return False

    def _validate_python(self):
        """Validate Python Deep Agent file."""
        # Check for import
        if "from deepagents import create_deep_agent" not in self.content:
            self.warnings.append(
                "Missing 'from deepagents import create_deep_agent' import"
            )

        # Parse AST to find create_deep_agent calls
        try:
            tree = ast.parse(self.content)
            self._find_deep_agent_calls_python(tree)
        except SyntaxError as e:
            self.errors.append(f"Python syntax error: {e}")
            return

        if not self.found_agents:
            self.errors.append("No create_deep_agent() calls found in file")
            return

        # Validate each agent configuration
        for i, agent_config in enumerate(self.found_agents):
            agent_num = f"Agent #{i+1}" if len(self.found_agents) > 1 else "Agent"
            self._validate_agent_config(agent_config, agent_num)

    def _validate_javascript(self):
        """Validate JavaScript Deep Agent file."""
        # Check for import
        if "createDeepAgent" not in self.content:
            self.warnings.append(
                "Missing 'createDeepAgent' import from deepagents"
            )

        # Use regex to find createDeepAgent calls (basic parsing)
        pattern = r"createDeepAgent\s*\(\s*\{([^}]*(?:\{[^}]*\}[^}]*)*)\}\s*\)"
        matches = re.finditer(pattern, self.content, re.DOTALL)

        for match in matches:
            config_text = match.group(1)
            agent_config = self._parse_javascript_config(config_text)
            self.found_agents.append(agent_config)

        # Fallback: detect calls with dynamic config (e.g. createDeepAgent(config))
        if not self.found_agents:
            dynamic_calls = list(re.finditer(r"createDeepAgent\s*\(", self.content))
            for _ in dynamic_calls:
                self.found_agents.append({"<dynamic_config>": True})

        if not self.found_agents:
            self.errors.append("No createDeepAgent() calls found in file")
            return

        # Validate each agent configuration
        for i, agent_config in enumerate(self.found_agents):
            agent_num = f"Agent #{i+1}" if len(self.found_agents) > 1 else "Agent"
            self._validate_agent_config(agent_config, agent_num)

    def _find_deep_agent_calls_python(self, tree: ast.AST):
        """Find all create_deep_agent calls in Python AST."""
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                # Check if it's a call to create_deep_agent
                if (isinstance(node.func, ast.Name) and
                    node.func.id == "create_deep_agent"):
                    config = self._extract_python_config(node)
                    self.found_agents.append(config)

    def _extract_python_config(self, call_node: ast.Call) -> Dict[str, Any]:
        """Extract configuration from Python create_deep_agent call."""
        config = {}

        # Extract keyword arguments
        for keyword in call_node.keywords:
            if keyword.arg:
                config[keyword.arg] = self._ast_to_value(keyword.value)
            else:
                # Handles create_deep_agent(**kwargs)
                config["<kwargs_unpack>"] = True

        return config

    def _ast_to_value(self, node: ast.AST) -> Any:
        """Convert AST node to Python value (simplified)."""
        if isinstance(node, ast.Constant):
            return node.value
        elif isinstance(node, ast.Str):  # Python 3.7 compatibility
            return node.s
        elif isinstance(node, ast.Num):  # Python 3.7 compatibility
            return node.n
        elif isinstance(node, ast.List):
            return [self._ast_to_value(elem) for elem in node.elts]
        elif isinstance(node, ast.Dict):
            return {
                self._ast_to_value(k): self._ast_to_value(v)
                for k, v in zip(node.keys, node.values)
            }
        elif isinstance(node, ast.Name):
            return f"<variable: {node.id}>"
        elif isinstance(node, ast.Call):
            return "<function_call>"
        elif isinstance(node, ast.Lambda):
            return "<lambda>"
        else:
            return "<complex_expression>"

    def _parse_javascript_config(self, config_text: str) -> Dict[str, Any]:
        """Parse JavaScript object configuration (simplified)."""
        config = {}

        # Extract simple key-value pairs
        # This is a simplified parser - won't handle all cases
        lines = config_text.split("\n")

        for line in lines:
            line = line.strip()
            if not line or line.startswith("//"):
                continue

            # Match: key: value,
            match = re.match(r"(\w+)\s*:\s*(.+?),?\s*$", line)
            if match:
                key = match.group(1)
                value = match.group(2).rstrip(",")

                # Try to parse value
                if value.startswith('"') or value.startswith("'"):
                    config[key] = value.strip('"\'')
                elif value == "true":
                    config[key] = True
                elif value == "false":
                    config[key] = False
                elif value.startswith("["):
                    config[key] = "<array>"
                elif value.startswith("{"):
                    config[key] = "<object>"
                else:
                    config[key] = value

        return config

    def _validate_agent_config(self, config: Dict[str, Any], agent_num: str):
        """Validate a single agent configuration."""
        has_dynamic_config = bool(
            config.get("<kwargs_unpack>") or config.get("<dynamic_config>")
        )
        if has_dynamic_config:
            self.warnings.append(
                f"{agent_num}: Dynamic config detected - some static checks were skipped"
            )

        # Check for model
        if "model" not in config:
            if has_dynamic_config:
                self.warnings.append(
                    f"{agent_num}: Could not statically verify required 'model' parameter"
                )
            else:
                self.errors.append(f"{agent_num}: Missing required 'model' parameter")
        else:
            model_value = config["model"]
            if isinstance(model_value, str):
                self._validate_model_name(model_value, agent_num)
            else:
                self.warnings.append(
                    f"{agent_num}: Model is a variable/expression: {model_value}"
                )

        # Check for tools
        if "tools" in config:
            tools_value = config["tools"]
            if tools_value == "[]" or tools_value == "<array>":
                self.warnings.append(
                    f"{agent_num}: Tools array appears empty - agent may have limited functionality"
                )
            elif isinstance(tools_value, str) and tools_value.startswith("<"):
                # It's a variable reference, which is fine
                pass
        else:
            if not has_dynamic_config:
                self.warnings.append(
                    f"{agent_num}: No 'tools' parameter - agent will only have middleware tools"
                )

        # Check for system_prompt or systemPrompt
        has_prompt = (
            "system_prompt" in config
            or "systemPrompt" in config
            or "system" in config
        )
        if not has_prompt and not has_dynamic_config:
            self.warnings.append(
                f"{agent_num}: No system prompt provided - agent may lack clear instructions"
            )

        # Check middleware configuration
        if "middleware" in config:
            middleware_value = config["middleware"]
            if isinstance(middleware_value, str) and not middleware_value.startswith("<"):
                self.warnings.append(
                    f"{agent_num}: Custom middleware detected - ensure it's properly configured"
                )

        # Check backend configuration
        if "backend" in config:
            backend_value = config["backend"]
            if backend_value not in ["<lambda>", "<function_call>", "<complex_expression>"]:
                self.warnings.append(
                    f"{agent_num}: Backend should typically be a lambda/function: {backend_value}"
                )

        # Check subagents
        if "subagents" in config:
            subagents_value = config["subagents"]
            if subagents_value in ["[]", "<array>"]:
                self.warnings.append(
                    f"{agent_num}: Subagents parameter present but appears empty"
                )

        # Check store configuration
        if "store" in config and "backend" not in config:
            self.warnings.append(
                f"{agent_num}: 'store' provided without custom 'backend' - "
                "consider using StoreBackend for persistence"
            )

        # Check human-in-the-loop interrupt config
        interrupt_key = None
        if "interrupt_on" in config:
            interrupt_key = "interrupt_on"
        elif "interruptOn" in config:
            interrupt_key = "interruptOn"

        if interrupt_key:
            interrupt_value = config[interrupt_key]
            if isinstance(interrupt_value, list):
                self.warnings.append(
                    f"{agent_num}: '{interrupt_key}' should be a tool-name mapping (dict/object), not a list"
                )
            elif isinstance(interrupt_value, str) and not (
                interrupt_value == "<object>" or interrupt_value.startswith("<variable:")
            ):
                self.warnings.append(
                    f"{agent_num}: '{interrupt_key}' should be a mapping of tool names to interrupt settings"
                )

            if "checkpointer" not in config:
                self.warnings.append(
                    f"{agent_num}: '{interrupt_key}' is configured without 'checkpointer' - interruptions require persistence"
                )

    def _validate_model_name(self, model: str, agent_num: str):
        """Validate model name."""
        # Check if it's a known model prefix or provider:model identifier
        has_provider_prefix = ":" in model and not model.startswith(":") and not model.endswith(":")
        is_valid = has_provider_prefix or any(
            model.startswith(prefix) for prefix in self.VALID_MODEL_PREFIXES
        )

        if not is_valid:
            self.warnings.append(
                f"{agent_num}: Unknown model name '{model}' - "
                f"expected model starting with: {', '.join(self.VALID_MODEL_PREFIXES)}"
            )

        # Check for deprecated or old models
        deprecated_patterns = [
            ("gpt-3.5-turbo-0301", "Use a current tool-calling model such as gpt-4.1+"),
            ("text-davinci-003", "Use a current chat model such as gpt-4.1+"),
            ("claude-2", "Use a current Claude 3.5/4+ model"),
        ]

        for pattern, suggestion in deprecated_patterns:
            if pattern in model:
                self.warnings.append(
                    f"{agent_num}: Model '{model}' may be deprecated. {suggestion}"
                )

    def print_results(self):
        """Print validation results."""
        print(f"📋 Validation Report for: {self.agent_path.name}\n")

        if self.found_agents:
            print(f"✅ Found {len(self.found_agents)} Deep Agent configuration(s)\n")

        if self.errors:
            print("❌ Errors:\n")
            for error in self.errors:
                print(f"  ERROR: {error}")
            print()

        if self.warnings:
            print("⚠️  Warnings:\n")
            for warning in self.warnings:
                print(f"  WARNING: {warning}")
            print()

        if not self.errors and not self.warnings:
            print("✅ Configuration is valid! No issues found.")
        elif not self.errors:
            print("✅ Configuration is valid (with warnings)")
        else:
            print("❌ Configuration has errors and needs fixes")

        return not self.errors


def main():
    parser = argparse.ArgumentParser(
        description="Validate Deep Agent configuration file"
    )
    parser.add_argument(
        "agent_file_path",
        help="Path to agent.py, agent.js, or agent.ts file"
    )

    args = parser.parse_args()
    agent_path = Path(args.agent_file_path).resolve()

    print(f"🔍 Validating Deep Agent configuration...\n")

    validator = DeepAgentValidator(agent_path)
    is_valid = validator.validate()
    validator.print_results()

    sys.exit(0 if is_valid else 1)


if __name__ == "__main__":
    main()
