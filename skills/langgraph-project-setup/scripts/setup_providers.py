#!/usr/bin/env python3
"""
Interactive script to configure LLM provider credentials.

Usage:
    uv run setup_providers.py [--output PATH]

Fallback:
    python3 setup_providers.py [--output PATH]

Creates or updates .env file with provider API keys.
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional


class ProviderSetup:
    """Interactive provider credential setup."""

    PROVIDERS = {
        "openai": {
            "name": "OpenAI",
            "env_vars": ["OPENAI_API_KEY"],
            "docs": "https://platform.openai.com/api-keys"
        },
        "anthropic": {
            "name": "Anthropic (Claude)",
            "env_vars": ["ANTHROPIC_API_KEY"],
            "docs": "https://console.anthropic.com/settings/keys"
        },
        "google": {
            "name": "Google (Gemini)",
            "env_vars": ["GOOGLE_API_KEY"],
            "docs": "https://aistudio.google.com/app/apikey"
        },
        "aws": {
            "name": "AWS Bedrock",
            "env_vars": [
                "AWS_ACCESS_KEY_ID",
                "AWS_SECRET_ACCESS_KEY",
                "AWS_DEFAULT_REGION"
            ],
            "docs": "https://docs.aws.amazon.com/bedrock/"
        },
        "langsmith": {
            "name": "LangSmith (Tracing & Monitoring)",
            "env_vars": [
                "LANGSMITH_API_KEY",
                "LANGSMITH_TRACING",
                "LANGSMITH_PROJECT"
            ],
            "docs": "https://smith.langchain.com/settings"
        },
        "tavily": {
            "name": "Tavily (Search)",
            "env_vars": ["TAVILY_API_KEY"],
            "docs": "https://app.tavily.com/"
        }
    }

    def __init__(self, env_path: Path):
        self.env_path = env_path
        self.existing_env: Dict[str, str] = {}
        self.new_env: Dict[str, str] = {}

        if self.env_path.exists():
            self._load_existing_env()

    def _load_existing_env(self):
        """Load existing .env file."""
        with open(self.env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    self.existing_env[key.strip()] = value.strip()

    def run_interactive_setup(self):
        """Run interactive provider setup."""
        print("🔧 LangGraph Provider Setup")
        print("=" * 60)
        print()
        print("This wizard will help you configure API keys for LLM providers.")
        print()

        # Get provider selection
        selected_providers = self._select_providers()

        if not selected_providers:
            print("\n❌ No providers selected. Exiting.")
            return False

        print()
        print("=" * 60)
        print("Configuring selected providers...")
        print("=" * 60)
        print()

        # Configure each provider
        for provider_id in selected_providers:
            self._configure_provider(provider_id)

        # Save configuration
        self._save_env_file()

        print()
        print("✅ Configuration saved to", self.env_path)
        print()
        print("📝 Next steps:")
        print("   1. Review the .env file")
        print("   2. Never commit .env to version control")
        print("   3. Start your development server: langgraph dev")

        return True

    def _select_providers(self) -> List[str]:
        """Prompt user to select providers."""
        print("Select providers to configure (comma-separated numbers):")
        print()

        provider_ids = list(self.PROVIDERS.keys())
        for i, provider_id in enumerate(provider_ids, 1):
            provider = self.PROVIDERS[provider_id]
            configured = any(
                var in self.existing_env
                for var in provider["env_vars"]
            )
            status = "✓ configured" if configured else "○ not configured"
            print(f"  {i}. {provider['name']:30s} [{status}]")

        print()
        print("  0. All providers")
        print()

        while True:
            try:
                choice = input("Enter selection (e.g., 1,2,5 or 0 for all): ").strip()

                if choice == "0":
                    return provider_ids

                selections = [int(x.strip()) for x in choice.split(",")]

                if all(1 <= s <= len(provider_ids) for s in selections):
                    return [provider_ids[s - 1] for s in selections]

                print("❌ Invalid selection. Try again.")
            except (ValueError, IndexError):
                print("❌ Invalid input. Try again.")

    def _configure_provider(self, provider_id: str):
        """Configure a specific provider."""
        provider = self.PROVIDERS[provider_id]

        print(f"\n📦 Configuring {provider['name']}")
        print(f"   Docs: {provider['docs']}")
        print()

        for env_var in provider["env_vars"]:
            existing_value = self.existing_env.get(env_var)

            if existing_value:
                print(f"   {env_var}: (already set)")
                update = input("   Update? (y/N): ").strip().lower()
                if update != "y":
                    self.new_env[env_var] = existing_value
                    continue

            # Special handling for certain variables
            if env_var == "LANGSMITH_TRACING":
                value = input(f"   Enable tracing? (y/N): ").strip().lower()
                self.new_env[env_var] = "true" if value == "y" else "false"
            elif env_var == "AWS_DEFAULT_REGION":
                default = "us-east-1"
                value = input(f"   {env_var} (default: {default}): ").strip()
                self.new_env[env_var] = value if value else default
            elif env_var == "LANGSMITH_PROJECT":
                default = "default"
                value = input(f"   {env_var} (default: {default}): ").strip()
                self.new_env[env_var] = value if value else default
            else:
                value = input(f"   {env_var}: ").strip()
                if value:
                    self.new_env[env_var] = value

    def _save_env_file(self):
        """Save configuration to .env file."""
        # Merge existing and new env vars
        all_vars = {**self.existing_env, **self.new_env}

        lines = []

        # Add header
        lines.append("# LangGraph Environment Configuration")
        lines.append("# Generated by setup_providers.py")
        lines.append("# WARNING: Never commit this file to version control!")
        lines.append("")

        # Group by provider
        for provider_id, provider in self.PROVIDERS.items():
            provider_vars = {
                var: all_vars.get(var)
                for var in provider["env_vars"]
                if var in all_vars
            }

            if provider_vars:
                lines.append(f"# {provider['name']}")
                for var, value in provider_vars.items():
                    lines.append(f"{var}={value}")
                lines.append("")

        # Add any other variables not in known providers
        known_vars = set()
        for provider in self.PROVIDERS.values():
            known_vars.update(provider["env_vars"])

        other_vars = {k: v for k, v in all_vars.items() if k not in known_vars}
        if other_vars:
            lines.append("# Other Variables")
            for var, value in other_vars.items():
                lines.append(f"{var}={value}")
            lines.append("")

        # Write file
        self.env_path.write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(
        description="Interactive LLM provider credential setup"
    )
    parser.add_argument(
        "--output",
        default=".env",
        help="Output path for .env file (default: .env)"
    )

    args = parser.parse_args()
    env_path = Path(args.output).resolve()

    setup = ProviderSetup(env_path)

    if not setup.run_interactive_setup():
        sys.exit(1)


if __name__ == "__main__":
    main()
