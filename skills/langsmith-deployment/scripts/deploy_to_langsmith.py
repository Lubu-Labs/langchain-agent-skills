#!/usr/bin/env python3
"""
Deploy LangGraph applications using the LangSmith Deployment control plane API.

Default control-plane endpoints (from official docs):
- US: https://api.host.langchain.com
- EU: https://eu.api.host.langchain.com

For self-hosted LangSmith Deployment control planes, pass --control-plane-url
with your instance URL ending in /api-host.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

try:
    import requests
except ImportError:
    print("Error: requests library not found. Install with: uv pip install requests")
    sys.exit(1)

DEFAULT_CONTROL_PLANE_US = "https://api.host.langchain.com"
DEFAULT_CONTROL_PLANE_EU = "https://eu.api.host.langchain.com"
DEFAULT_CLOUD_UI_URL = "https://smith.langchain.com"
REQUEST_TIMEOUT_SECONDS = 30


class LangSmithDeployer:
    """Deploy LangGraph applications to LangSmith Deployment."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        control_plane_url: Optional[str] = None,
        tenant_id: Optional[str] = None,
    ):
        self.api_key = api_key or os.getenv("LANGSMITH_API_KEY")
        if not self.api_key:
            raise ValueError(
                "LANGSMITH_API_KEY not found. Set it via environment variable or --api-key"
            )

        self.base_url = (control_plane_url or DEFAULT_CONTROL_PLANE_US).rstrip("/")
        self.tenant_id = tenant_id or os.getenv("LANGSMITH_TENANT_ID")

        self.headers = {
            "X-Api-Key": self.api_key,
            "Content-Type": "application/json",
        }
        if self.tenant_id:
            self.headers["X-Tenant-Id"] = self.tenant_id

    @staticmethod
    def _safe_json(response: requests.Response) -> Any:
        try:
            return response.json()
        except Exception:
            return None

    def _extract_error(self, response: requests.Response) -> str:
        payload = self._safe_json(response)
        if isinstance(payload, dict):
            detail = payload.get("detail") or payload.get("message")
            if detail:
                return str(detail)
        return response.text.strip() or f"HTTP {response.status_code}"

    def validate_langgraph_config(self, config_path: Path) -> Dict[str, Any]:
        """Validate local langgraph.json for required keys."""
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        with config_path.open("r", encoding="utf-8") as f:
            config = json.load(f)

        if "graphs" not in config:
            raise ValueError("Missing required field in langgraph.json: ['graphs']")
        if "node_version" not in config and "dependencies" not in config:
            raise ValueError(
                "Missing required field in langgraph.json: ['dependencies'] "
                "(required for Python projects)"
            )

        graphs = config.get("graphs", {})
        graph_count = len(graphs) if isinstance(graphs, dict) else 0
        print(f"✓ Configuration valid: {graph_count} graph(s) defined")
        return config

    def check_github_integration(self, owner: str, repo: str) -> bool:
        """Check if GitHub repository is integrated with LangSmith Deployment."""
        url = f"{self.base_url}/api/v1/integrations/github/repos"
        response = requests.get(url, headers=self.headers, timeout=REQUEST_TIMEOUT_SECONDS)

        if response.status_code != 200:
            print(f"Warning: Could not check GitHub integration: {response.status_code}")
            return False

        payload = self._safe_json(response)
        if not isinstance(payload, list):
            print("Warning: Unexpected integrations response format")
            return False

        repo_full_name = f"{owner}/{repo}"
        integrated = any(
            isinstance(repo_data, dict) and repo_data.get("full_name") == repo_full_name
            for repo_data in payload
        )

        if integrated:
            print(f"✓ GitHub repository {repo_full_name} is integrated")
        else:
            print(f"✗ GitHub repository {repo_full_name} is not integrated")
            print("  Add it in the LangSmith UI under GitHub integrations.")

        return integrated

    def create_deployment(
        self,
        name: str,
        owner: str,
        repo: str,
        branch: str,
        config_path: str,
        env_vars: Optional[Dict[str, str]] = None,
        shareable: bool = False,
    ) -> Dict[str, Any]:
        """Create a new deployment."""
        url = f"{self.base_url}/api/v2/deployments"

        payload = {
            "display_name": name,
            "config": {
                "git_repo": {
                    "url": f"https://github.com/{owner}/{repo}",
                    "branch": branch,
                    "api_config_relative_path": config_path,
                },
                "env_vars": env_vars or {},
                "shareable": shareable,
            },
        }

        print(f"\nCreating deployment '{name}'...")
        print(f"  Repository: {owner}/{repo}")
        print(f"  Branch: {branch}")
        print(f"  Config: {config_path}")

        response = requests.post(
            url,
            headers=self.headers,
            json=payload,
            timeout=REQUEST_TIMEOUT_SECONDS,
        )

        if response.status_code == 201:
            deployment = self._safe_json(response)
            if not isinstance(deployment, dict):
                raise RuntimeError("Deployment created but response format was invalid")
            print("\n✓ Deployment created successfully")
            print(f"  Deployment ID: {deployment.get('id', 'unknown')}")
            return deployment

        raise RuntimeError(f"Deployment failed: {self._extract_error(response)}")

    def create_revision(
        self,
        deployment_id: str,
        branch: Optional[str] = None,
        config_path: Optional[str] = None,
        env_vars: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """Create a new revision for an existing deployment."""
        url = f"{self.base_url}/api/v2/deployments/{deployment_id}/revisions"

        payload: Dict[str, Any] = {}
        if branch:
            payload["git_branch"] = branch
        if config_path:
            payload["api_config_relative_path"] = config_path
        if env_vars:
            payload["env_vars"] = env_vars

        print(f"\nCreating revision for deployment {deployment_id}...")
        response = requests.post(
            url,
            headers=self.headers,
            json=payload,
            timeout=REQUEST_TIMEOUT_SECONDS,
        )

        if response.status_code == 201:
            revision = self._safe_json(response)
            if not isinstance(revision, dict):
                raise RuntimeError("Revision created but response format was invalid")
            print("✓ Revision created successfully")
            print(f"  Revision ID: {revision.get('id', 'unknown')}")
            return revision

        raise RuntimeError(f"Revision creation failed: {self._extract_error(response)}")


def load_env_file(env_file: Path) -> Dict[str, str]:
    """Load KEY=VALUE pairs from a .env file."""
    env_vars: Dict[str, str] = {}
    if not env_file.exists():
        return env_vars

    with env_file.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            cleaned = value.strip()
            if (
                (cleaned.startswith('"') and cleaned.endswith('"'))
                or (cleaned.startswith("'") and cleaned.endswith("'"))
            ) and len(cleaned) >= 2:
                cleaned = cleaned[1:-1]
            env_vars[key.strip()] = cleaned

    return env_vars


def resolve_control_plane_url(region: str, explicit_url: Optional[str]) -> str:
    """Resolve control-plane base URL from args/env defaults."""
    if explicit_url:
        return explicit_url

    env_url = os.getenv("LANGSMITH_CONTROL_PLANE_URL")
    if env_url:
        return env_url

    if region == "eu":
        return DEFAULT_CONTROL_PLANE_EU
    return DEFAULT_CONTROL_PLANE_US


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Deploy LangGraph applications using LangSmith Deployment control-plane API"
    )
    parser.add_argument("--name", required=True, help="Deployment display name")
    parser.add_argument("--owner", required=True, help="GitHub owner (user/org)")
    parser.add_argument("--repo", required=True, help="GitHub repository name")
    parser.add_argument("--branch", default="main", help="Git branch (default: main)")
    parser.add_argument(
        "--config",
        default="langgraph.json",
        help="Path to langgraph.json in repo (default: langgraph.json)",
    )
    parser.add_argument("--env-file", type=Path, help="Path to .env file")
    parser.add_argument(
        "--env",
        action="append",
        help="Environment variable in KEY=VALUE format (repeatable)",
    )
    parser.add_argument(
        "--shareable",
        action="store_true",
        help="Make deployment accessible in Studio without API key",
    )
    parser.add_argument(
        "--api-key",
        help="LangSmith API key (or set LANGSMITH_API_KEY)",
    )
    parser.add_argument(
        "--tenant-id",
        help="Workspace ID for org-scoped API keys (or set LANGSMITH_TENANT_ID)",
    )
    parser.add_argument(
        "--region",
        choices=["us", "eu"],
        default="us",
        help="Cloud region for default control-plane endpoint (default: us)",
    )
    parser.add_argument(
        "--control-plane-url",
        help=(
            "Override control-plane base URL. "
            "Cloud defaults: https://api.host.langchain.com (US), "
            "https://eu.api.host.langchain.com (EU). "
            "Self-hosted is typically https://<host>/api-host."
        ),
    )
    parser.add_argument(
        "--deployment-id",
        help="Existing deployment ID (creates new revision)",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only validate local langgraph.json",
    )
    parser.add_argument(
        "--skip-github-check",
        action="store_true",
        help="Skip GitHub integration check",
    )
    parser.add_argument(
        "--ui-url",
        default=DEFAULT_CLOUD_UI_URL,
        help="UI base URL for printed deployment links (default: cloud UI)",
    )

    args = parser.parse_args()

    control_plane_url = resolve_control_plane_url(args.region, args.control_plane_url)

    try:
        deployer = LangSmithDeployer(
            api_key=args.api_key,
            control_plane_url=control_plane_url,
            tenant_id=args.tenant_id,
        )

        local_config = Path(args.config)
        if local_config.exists():
            deployer.validate_langgraph_config(local_config)
            if args.validate_only:
                print("\n✓ Validation complete")
                return 0
        else:
            print(f"Note: Local {args.config} not found. Using repository path at deploy time.")
            if args.validate_only:
                print("Warning: Cannot validate config that does not exist locally.")
                return 1

        if not args.skip_github_check:
            deployer.check_github_integration(args.owner, args.repo)

        env_vars: Dict[str, str] = {}
        if args.env_file:
            env_vars.update(load_env_file(args.env_file))
        if args.env:
            for env_pair in args.env:
                if "=" not in env_pair:
                    print(f"Warning: Ignoring invalid --env value '{env_pair}' (expected KEY=VALUE)")
                    continue
                key, value = env_pair.split("=", 1)
                env_vars[key.strip()] = value.strip()

        if args.deployment_id:
            deployer.create_revision(
                deployment_id=args.deployment_id,
                branch=args.branch,
                config_path=args.config,
                env_vars=env_vars if env_vars else None,
            )
        else:
            deployment = deployer.create_deployment(
                name=args.name,
                owner=args.owner,
                repo=args.repo,
                branch=args.branch,
                config_path=args.config,
                env_vars=env_vars,
                shareable=args.shareable,
            )

            deployment_id = deployment.get("id")
            if deployment_id:
                ui_url = args.ui_url.rstrip("/")
                print("\n" + "=" * 60)
                print("Deployment initiated. Monitor progress at:")
                print(f"{ui_url}/deployments/{deployment_id}")
                print("=" * 60)

        return 0

    except Exception as e:
        print(f"\n✗ Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
