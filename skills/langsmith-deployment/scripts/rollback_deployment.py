#!/usr/bin/env python3
"""
Rollback a LangSmith deployment to a previous revision using the control plane API.

Default control-plane endpoints (from official docs):
- US: https://api.host.langchain.com
- EU: https://eu.api.host.langchain.com

For self-hosted control planes, pass --control-plane-url with your /api-host URL.
"""

import argparse
import os
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

try:
    import requests
except ImportError:
    print("Error: requests library not found. Install with: uv pip install requests")
    sys.exit(1)

DEFAULT_CONTROL_PLANE_US = "https://api.host.langchain.com"
DEFAULT_CONTROL_PLANE_EU = "https://eu.api.host.langchain.com"
DEFAULT_CLOUD_UI_URL = "https://smith.langchain.com"
REQUEST_TIMEOUT_SECONDS = 30


class DeploymentRollback:
    """Manage deployment revisions and rollback actions."""

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

    def list_revisions(self, deployment_id: str) -> List[Dict[str, Any]]:
        """List all revisions for a deployment."""
        url = f"{self.base_url}/api/v2/deployments/{deployment_id}/revisions"
        response = requests.get(url, headers=self.headers, timeout=REQUEST_TIMEOUT_SECONDS)

        if response.status_code != 200:
            raise RuntimeError(f"Could not fetch revisions: {self._extract_error(response)}")

        payload = self._safe_json(response)
        if not isinstance(payload, list):
            raise RuntimeError("Could not parse revisions response")

        return [item for item in payload if isinstance(item, dict)]

    def get_revision_details(self, deployment_id: str, revision_id: str) -> Dict[str, Any]:
        """Get details of a specific revision."""
        url = f"{self.base_url}/api/v2/deployments/{deployment_id}/revisions/{revision_id}"
        response = requests.get(url, headers=self.headers, timeout=REQUEST_TIMEOUT_SECONDS)

        if response.status_code != 200:
            raise RuntimeError(
                f"Could not fetch revision details: {self._extract_error(response)}"
            )

        payload = self._safe_json(response)
        if not isinstance(payload, dict):
            raise RuntimeError("Could not parse revision details response")

        return payload

    def set_active_revision(self, deployment_id: str, revision_id: str) -> Dict[str, Any]:
        """Create a new revision from an existing one to effectively rollback."""
        url = f"{self.base_url}/api/v2/deployments/{deployment_id}/revisions"

        target_revision = self.get_revision_details(deployment_id, revision_id)

        payload = {
            "git_branch": target_revision.get("git_branch"),
            "api_config_relative_path": target_revision.get("api_config_relative_path"),
            "env_vars": target_revision.get("env_vars", {}),
        }

        response = requests.post(
            url,
            headers=self.headers,
            json=payload,
            timeout=REQUEST_TIMEOUT_SECONDS,
        )

        if response.status_code != 201:
            raise RuntimeError(f"Rollback failed: {self._extract_error(response)}")

        body = self._safe_json(response)
        if not isinstance(body, dict):
            raise RuntimeError("Rollback response format invalid")
        return body

    @staticmethod
    def _parse_iso_datetime(value: Any) -> datetime:
        if not isinstance(value, str):
            return datetime.min.replace(tzinfo=timezone.utc)
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return datetime.min.replace(tzinfo=timezone.utc)

    def display_revisions(self, deployment_id: str) -> List[Dict[str, Any]]:
        """Display revision history in a readable table."""
        revisions = self.list_revisions(deployment_id)

        if not revisions:
            print("No revisions found for this deployment")
            return []

        sorted_revisions = sorted(
            revisions,
            key=lambda rev: self._parse_iso_datetime(rev.get("created_at")),
            reverse=True,
        )

        print(f"\n{'=' * 80}")
        print(f"REVISION HISTORY (Deployment: {deployment_id})")
        print(f"{'=' * 80}\n")

        for i, rev in enumerate(sorted_revisions, 1):
            status = rev.get("status", "unknown")
            created_at = rev.get("created_at", "unknown")
            revision_id = str(rev.get("id", "unknown"))
            is_active = bool(rev.get("is_active", False))

            status_symbol = (
                "✓" if status == "deployed" else "⧗" if status == "deploying" else "✗"
            )
            active_marker = " [ACTIVE]" if is_active else ""

            print(f"{i}. {status_symbol} Revision {revision_id[:8]}...{active_marker}")
            print(f"   Status: {status}")
            print(f"   Created: {created_at}")
            print(f"   Branch: {rev.get('git_branch', 'N/A')}")
            print(f"   Config: {rev.get('api_config_relative_path', 'N/A')}")
            print()

        return sorted_revisions

    def rollback(self, deployment_id: str, target_revision: Optional[str] = None) -> Dict[str, Any]:
        """Rollback to a previous deployed revision."""
        revisions = self.list_revisions(deployment_id)

        if not revisions:
            raise RuntimeError("No revisions found for this deployment")

        if not target_revision:
            current_active = next((r for r in revisions if r.get("is_active")), None)
            current_active_id = current_active.get("id") if isinstance(current_active, dict) else None

            successful_revisions = [
                r
                for r in revisions
                if r.get("status") == "deployed" and r.get("id") != current_active_id
            ]

            if not successful_revisions:
                raise RuntimeError("No previous successful revision found for rollback")

            successful_revisions = sorted(
                successful_revisions,
                key=lambda rev: self._parse_iso_datetime(rev.get("created_at")),
                reverse=True,
            )

            target_revision = str(successful_revisions[0]["id"])
            print(
                "No target revision specified. "
                f"Using latest previously deployed revision: {target_revision[:8]}..."
            )

        print(f"\nRolling back deployment {deployment_id}")
        print(f"  Target revision: {target_revision}")

        new_revision = self.set_active_revision(deployment_id, target_revision)

        print("\n✓ Rollback revision created")
        print(f"  New revision ID: {new_revision.get('id', 'unknown')}")

        return new_revision


def resolve_control_plane_url(region: str, explicit_url: Optional[str]) -> str:
    if explicit_url:
        return explicit_url
    env_url = os.getenv("LANGSMITH_CONTROL_PLANE_URL")
    if env_url:
        return env_url
    return DEFAULT_CONTROL_PLANE_EU if region == "eu" else DEFAULT_CONTROL_PLANE_US


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Rollback LangSmith Deployment to a previous revision"
    )
    parser.add_argument("--deployment-id", required=True, help="Deployment ID to rollback")
    parser.add_argument("--revision-id", help="Specific revision ID to rollback to")
    parser.add_argument(
        "--list-revisions",
        action="store_true",
        help="List revisions without performing rollback",
    )
    parser.add_argument("--api-key", help="LangSmith API key (or LANGSMITH_API_KEY)")
    parser.add_argument(
        "--tenant-id",
        help="Workspace ID for org-scoped API keys (or LANGSMITH_TENANT_ID)",
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
        "--ui-url",
        default=DEFAULT_CLOUD_UI_URL,
        help="Base UI URL for printed deployment links",
    )

    args = parser.parse_args()

    control_plane_url = resolve_control_plane_url(args.region, args.control_plane_url)

    try:
        rollback_manager = DeploymentRollback(
            api_key=args.api_key,
            control_plane_url=control_plane_url,
            tenant_id=args.tenant_id,
        )

        rollback_manager.display_revisions(args.deployment_id)

        if args.list_revisions:
            return 0

        print(f"\n{'=' * 80}")
        rollback_manager.rollback(args.deployment_id, args.revision_id)
        ui_url = args.ui_url.rstrip("/")
        print(f"  Monitor deployment progress at: {ui_url}/deployments/{args.deployment_id}")
        print(f"{'=' * 80}\n")
        return 0

    except Exception as e:
        print(f"\n✗ Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
