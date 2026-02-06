#!/usr/bin/env python3
"""
Generate a LangSmith monitoring setup plan based on current official docs.

Notes:
- LangSmith alerts are configured per project in the UI.
- Official docs document the alert flow and webhook configuration UI, including
  supported metric families and webhook payload fields.
- This script intentionally generates setup plans/templates instead of calling
  undocumented alert-management REST endpoints.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

DOCS_ALERTS = "https://docs.langchain.com/langsmith/alerts"
DOCS_ALERTS_WEBHOOK = "https://docs.langchain.com/langsmith/alerts-webhook"
DOCS_DASHBOARDS = "https://docs.langchain.com/langsmith/dashboards"
DEFAULT_WORKSPACE_URL = "https://smith.langchain.com"


def parse_headers(values: Optional[List[str]]) -> Dict[str, str]:
    """Parse --header KEY=VALUE flags into a dictionary."""
    headers: Dict[str, str] = {}
    if not values:
        return headers

    for raw in values:
        if "=" not in raw:
            raise ValueError(f"Invalid --header '{raw}'. Expected KEY=VALUE")
        key, value = raw.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"Invalid --header '{raw}'. Header name cannot be empty")
        headers[key] = value.strip()

    return headers


def default_alert_recommendations(project_name: str) -> List[Dict[str, Any]]:
    """Return conservative default alert recommendations using documented metric types."""
    return [
        {
            "name": f"{project_name} - High Error Rate",
            "metric_type": "Errored Runs",
            "condition": {
                "aggregation_method": "Percentage",
                "operator": ">=",
                "threshold": 5,
                "window_minutes": 5,
            },
            "rationale": "Detect spikes in failed runs quickly.",
        },
        {
            "name": f"{project_name} - High Latency",
            "metric_type": "Latency",
            "condition": {
                "aggregation_method": "Average",
                "operator": ">=",
                "threshold": 60,
                "window_minutes": 5,
                "threshold_unit": "seconds",
            },
            "rationale": "Detect performance regressions.",
        },
        {
            "name": f"{project_name} - Low Feedback Score",
            "metric_type": "Feedback Score",
            "condition": {
                "aggregation_method": "Average",
                "operator": "<=",
                "threshold": 0.7,
                "window_minutes": 15,
            },
            "rationale": "Detect quality regressions surfaced by user/evaluator feedback.",
        },
    ]


def webhook_template(
    url: str,
    headers: Dict[str, str],
    include_slack_template: bool,
    slack_channel_id: Optional[str],
) -> Dict[str, Any]:
    """Build webhook notification template with documented LangSmith payload fields."""
    template: Dict[str, Any] = {
        "webhook_url": url,
        "headers": headers,
        "expected_langsmith_fields": [
            "project_name",
            "alert_rule_id",
            "alert_rule_name",
            "alert_rule_type",
            "alert_rule_attribute",
            "triggered_metric_value",
            "triggered_threshold",
            "timestamp",
        ],
    }

    if include_slack_template:
        if not slack_channel_id:
            raise ValueError("--slack-channel-id is required when --include-slack-template is set")

        template["slack_example_body"] = {
            "channel": slack_channel_id,
            "text": "{alert_rule_name} triggered for {project_name}",
            "blocks": [
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": "Alert triggered in LangSmith",
                    },
                },
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": "Rule: {alert_rule_name} | Project: {project_name}",
                    },
                },
            ],
        }

    return template


def build_plan(
    project_name: str,
    workspace_url: str,
    alerts: List[Dict[str, Any]],
    webhook: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Create full monitoring setup plan."""
    cleaned_workspace_url = workspace_url.rstrip("/")
    ui_hints = {
        "workspace": cleaned_workspace_url,
        "monitoring_home": f"{cleaned_workspace_url}/",
    }

    steps = [
        "Open LangSmith and select the target tracing project.",
        "Open Monitoring to review the prebuilt dashboard and add custom charts if needed.",
        "Open Alerts and create one alert per policy in alert_recommendations.",
        "For each alert, choose metric type and configure condition fields: aggregation method, operator, threshold, and 5- or 15-minute window.",
    ]

    if webhook:
        steps.append(
            "In alert notifications, select Webhook and paste webhook configuration from this plan."
        )

    steps.append("Validate by forcing a known condition in a non-production environment.")

    return {
        "project": project_name,
        "ui_hints": ui_hints,
        "references": {
            "alerts": DOCS_ALERTS,
            "alerts_webhook": DOCS_ALERTS_WEBHOOK,
            "dashboards": DOCS_DASHBOARDS,
        },
        "alert_recommendations": alerts,
        "webhook": webhook,
        "setup_steps": steps,
    }


def print_plan(plan: Dict[str, Any]):
    """Render a compact human-readable plan."""
    print("\n" + "=" * 72)
    print("LANGSMITH MONITORING SETUP PLAN")
    print("=" * 72)
    print(f"Project: {plan['project']}")
    print(f"Workspace: {plan['ui_hints']['workspace']}")

    print("\nReferences:")
    refs = plan["references"]
    print(f"  Alerts: {refs['alerts']}")
    print(f"  Webhook notifications: {refs['alerts_webhook']}")
    print(f"  Dashboards: {refs['dashboards']}")

    print("\nRecommended alerts:")
    for alert in plan["alert_recommendations"]:
        condition = alert["condition"]
        unit_suffix = f" {condition['threshold_unit']}" if "threshold_unit" in condition else ""
        print(f"  - {alert['name']}")
        print(f"    Metric: {alert['metric_type']}")
        print(
            "    Condition: "
            f"{condition['aggregation_method']} {condition['operator']} "
            f"{condition['threshold']}{unit_suffix}, window={condition['window_minutes']} min"
        )
        print(f"    Why: {alert['rationale']}")

    if plan.get("webhook"):
        print("\nWebhook template is included in the exported JSON plan.")

    print("\nSetup steps:")
    for i, step in enumerate(plan["setup_steps"], 1):
        print(f"  {i}. {step}")

    print("=" * 72 + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate a docs-aligned LangSmith monitoring/alerts setup plan"
    )
    parser.add_argument("--project", required=True, help="LangSmith project name")
    parser.add_argument(
        "--workspace-url",
        default=DEFAULT_WORKSPACE_URL,
        help="LangSmith workspace URL (default: cloud UI)",
    )
    parser.add_argument(
        "--webhook-url",
        help="Optional webhook endpoint to include in generated notification template",
    )
    parser.add_argument(
        "--header",
        action="append",
        help="Webhook header KEY=VALUE (repeatable)",
    )
    parser.add_argument(
        "--include-slack-template",
        action="store_true",
        help="Include Slack JSON body example in the webhook section",
    )
    parser.add_argument(
        "--slack-channel-id",
        help="Slack channel ID for --include-slack-template",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        help="Write the full plan to a JSON file",
    )

    args = parser.parse_args()

    try:
        headers = parse_headers(args.header)

        webhook = None
        if args.webhook_url:
            webhook = webhook_template(
                url=args.webhook_url,
                headers=headers,
                include_slack_template=args.include_slack_template,
                slack_channel_id=args.slack_channel_id,
            )

        plan = build_plan(
            project_name=args.project,
            workspace_url=args.workspace_url,
            alerts=default_alert_recommendations(args.project),
            webhook=webhook,
        )

        print_plan(plan)

        if args.output_json:
            args.output_json.parent.mkdir(parents=True, exist_ok=True)
            args.output_json.write_text(
                json.dumps(plan, indent=2) + "\n",
                encoding="utf-8",
            )
            print(f"Saved setup plan to: {args.output_json}")

        return 0

    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
