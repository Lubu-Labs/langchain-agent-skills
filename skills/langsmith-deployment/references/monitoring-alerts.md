# Monitoring and Alerts

Current guidance for monitoring LangSmith projects and configuring alerts.

## Monitoring in LangSmith

LangSmith provides a **Monitoring** area with:

- **Prebuilt dashboards** (automatically generated per tracing project)
- **Custom dashboards** (user-defined charts/filters)

Reference: https://docs.langchain.com/langsmith/dashboards

## Alerts overview

Alerts are **project-scoped**, so configure them per project.

LangSmith alerting is threshold-based on these metric types:

1. **Errored Runs**
2. **Feedback Score**
3. **Latency**

Reference: https://docs.langchain.com/langsmith/alerts

### Self-hosted requirement

For self-hosted LangSmith, alerts require Helm chart `0.10.3` or later.

Reference: https://docs.langchain.com/langsmith/alerts

## Creating alert conditions

Alert conditions include:

- Aggregation method (`Average`, `Percentage`, or `Count`)
- Comparison operator (`>=`, `<=`, or exceeds threshold)
- Threshold value
- Aggregation window (currently `5` or `15` minutes)
- Feedback key (for Feedback Score alerts)

Reference: https://docs.langchain.com/langsmith/alerts

## Webhook notifications

LangSmith supports webhook notifications for alerts.

Required webhook field:
- `URL`

Optional webhook fields:
- `Headers` (JSON map)
- `Request Body Template`

When webhooks fire, LangSmith appends metadata fields such as:

- `project_name`
- `alert_rule_id`
- `alert_rule_name`
- `alert_rule_type` (currently threshold)
- `alert_rule_attribute` (`error_count`, `feedback_score`, `latency`)
- `triggered_metric_value`
- `triggered_threshold`
- `timestamp`

Reference: https://docs.langchain.com/langsmith/alerts-webhook

## Practical setup flow

1. Open target project in LangSmith.
2. Go to **Alerts** -> **Create Alert**.
3. Choose metric type and condition.
4. Add notification channel (email/webhook).
5. Save and validate by triggering a test condition.

References:
- https://docs.langchain.com/langsmith/alerts
- https://docs.langchain.com/langsmith/alerts-webhook

## Suggested production baseline

- One **Errored Runs** alert for sustained failure spikes.
- One **Latency** alert for user-facing performance regressions.
- One **Feedback Score** alert for quality drift.

Tune thresholds using recent historical behavior before enabling paging notifications.

## Dashboard and alert checklist

- Create a custom dashboard per production project.
- Add charts for error trend, latency trend, and feedback trend.
- Confirm alerts route to on-call channel.
- Revisit thresholds after major model/prompt changes.
- Keep alert ownership explicit (team or rotation).
