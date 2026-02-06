# Troubleshooting Deployment Issues

Current troubleshooting guidance for LangSmith Deployment.

## 1. Build/config failures

### Symptom: invalid `langgraph.json`

Common causes:
- Invalid JSON syntax
- Missing required keys (`graphs`, `dependencies`)
- Invalid graph target path

Validate JSON:

```bash
uv run python -m json.tool langgraph.json
```

Minimal valid shape:

```json
{
  "graphs": {
    "agent": "./src/agent.py:graph"
  },
  "dependencies": ["."],
  "env": ".env"
}
```

Reference: https://docs.langchain.com/langsmith/cli

### Symptom: dependency import errors in deployed runtime

Checks:

1. Ensure dependencies are declared in `langgraph.json` `dependencies`.
2. Ensure graph path points to importable symbol (`./path/file.py:graph_obj`).
3. Ensure package/module layout is valid for your runtime.

Reference:
- https://docs.langchain.com/langsmith/cli
- https://docs.langchain.com/langsmith/application-structure

### Symptom: Python version mismatch

Set a supported runtime in `langgraph.json`:

```json
{
  "python_version": "3.12"
}
```

Supported values are `3.11`, `3.12`, `3.13`.

Reference: https://docs.langchain.com/langsmith/cli

## 2. Environment and secrets issues

### Symptom: required env vars missing at runtime

For standalone servers, verify all required vars are present:

- `REDIS_URI`
- `DATABASE_URI`
- `LANGSMITH_API_KEY`
- `LANGGRAPH_CLOUD_LICENSE_KEY`

Reference: https://docs.langchain.com/langsmith/deploy-standalone-server

### Symptom: tracing/auth confusion in control-plane deployments

In control-plane deployments, tracing project + tracing env wiring is managed by control plane.

Reference: https://docs.langchain.com/langsmith/control-plane

### Fast-fail startup check

```python
import os

required = ["OPENAI_API_KEY"]
missing = [k for k in required if not os.getenv(k)]
if missing:
    raise RuntimeError(f"Missing env vars: {missing}")
```

## 3. Datastore connectivity issues (standalone)

### Symptom: cannot connect to Postgres

- Verify `DATABASE_URI` format and credentials.
- Verify network path/security rules.
- Verify target database exists and is reachable.

### Symptom: cannot connect to Redis / streaming issues

- Verify `REDIS_URI` format and credentials.
- Verify network path/security rules.
- If sharing Redis, ensure distinct key/database namespace per deployment.

Reference: https://docs.langchain.com/langsmith/deploy-standalone-server

## 4. Deployment state issues

### Symptom: deployment revision not becoming active

- Check deployment events/logs in LangSmith UI.
- Confirm image/tag (for hybrid/self-hosted control-plane) exists and is pullable.
- Confirm data plane listener and operator health for self-hosted control-plane setups.

References:
- https://docs.langchain.com/langsmith/deploy-with-control-plane
- https://docs.langchain.com/langsmith/diagnostics-self-hosted

### Symptom: API behavior unexpected after deploy

Inspect the deployment's Agent Server OpenAPI at `/docs`.

Reference: https://docs.langchain.com/langsmith/server-api-ref

## 5. Performance and reliability issues

### Symptom: rising latency

- Check LangSmith dashboard trends for latency and errors.
- Profile expensive graph steps/tool calls.
- Scale replicas (hybrid/self-hosted) and tune datastore resources.

### Symptom: intermittent failures under load

- Add retries/backoff for provider/tool calls where appropriate.
- Validate provider rate limits and concurrency behavior.
- Re-run load tests after scaling/tuning changes.

References:
- https://docs.langchain.com/langsmith/dashboards
- https://docs.langchain.com/langsmith/alerts

## 6. Rollback strategy

Recommended rollback path:

1. Keep previous known-good revision available.
2. Re-activate known-good revision in UI or via Control Plane API.
3. Verify with SDK smoke tests.

Reference: https://docs.langchain.com/langsmith/api-ref-control-plane

## 7. Useful links

- Deployment overview: https://docs.langchain.com/langsmith/deployments
- Control plane API: https://docs.langchain.com/langsmith/api-ref-control-plane
- Agent Server API docs endpoint: https://docs.langchain.com/langsmith/server-api-ref
- Alerts: https://docs.langchain.com/langsmith/alerts
- Dashboards: https://docs.langchain.com/langsmith/dashboards
