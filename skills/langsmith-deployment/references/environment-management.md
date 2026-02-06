# Environment Variable Management

Guide for managing environment variables and secrets across LangSmith deployment modes.

## Sources of environment variables

Current docs support these common patterns:

1. `langgraph.json` `env` field (path to `.env` or key/value mapping)
2. Deployment UI / control plane settings
3. Container runtime env (Kubernetes, Docker, VM)
4. External secret managers (AWS/GCP/Azure)

Reference: https://docs.langchain.com/langsmith/cli

## `langgraph.json` env configuration

The `env` field can be either:

- a file path (`".env"`), or
- an inline mapping of keys to values.

```json
{
  "graphs": {
    "agent": "./src/agent.py:graph"
  },
  "dependencies": ["."],
  "env": ".env"
}
```

Or:

```json
{
  "graphs": {
    "agent": "./src/agent.py:graph"
  },
  "dependencies": ["."],
  "env": {
    "OPENAI_API_KEY": "",
    "LANGSMITH_PROJECT": "my-agent-dev"
  }
}
```

Best practice:
- Commit only placeholders in git-tracked config.
- Keep real secrets in deployment environment or secret manager.

## Cloud and control-plane deployments

For deployments managed by LangSmith control plane:

- A tracing project is created per deployment.
- `LANGCHAIN_TRACING` and `LANGSMITH_API_KEY` / `LANGCHAIN_API_KEY` do not need to be manually set during deployment creation; control plane sets them.

Reference: https://docs.langchain.com/langsmith/control-plane

## Standalone server required env vars

Per standalone deployment docs, these are required:

- `REDIS_URI`
- `DATABASE_URI`
- `LANGSMITH_API_KEY`
- `LANGGRAPH_CLOUD_LICENSE_KEY`

Optional:
- `LANGSMITH_ENDPOINT` (when sending traces to self-hosted LangSmith)

Reference: https://docs.langchain.com/langsmith/deploy-standalone-server

## Secret management patterns

### Kubernetes secrets

```bash
kubectl create secret generic agent-secrets \
  --from-literal=OPENAI_API_KEY=<value> \
  --from-literal=LANGSMITH_API_KEY=<value> \
  --namespace langsmith
```

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: my-agent
spec:
  template:
    spec:
      containers:
        - name: agent
          image: my-agent:latest
          env:
            - name: OPENAI_API_KEY
              valueFrom:
                secretKeyRef:
                  name: agent-secrets
                  key: OPENAI_API_KEY
            - name: LANGSMITH_API_KEY
              valueFrom:
                secretKeyRef:
                  name: agent-secrets
                  key: LANGSMITH_API_KEY
```

### AWS Secrets Manager (example)

```python
import boto3


def get_secret(secret_id: str) -> str:
    client = boto3.client("secretsmanager")
    return client.get_secret_value(SecretId=secret_id)["SecretString"]
```

### GCP Secret Manager (example)

```python
from google.cloud import secretmanager


def get_secret(project_id: str, secret_id: str, version: str = "latest") -> str:
    client = secretmanager.SecretManagerServiceClient()
    name = f"projects/{project_id}/secrets/{secret_id}/versions/{version}"
    response = client.access_secret_version(request={"name": name})
    return response.payload.data.decode("utf-8")
```

## Validation snippet

Run this during startup/CI to fail fast on missing required keys:

```python
import os
import sys

REQUIRED = ["OPENAI_API_KEY"]

missing = [k for k in REQUIRED if not os.getenv(k)]
if missing:
    print(f"Missing required environment variables: {missing}")
    sys.exit(1)
```

For standalone deployments, include `DATABASE_URI`, `REDIS_URI`, `LANGSMITH_API_KEY`, and `LANGGRAPH_CLOUD_LICENSE_KEY` in required checks.

## Security checklist

- Never commit real secrets.
- Use separate credentials per environment.
- Rotate secrets on a regular schedule.
- Use least-privilege IAM/service-account permissions.
- Prefer managed secret stores over plaintext files.
