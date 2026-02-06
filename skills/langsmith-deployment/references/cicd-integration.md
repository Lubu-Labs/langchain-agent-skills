# CI/CD Integration

This guide shows CI/CD patterns for LangSmith Deployment using currently documented workflows.

## What is officially supported

LangSmith docs describe two primary deployment paths in CI/CD:

1. **Cloud deployment**: Deploy from a GitHub repository (no Docker build step required).
2. **Hybrid / Self-hosted with control plane**: Build and push a Docker image, then create/update deployment via control plane.

Official reference:
- https://docs.langchain.com/langsmith/deployments
- https://docs.langchain.com/langsmith/cicd-pipeline-example
- https://docs.langchain.com/langsmith/api-ref-control-plane

## Recommended pipeline stages

1. **Validate**: lint, typecheck, and graph/config validation.
2. **Test**: unit/integration tests plus agent quality/evaluation checks.
3. **Package**: build image for Hybrid/Self-hosted control-plane deployments.
4. **Deploy**: create/update deployment through LangSmith UI or Control Plane API.
5. **Verify**: smoke test using LangGraph SDK.

## GitHub Actions skeleton

```yaml
name: LangSmith CI/CD

on:
  pull_request:
    branches: [main]
  push:
    branches: [main]

env:
  PYTHON_VERSION: "3.12"

jobs:
  validate-and-test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-python@v5
        with:
          python-version: ${{ env.PYTHON_VERSION }}

      - name: Install uv
        run: pip install uv

      - name: Install dependencies
        run: uv sync

      - name: Validate config
        run: uv run python -m json.tool langgraph.json

      - name: Run tests
        run: uv run pytest

  # Hybrid / Self-hosted with control plane
  build-image:
    if: github.ref == 'refs/heads/main'
    runs-on: ubuntu-latest
    needs: validate-and-test
    steps:
      - uses: actions/checkout@v4

      - name: Install LangGraph CLI
        run: pip install -U "langgraph-cli[inmem]"

      - name: Build image
        run: langgraph build -t my-registry/my-agent:${{ github.sha }}

      - name: Push image
        run: |
          docker login my-registry -u "${{ secrets.REGISTRY_USERNAME }}" -p "${{ secrets.REGISTRY_PASSWORD }}"
          docker push my-registry/my-agent:${{ github.sha }}

  # Deploy step intentionally uses your own helper wrapper around Control Plane API.
  # LangChain's official CI/CD example also uses a helper script for these calls.
  deploy:
    if: github.ref == 'refs/heads/main'
    runs-on: ubuntu-latest
    needs: [validate-and-test, build-image]
    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-python@v5
        with:
          python-version: ${{ env.PYTHON_VERSION }}

      - name: Install uv
        run: pip install uv

      - name: Deploy via control plane helper
        run: |
          uv run python .github/scripts/langgraph_api.py \
            --deployment-id "${{ secrets.PROD_DEPLOYMENT_ID }}" \
            --image "my-registry/my-agent:${{ github.sha }}"
```

## Cloud vs Hybrid/Self-hosted behavior

### Cloud
- Source of truth is your GitHub repo + `langgraph.json`.
- CI/CD typically creates/updates deployment revisions from git refs.

### Hybrid/Self-hosted with control plane
- CI/CD builds/pushes image artifacts.
- Deployment revisions reference pushed image tags.

## Required secrets (typical)

```bash
LANGSMITH_API_KEY=lsv2_...
PROD_DEPLOYMENT_ID=<deployment-id>
REGISTRY_USERNAME=<registry-user>
REGISTRY_PASSWORD=<registry-password>
```

If your deployment uses provider keys (for example OpenAI/Anthropic), store them in deployment environment variables/secrets, not in source control.

## PR preview environments

A common pattern (also shown in the official pipeline example):

1. On PR open/update, create a preview deployment.
2. Run smoke tests against that preview.
3. On merge to `main`, promote/create production revision and clean up preview.

## Smoke test example (SDK)

```python
import asyncio
from langgraph_sdk import get_client


async def main() -> None:
    client = get_client(
        url="<deployment-url>",
        api_key="<langsmith-api-key>",
    )

    async for chunk in client.runs.stream(
        None,
        "agent",
        input={"messages": [{"role": "human", "content": "health check"}]},
        stream_mode="updates",
    ):
        print(chunk.event)


if __name__ == "__main__":
    asyncio.run(main())
```

## Best practices

- Prefer immutable image tags (for example commit SHA) for deploy steps.
- Keep deployment config (`langgraph.json`) under version control.
- Run offline evals in CI before production rollout.
- Use staged promotion (preview -> staging -> production).
- Keep rollback procedure documented in your runbook.
