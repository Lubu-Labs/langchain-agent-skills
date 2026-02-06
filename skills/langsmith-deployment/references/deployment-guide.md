# LangSmith Deployment Guide

Current deployment guidance for LangChain/LangGraph apps on LangSmith.

## Deployment options

LangSmith supports three deployment patterns:

1. **Cloud**: Managed hosting; deploy from GitHub.
2. **Hybrid / Self-hosted with control plane**: Control plane + data plane in your environment.
3. **Standalone server**: Run Agent Server directly without the control plane UI.

Official references:
- https://docs.langchain.com/langsmith/deployments
- https://docs.langchain.com/langsmith/deploy-hybrid
- https://docs.langchain.com/langsmith/self-hosted
- https://docs.langchain.com/langsmith/deploy-with-control-plane
- https://docs.langchain.com/langsmith/deploy-standalone-server

## Prerequisites (all deployment modes)

A deployable app should include:

- `langgraph.json`
- Graph implementation(s)
- Dependency definition (`pyproject.toml`, `requirements.txt`, or `package.json`)
- Optional `.env` file for local development

### Minimal `langgraph.json` (Python)

```json
{
  "graphs": {
    "agent": "./src/agent.py:graph"
  },
  "dependencies": ["."],
  "env": ".env",
  "python_version": "3.12",
  "image_distro": "wolfi",
  "pip_installer": "uv"
}
```

Notes based on current config reference:
- `graphs` and `dependencies` are required.
- `python_version` supports `3.11`, `3.12`, `3.13`.
- `pip_installer` supports `auto`, `pip`, `uv`.

Reference: https://docs.langchain.com/langsmith/cli

## Cloud deployment

Cloud is the simplest path: connect repo in LangSmith and deploy from branch/commit.

Typical flow:

1. Connect GitHub integration in LangSmith.
2. Create deployment from repository.
3. Set environment variables in deployment config.
4. Create revisions from new commits.
5. Observe traces, dashboards, and alerts.

Reference:
- https://docs.langchain.com/oss/python/langgraph/deploy
- https://docs.langchain.com/langsmith/deployment-quickstart

## Hybrid / Self-hosted with control plane

Use when you need a managed deployment UX with infrastructure in your own cloud.

High-level flow:

1. Build image with LangGraph CLI.
2. Push image to registry accessible by your cluster.
3. Create/update deployment in LangSmith control plane.

```bash
# Build image
langgraph build -t my-registry/my-agent:v1.0.0

# Push image
docker push my-registry/my-agent:v1.0.0
```

References:
- https://docs.langchain.com/langsmith/deploy-with-control-plane
- https://docs.langchain.com/langsmith/api-ref-control-plane

## Standalone server deployment

Standalone is the lightest self-hosted option (no control plane UI).

Required environment variables (documented):

- `REDIS_URI`
- `DATABASE_URI`
- `LANGSMITH_API_KEY`
- `LANGGRAPH_CLOUD_LICENSE_KEY`

Optional:

- `LANGSMITH_ENDPOINT` (for self-hosted LangSmith tracing endpoint)

Reference: https://docs.langchain.com/langsmith/deploy-standalone-server

### Important standalone notes

- Do **not** run standalone servers in serverless/scale-to-zero environments.
- Allow egress to `https://beacon.langchain.com` for license verification/usage reporting unless air-gapped mode is configured.

References:
- https://docs.langchain.com/langsmith/self-hosted
- https://docs.langchain.com/langsmith/deploy-standalone-server

## Verify a deployment with SDK

Use LangGraph SDK for smoke tests.

```python
import asyncio
from langgraph_sdk import get_client


async def smoke_test() -> None:
    client = get_client(url="<deployment-url>", api_key="<langsmith-api-key>")

    async for chunk in client.runs.stream(
        None,
        "agent",
        input={
            "messages": [{"role": "human", "content": "What is LangGraph?"}]
        },
        stream_mode="updates",
    ):
        print(chunk.event)


if __name__ == "__main__":
    asyncio.run(smoke_test())
```

Reference: https://docs.langchain.com/langsmith/deployment-quickstart

## Operational best practices

- Keep `langgraph.json` minimal and explicit.
- Keep secrets out of source control.
- Use immutable image tags for deployments.
- Validate with CI tests + evaluation before production.
- Use project dashboards and project-scoped alerts after deployment.
