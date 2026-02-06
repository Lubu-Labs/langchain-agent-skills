# Scaling Configuration

Guide for scaling LangSmith deployments with currently documented controls.

## Deployment model and scaling responsibility

### Cloud

LangSmith manages infrastructure/scaling for Cloud deployments.

Use project monitoring and alerts to detect saturation and contact support for plan-specific capacity guidance.

Reference: https://docs.langchain.com/langsmith/deployments

### Hybrid / Self-hosted with control plane

You manage Kubernetes capacity for the data plane (Agent Servers + backing services).

Use Kubernetes scaling primitives such as:

- `HorizontalPodAutoscaler` for Agent Server deployments
- CPU/memory requests and limits
- registry/image rollout strategy and health probes

Reference architecture concepts:
- https://docs.langchain.com/langsmith/data-plane
- https://docs.langchain.com/langsmith/components

### Standalone server

Standalone gives full scaling control but you own reliability and orchestration.

Important:
- Do not use standalone deployments in serverless scale-to-zero environments.

Reference: https://docs.langchain.com/langsmith/self-hosted

## Documented server tuning knobs

### Postgres connection pool

`LANGGRAPH_POSTGRES_POOL_MAX_SIZE` (API server `0.2.12+`) controls max Postgres connections per replica.

- Default: `150`
- Total possible connections scale with replica count

Reference: https://docs.langchain.com/langsmith/env-var

### Redis key namespace

`REDIS_KEY_PREFIX` (API server `0.1.9+`) allows multiple Agent Server instances to share one Redis instance with key separation.

Reference: https://docs.langchain.com/langsmith/env-var

## Standalone datastore guidance

For standalone Agent Servers:

- `REDIS_URI` must point to a valid Redis URI.
- `DATABASE_URI` must point to a valid Postgres URI.
- Separate deployments should use separate Redis DB/keyspace and separate Postgres databases.

Reference: https://docs.langchain.com/langsmith/deploy-standalone-server

## Kubernetes baseline example

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: my-agent-hpa
  namespace: langsmith
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: my-agent
  minReplicas: 2
  maxReplicas: 10
  metrics:
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: 70
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
          resources:
            requests:
              cpu: "500m"
              memory: "1Gi"
            limits:
              cpu: "2000m"
              memory: "4Gi"
```

These values are starting points; tune from real load tests and SLOs.

## Load testing pattern (SDK)

```python
import asyncio
from langgraph_sdk import get_client


async def load_test(url: str, assistant_id: str, n: int = 50) -> None:
    client = get_client(url=url)

    async def one() -> bool:
        try:
            async for _chunk in client.runs.stream(
                None,
                assistant_id,
                input={"messages": [{"role": "human", "content": "ping"}]},
                stream_mode="updates",
            ):
                pass
            return True
        except Exception:
            return False

    results = await asyncio.gather(*[one() for _ in range(n)])
    print(f"success={sum(results)}/{n}")


if __name__ == "__main__":
    asyncio.run(load_test("<deployment-url>", "agent"))
```

## Scaling checklist

- Set explicit CPU/memory requests and limits.
- Enable HPA for Agent Server workloads.
- Track p95 latency and error rates in LangSmith dashboards.
- Tune Postgres pool with `LANGGRAPH_POSTGRES_POOL_MAX_SIZE` when needed.
- Use `REDIS_KEY_PREFIX` when sharing Redis across deployments.
- Re-load-test after major graph/model changes.
