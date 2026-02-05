# Deployment Targets

Configuration differences for local development vs. production deployment.

## Local Development

### Using `langgraph dev`

Lightweight development server for rapid iteration:

```bash
# Install CLI with in-memory backend
uv pip install "langgraph-cli[inmem]"

# Fallback if uv not available
pip install "langgraph-cli[inmem]"

# Start server
langgraph dev
```

**Features:**
- No Docker required
- Local persistence (state persisted to local directory)
- Hot reloading (auto-restart on code changes)
- Default port: 2024
- Lightweight resource usage
- Studio access requires a LangSmith API key

**Access Studio:**
```
https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024
```

### Using `langgraph up`

Production-like testing environment:

```bash
# Install CLI
uv pip install langgraph-cli

# Fallback if uv not available
pip install langgraph-cli

# Start server
langgraph up
```

**Features:**
- Docker required
- PostgreSQL state persistence
- Optional hot reloading with `--watch`
- Default port: 8123
- Production-like environment
- Requires LangSmith API key for local dev; license key for production use

## LangSmith Deployment (Cloud/Hybrid)

### Configuration

No special configuration needed beyond standard `langgraph.json`:

```json
{
  "dependencies": ["."],
  "graphs": {
    "agent": "./my_agent/agent.py:graph"
  },
  "env": ".env",
  "python_version": "3.11"
}
```

### Deployment

Deploy via Git integration or the LangSmith control plane (UI/API).

**Features:**
- Managed infrastructure
- Auto-scaling
- Built-in monitoring
- Persistent PostgreSQL storage

### Environment Variables

Set in LangSmith UI under deployment settings (not in `.env` file).

## Self-Hosted Deployment (Standalone or Control Plane)

### Requirements

- Docker and Docker Compose
- PostgreSQL database
- Redis (optional, for caching)

### Configuration

```json
{
  "dependencies": ["."],
  "graphs": {
    "agent": "./my_agent/agent.py:graph"
  },
  "python_version": "3.11",
  "base_image": "langchain/langgraph-server:0.2",
  "dockerfile_lines": [
    "RUN apt-get update && apt-get install -y postgresql-client"
  ]
}
```

### Build and Run

```bash
# Build Docker image
langgraph build

# Run with docker-compose.yml
docker-compose up
```

## Configuration Comparison

| Feature | langgraph dev | langgraph up | Cloud | Self-Hosted |
|---------|---------------|--------------|-------|-------------|
| Docker | ❌ | ✅ | N/A | ✅ |
| State Persistence | In-memory | PostgreSQL | PostgreSQL | PostgreSQL |
| Hot Reload | ✅ (default) | ⚠️  (--watch) | ❌ | ❌ |
| Default Port | 2024 | 8123 | 443 | Custom |
| Use Case | Development | Testing | Production | Production |

## Best Practices

### Development
- Use `langgraph dev` for fastest iteration
- Keep `.env` file for local credentials
- Use LangSmith tracing to debug

### Testing
- Use `langgraph up` before deploying
- Test with production-like data volumes
- Verify state persistence

### Production
- Set environment variables via deployment UI/config
- Enable monitoring and alerting
- Use appropriate scaling configuration
- Implement authentication
- Configure CORS policies

## References

- [Local Development Guide](https://docs.langchain.com/langsmith/local-dev-testing)
- [LangSmith Deployment Docs](https://docs.langchain.com/langsmith/deployments)
