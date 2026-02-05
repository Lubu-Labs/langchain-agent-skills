# LangGraph.json Configuration Reference

Complete reference for the `langgraph.json` configuration file.

## Overview

The `langgraph.json` file specifies dependencies, graphs, environment variables, and other settings required to deploy a LangGraph application. The LangGraph CLI uses this file to build and run your application.

## Required Fields

### dependencies

**Type:** `array` of strings
**Required:** Yes

Array of dependencies for the application. Can be:
- `"."` - Current directory (looks for `pyproject.toml`, `setup.py`, `requirements.txt`, or `package.json`)
- `"./path"` - Relative path to a directory with package configuration
- `"package-name"` or `"package-name==1.0.0"` - Package name (optionally with version)

**Examples:**
```json
{
  "dependencies": ["."]
}
```

```json
{
  "dependencies": ["./my_package", "langchain==0.3.0"]
}
```

### graphs

**Type:** `object` (string → string mapping)
**Required:** Yes

Mapping from graph ID to path where the compiled graph or factory function is defined.

**Format:** `./path/to/file.{py|js|ts}:variable` or `package.module:variable`

Where `variable` is either:
- An instance of `CompiledStateGraph`
- A function that returns a `StateGraph` or `CompiledStateGraph`

**Examples:**
```json
{
  "graphs": {
    "agent": "./my_agent/agent.py:graph",
    "chatbot": "./my_agent/chatbot.py:create_graph"
  }
}
```

## Optional Fields

### env

**Type:** `string` (path) or `object` (key-value pairs)
**Default:** None

Path to `.env` file or direct environment variable definitions.

**Examples:**
```json
{
  "env": ".env"
}
```

```json
{
  "env": {
    "MY_VAR": "value",
    "OPENAI_API_KEY": "sk-..."
  }
}
```

### python_version

**Type:** `string`
**Options:** `"3.11"`, `"3.12"`, `"3.13"`
**Default:** `"3.11"`

Python version to use (Python projects only).

**Example:**
```json
{
  "python_version": "3.12"
}
```

### node_version

**Type:** `string`
**Options:** `"20"`
**Default:** None (omit for Python projects)

Specifying `node_version: "20"` indicates a JavaScript/TypeScript project.

**Example:**
```json
{
  "node_version": "20"
}
```

### base_image

**Type:** `string`
**Default:** `"langchain/langgraph-api"` (Python) or `"langchain/langgraphjs-api"` (JavaScript)

Base Docker image for the LangGraph API server.
Use a versioned tag to pin (for example, `langchain/langgraph-server:0.2`).

**Example:**
```json
{
  "base_image": "langchain/langgraph-server:0.2"
}
```

### image_distro

**Type:** `string`
**Options:** `"debian"`, `"wolfi"`, `"bookworm"`, `"bullseye"`
**Default:** `"debian"`
**Minimum CLI version:** `0.2.11`

Linux distribution for the base image.

**Example:**
```json
{
  "image_distro": "wolfi"
}
```

### pip_installer

**Type:** `string`
**Options:** `"auto"`, `"pip"`, `"uv"`
**Default:** `"auto"` (uses `uv` if available)
**Minimum CLI version:** `0.3.0`

Python package installer to use.

**Example:**
```json
{
  "pip_installer": "uv"
}
```

### pip_config_file

**Type:** `string`
**Default:** None

Path to pip configuration file.

**Example:**
```json
{
  "pip_config_file": "./pip.conf"
}
```

### keep_pkg_tools

**Type:** `boolean` or `array` of strings
**Options:** `true`, `false`, or array with `["pip", "setuptools", "wheel"]`
**Default:** `false` (uninstall all)
**Minimum CLI version:** `0.3.4`

Controls whether to retain Python packaging tools in the final image.

**Example:**
```json
{
  "keep_pkg_tools": ["pip", "setuptools"]
}
```

### dockerfile_lines

**Type:** `array` of strings
**Default:** `[]`

Additional Dockerfile lines to add after the base image import.

**Example:**
```json
{
  "dockerfile_lines": [
    "RUN apt-get update && apt-get install -y postgresql-client",
    "ENV CUSTOM_VAR=value"
  ]
}
```

### checkpointer

**Type:** `object`
**Default:** None

Configuration for the checkpointer (state persistence).

**Sub-fields:**
- `ttl`: Object with `strategy`, `sweep_interval_minutes`, `default_ttl`
- `serde` (v0.5+): Object with `allowed_json_modules` and `pickle_fallback`

**Example:**
```json
{
  "checkpointer": {
    "ttl": {
      "strategy": "delete",
      "sweep_interval_minutes": 60,
      "default_ttl": 1440
    },
    "serde": {
      "allowed_json_modules": ["my_module"],
      "pickle_fallback": true
    }
  }
}
```

### store

**Type:** `object`
**Default:** None

Configuration for semantic search and TTL in BaseStore.

**Sub-fields:**
- `index`: Configuration for semantic search
  - `embed`: Embedding model configuration
  - `dims`: Embedding dimensions
  - `fields`: Fields to index
- `ttl`: Time-to-live configuration
  - `refresh_on_read`: Boolean (default: `true`)
  - `default_ttl`: Float (minutes)
  - `sweep_interval_minutes`: Integer

**Example:**
```json
{
  "store": {
    "index": {
      "embed": "openai:text-embedding-3-small",
      "dims": 1536,
      "fields": ["$"]
    },
    "ttl": {
      "default_ttl": 60,
      "sweep_interval_minutes": 10
    }
  }
}
```

### http

**Type:** `object`
**Default:** None

HTTP server configuration.

**Sub-fields:**
- `app`: Path to custom Starlette/FastAPI app
- `cors`: CORS configuration
- `configurable_headers`: Define which headers to expose
- `logging_headers`: Exclude sensitive headers from logs
- `middleware_order`: `"auth_first"` or `"middleware_first"` (default)
- `enable_custom_route_auth`: Apply auth to custom routes
- `disable_*`: Disable built-in routes (`disable_a2a`, `disable_assistants`, `disable_mcp`, `disable_runs`, `disable_store`, `disable_threads`, `disable_ui`, `disable_webhooks`, `disable_meta`)
- `mount_prefix`: Prefix for mounted routes

**Example:**
```json
{
  "http": {
    "cors": {
      "allow_origins": ["https://example.com"],
      "allow_methods": ["GET", "POST"],
      "allow_credentials": true
    },
    "configurable_headers": {
      "includes": ["x-user-id"],
      "excludes": ["authorization"]
    }
  }
}
```

### webhooks

**Type:** `object`
**Default:** None
**Minimum CLI version:** `0.5.36`

Configuration for outbound webhook delivery.

**Sub-fields:**
- `env_prefix`: Required prefix for env vars (default: `"LG_WEBHOOK_"`)
- `headers`: Static headers for webhooks
- `url`: URL validation policy (`allowed_domains`, `allowed_ports`, `require_https`, `disable_loopback`, `max_url_length`)

**Example:**
```json
{
  "webhooks": {
    "env_prefix": "LG_WEBHOOK_",
    "headers": {
      "X-Custom-Header": "${{ env.HEADER_VALUE }}"
    },
    "url": {
      "require_https": true,
      "disable_loopback": true
    }
  }
}
```

### auth

**Type:** `object`
**Default:** None
**Minimum CLI version:** `0.0.11`

Authentication configuration. `path` uses format `./path/to/auth.py:auth`.

**Example:**
```json
{
  "auth": {
    "path": "./my_agent/auth.py:auth"
  }
}
```

### ui

**Type:** `object`
**Default:** None
**Minimum CLI version:** `0.1.84`

Named definitions of UI components emitted by the agent.

**Example:**
```json
{
  "ui": {
    "my_component": "./ui/component.js"
  }
}
```

### api_version

**Type:** `string`
**Default:** Latest
**Minimum CLI version:** `0.3.7`

Semantic version of the LangGraph API server to use.

**Example:**
```json
{
  "api_version": "0.3"
}
```

## Complete Example

### Python Project

```json
{
  "dependencies": ["."],
  "graphs": {
    "agent": "./my_agent/agent.py:graph"
  },
  "env": ".env",
  "python_version": "3.11",
  "pip_installer": "uv",
  "checkpointer": {
    "ttl": {
      "default_ttl": 1440
    }
  },
  "http": {
    "cors": {
      "allow_origins": ["*"],
      "allow_methods": ["GET", "POST", "PUT", "DELETE"]
    }
  }
}
```

### JavaScript Project

```json
{
  "dependencies": ["."],
  "graphs": {
    "agent": "./src/agent.ts:graph"
  },
  "env": ".env",
  "node_version": "20",
  "http": {
    "cors": {
      "allow_origins": ["*"]
    }
  }
}
```

## Validation

Use the provided validation script to check your configuration:

```bash
uv run scripts/validate_langgraph_config.py

# Fallback if uv not available
python3 scripts/validate_langgraph_config.py
```

## References

- [LangGraph CLI Documentation](https://docs.langchain.com/langsmith/cli)
- [Application Structure Guide](https://docs.langchain.com/oss/python/langgraph/application-structure)
