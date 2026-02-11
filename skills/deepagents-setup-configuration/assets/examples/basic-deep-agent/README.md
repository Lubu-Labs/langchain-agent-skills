# Basic Deep Agent Example

Minimal working Deep Agent example in Python.

## Files

- `agent.py`: Python example

## Setup

Install dependencies (latest compatible ecosystem packages):

```bash
uv init
uv add deepagents langchain langgraph langchain-anthropic python-dotenv
uv sync
```

Set your model provider key:

```bash
export ANTHROPIC_API_KEY="your-key-here"
# or
export OPENAI_API_KEY="your-key-here"
```

Optional LangSmith tracing:

```bash
export LANGSMITH_TRACING=true
export LANGSMITH_API_KEY="your-langsmith-key"
export LANGSMITH_PROJECT="basic-deep-agent"
```

## Run

```bash
uv run agent.py
```

## Notes

- The example demonstrates custom tools, message-based invocation, and interactive mode.
- It uses Anthropic model configuration and requires valid credentials.
