# Testing and Evaluation Assets

This directory currently contains documentation only. Runnable starter assets live in:

- `../templates/test_template.py`
- `../datasets/sample_dataset.json`

## What to Use Today

1. Use `../templates/test_template.py` as a baseline for LangGraph unit/integration tests.
2. Use `../datasets/sample_dataset.json` as seed data for LangSmith datasets.
3. Use `../../references/` for deeper evaluation patterns (unit tests, trajectory evals, A/B testing, LangSmith evaluation).

## Quick Start (Python)

```bash
# 1) Copy the template into your project
cp skills/langgraph-testing-evaluation/assets/templates/test_template.py tests/test_graph.py

# 2) Run tests (preferred)
uv run pytest tests/test_graph.py
```

## LangSmith Dataset Upload Example

```python
from pathlib import Path
import json
from langsmith import Client

client = Client()
dataset = client.create_dataset(dataset_name="my-agent-dataset")

payload = json.loads(
    Path("skills/langgraph-testing-evaluation/assets/datasets/sample_dataset.json").read_text()
)
client.create_examples(dataset_id=dataset.id, examples=payload["examples"])
```

## Official Documentation

- LangGraph testing: https://docs.langchain.com/oss/python/langgraph/test
- LangSmith evaluate guide: https://docs.langchain.com/langsmith/evaluate-llm-application
- LangSmith dataset management: https://docs.langchain.com/langsmith/manage-datasets-programmatically
