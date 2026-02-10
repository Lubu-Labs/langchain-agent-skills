# LangSmith Evaluation Guide

Guide to running reliable offline and online evaluations for LangGraph agents with LangSmith.

## Table of Contents

- [Setup](#setup)
- [Creating Datasets](#creating-datasets)
- [Running Evaluations](#running-evaluations)
- [Test Framework Integration](#test-framework-integration)
- [Custom Evaluators](#custom-evaluators)
- [Viewing Results](#viewing-results)
- [CI/CD Integration](#cicd-integration)
- [Best Practices](#best-practices)

## Setup

### Environment Variables

```bash
export LANGSMITH_API_KEY="your-api-key"
export LANGSMITH_TRACING="true"
export LANGSMITH_PROJECT="your-project-name"      # Optional
export LANGSMITH_WORKSPACE_ID="your-workspace-id" # Optional, if key spans workspaces
```

### Installation

```bash
# Python (LangSmith evaluate docs require >=0.3.13)
uv pip install -U "langsmith>=0.3.13"

# TypeScript
npm install langsmith
```

## Creating Datasets

### Via UI

1. Navigate to `https://smith.langchain.com`
2. Open `Datasets`
3. Create a dataset and add examples manually or by import

### Via SDK (Python)

```python
from langsmith import Client

client = Client()

dataset = client.create_dataset(
    dataset_name="agent_test_cases",
    description="Test cases for production agent",
)

examples = [
    {
        "inputs": {"question": "What's 2+2?"},
        "outputs": {"answer": "4"},
        "metadata": {"category": "math", "difficulty": "easy"},
    },
    {
        "inputs": {"question": "What is the capital of France?"},
        "outputs": {"answer": "Paris"},
        "metadata": {"category": "geography", "difficulty": "easy"},
    },
]

client.create_examples(dataset_id=dataset.id, examples=examples)
print(f"Created dataset: {dataset.name}")
```

### Via SDK (TypeScript)

```typescript
import { Client } from "langsmith";

const client = new Client();

const dataset = await client.createDataset("agent_test_cases", {
  description: "Test cases for production agent",
});

const inputs = [
  { question: "What's 2+2?" },
  { question: "What is the capital of France?" },
];
const outputs = [{ answer: "4" }, { answer: "Paris" }];
const metadata = [
  { category: "math", difficulty: "easy" },
  { category: "geography", difficulty: "easy" },
];

await client.createExamples({
  datasetId: dataset.id,
  inputs,
  outputs,
  metadata,
});
```

### From Production Traces (Python)

```python
from langsmith import Client

client = Client()

runs = client.list_runs(
    project_name="production",
    is_root=True,
    error=False,
    limit=100,
)

dataset = client.create_dataset(
    dataset_name="high_quality_interactions",
    description="Examples exported from production traces",
)

examples = [
    {"inputs": run.inputs, "outputs": run.outputs}
    for run in runs
    if run.inputs is not None and run.outputs is not None
]

client.create_examples(dataset_id=dataset.id, examples=examples)
```

## Running Evaluations

### Python SDK

```python
from langsmith import Client

client = Client()

def run_agent(inputs: dict) -> dict:
    result = agent.invoke(inputs)
    # Return a stable structure for evaluators
    return {"answer": result["messages"][-1].content}

def accuracy_evaluator(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
    score = outputs.get("answer") == reference_outputs.get("answer")
    return {"key": "accuracy", "score": score}

results = client.evaluate(
    run_agent,
    data="agent_test_cases",
    evaluators=[accuracy_evaluator],
    experiment_prefix="v1_evaluation",
    metadata={"version": "1.0"},
    max_concurrency=4,
)

print(f"Experiment: {results.experiment_name}")
```

Local-only run (do not upload to LangSmith):

```python
results = evaluate(
    run_agent,
    data="agent_test_cases",
    evaluators=[accuracy_evaluator],
    upload_results=False,
)
```

### TypeScript SDK

```typescript
import { evaluate } from "langsmith/evaluation";
import type { EvaluationResult } from "langsmith/evaluation";

async function runAgent(inputs: Record<string, any>): Promise<Record<string, any>> {
  const result = await agent.invoke(inputs);
  return { answer: result.messages[result.messages.length - 1]?.content ?? "" };
}

function accuracyEvaluator({
  outputs,
  referenceOutputs,
}: {
  outputs: Record<string, any>;
  referenceOutputs?: Record<string, any>;
}): EvaluationResult {
  const score = outputs.answer === referenceOutputs?.answer;
  return { key: "accuracy", score };
}

const results = await evaluate(runAgent, {
  data: "agent_test_cases",
  evaluators: [accuracyEvaluator],
  experimentPrefix: "v1_evaluation",
  metadata: { version: "1.0" },
  maxConcurrency: 4,
});

console.log(`Experiment: ${results.experimentName}`);
```

## Test Framework Integration

### Pytest

```python
import pytest

@pytest.mark.langsmith
def test_agent_basic_response():
    result = agent.invoke({"question": "What is 2+2?"})
    assert result is not None
```

Run with tracking:

```bash
LANGSMITH_TEST_SUITE="agent-tests" pytest --langsmith-output tests/
```

### Vitest (LangSmith Integration)

```typescript
import * as ls from "langsmith/vitest";
import { Client, type Example } from "langsmith";

const client = new Client();
const examples: Example[] = [];

for await (const ex of client.listExamples({ datasetName: "agent_test_cases" })) {
  examples.push(ex);
}

ls.describe("Agent tests", () => {
  ls.test.each(examples)("handles dataset example", async ({ inputs }) => {
    const result = await agent.invoke(inputs);
    ls.logOutputs(result);
  });
});
```

## Custom Evaluators

### Row-Level Evaluator (Python)

```python
def exact_match_evaluator(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
    actual = outputs.get("answer", "")
    expected = reference_outputs.get("answer", "")

    return {
        "key": "exact_match",
        "score": actual == expected,
        "comment": f"expected={expected!r} actual={actual!r}",
    }
```

### Summary Evaluator (TypeScript)

```typescript
import type { EvaluationResult } from "langsmith/evaluation";

function summaryEval({ outputs, referenceOutputs }: {
  outputs: Array<Record<string, any>>;
  referenceOutputs: Array<Record<string, any>>;
}): EvaluationResult {
  let correct = 0;
  for (let i = 0; i < outputs.length; i += 1) {
    if (outputs[i]?.answer === referenceOutputs[i]?.answer) correct += 1;
  }
  return { key: "summary_accuracy", score: outputs.length ? correct / outputs.length : 0 };
}
```

### LLM-as-Judge (Python via OpenEvals)

```python
from openevals.llm import create_llm_as_judge
from openevals.prompts import CORRECTNESS_PROMPT

correctness_judge = create_llm_as_judge(
    prompt=CORRECTNESS_PROMPT,
    model="openai:o3-mini",
    feedback_key="correctness",
)

# Inside evaluate(...), this can be passed as another evaluator.
```

## Viewing Results

### In LangSmith UI

1. Go to `https://smith.langchain.com`
2. Open `Experiments`
3. Select the experiment
4. Inspect aggregate stats and per-run traces

### Fetch Programmatically (Python)

```python
from langsmith import Client

client = Client()

experiment = client.read_project(
    project_name="v1_evaluation-abc123",
    include_stats=True,
)

print(f"Total runs: {experiment.run_count}")
print(f"Error rate: {experiment.error_rate:.2%}")
print(f"P50 latency: {experiment.latency_p50:.2f}s")
print("Feedback stats:", experiment.feedback_stats)
```

### Fetch Programmatically (TypeScript)

```typescript
import { Client } from "langsmith";

const client = new Client();

const experiment = await client.readProject({
  projectName: "v1_evaluation-abc123",
  includeStats: true,
});

console.log(experiment.feedbackStats);
```

## CI/CD Integration

### GitHub Actions Example

```yaml
name: Agent Evaluation

on:
  pull_request:
    branches: [main]

jobs:
  evaluate:
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.12"

      - name: Install dependencies
        run: |
          uv sync

      - name: Run evaluation
        env:
          LANGSMITH_API_KEY: ${{ secrets.LANGSMITH_API_KEY }}
          LANGSMITH_PROJECT: ci-evaluation
        run: |
          uv run skills/langgraph-testing-evaluation/scripts/evaluate_with_langsmith.py \
            my_agent:run_agent \
            ci_test_dataset \
            --evaluators accuracy,latency \
            --experiment-prefix ci-${{ github.sha }}

      - name: Check thresholds
        run: |
          uv run scripts/check_eval_thresholds.py \
            --experiment-name ci-${{ github.sha }} \
            --min-accuracy 0.8 \
            --max-latency 2.0
```

### Threshold Checking Script

```python
import argparse
import sys
from langsmith import Client

def check_thresholds(experiment_name: str, min_accuracy: float, max_latency: float) -> int:
    client = Client()
    exp = client.read_project(project_name=experiment_name, include_stats=True)

    accuracy = exp.feedback_stats.get("accuracy", {}).get("avg", 0)
    latency = exp.latency_p50

    passed = True

    if accuracy < min_accuracy:
        print(f"Accuracy {accuracy:.2f} below threshold {min_accuracy}")
        passed = False

    if latency > max_latency:
        print(f"Latency {latency:.2f}s above threshold {max_latency}s")
        passed = False

    if passed:
        print("All thresholds passed")
        return 0

    return 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-name", required=True)
    parser.add_argument("--min-accuracy", type=float, default=0.8)
    parser.add_argument("--max-latency", type=float, default=2.0)
    args = parser.parse_args()

    sys.exit(check_thresholds(args.experiment_name, args.min_accuracy, args.max_latency))
```

## Best Practices

1. Version datasets and keep dataset schemas stable.
2. Use consistent experiment naming (prefix with service/model/version).
3. Track both quality and operational metrics (latency/cost/error rate).
4. Prefer deterministic evaluators in CI, and use LLM-as-judge where needed.
5. Keep evaluator prompts and logic versioned.
6. Compare against a baseline before rollout.
7. Automate threshold checks in CI/CD.
8. Inspect failed traces, not only aggregate scores.
