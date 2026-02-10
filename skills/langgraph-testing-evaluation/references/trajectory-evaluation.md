# Trajectory Evaluation for LangGraph Agents

Trajectory evaluation assesses the sequence of actions and decisions an agent makes, not just the final output. This is critical for multi-step reasoning agents where the execution path matters.

## Table of Contents

- [What is Trajectory Evaluation](#what-is-trajectory-evaluation)
- [Trajectory Match Evaluator](#trajectory-match-evaluator)
- [LLM-as-Judge Evaluator](#llm-as-judge-evaluator)
- [Creating Datasets](#creating-datasets)
- [Using AgentEvals with LangSmith](#using-agentevals-with-langsmith)
- [Best Practices](#best-practices)
- [Common Patterns](#common-patterns)

## What is Trajectory Evaluation

A trajectory is the complete sequence of:
- Messages exchanged
- Tool calls made
- Decisions at conditional edges
- State transformations

Output-only evaluation misses important quality signals like:
- Did the agent call the right tools?
- Were tools called in a logical order?
- Did the agent make unnecessary steps?
- Was the reasoning process sound?

## Trajectory Match Evaluator

Use trajectory match when you want deterministic validation against a reference trajectory.

AgentEvals supports four match modes:
- `strict`: exact sequence match
- `unordered`: same tool calls, order does not matter
- `subset`: no tool calls outside reference
- `superset`: includes all required reference tool calls (extras allowed)

### Python Example

```python
from langchain.messages import HumanMessage, AIMessage, ToolMessage
from agentevals.trajectory.match import create_trajectory_match_evaluator

# Strict match of the trajectory
trajectory_evaluator = create_trajectory_match_evaluator(
    trajectory_match_mode="strict"
)

reference_trajectory = [
    HumanMessage(content="What's the weather in San Francisco?"),
    AIMessage(
        content="",
        tool_calls=[
            {"id": "call_1", "name": "get_weather", "args": {"city": "San Francisco"}}
        ],
    ),
    ToolMessage(content="It's 72F and sunny in San Francisco.", tool_call_id="call_1"),
    AIMessage(content="The weather in San Francisco is 72F and sunny."),
]

result = agent.invoke({
    "messages": [HumanMessage(content="What's the weather in San Francisco?")]
})

evaluation = trajectory_evaluator(
    outputs=result["messages"],
    reference_outputs=reference_trajectory,
)

assert evaluation["score"] is True
```

### TypeScript Example

```typescript
import { HumanMessage, AIMessage, ToolMessage } from "@langchain/core/messages";
import { createTrajectoryMatchEvaluator } from "agentevals";

const trajectoryEvaluator = createTrajectoryMatchEvaluator({
  trajectoryMatchMode: "strict",
});

const referenceTrajectory = [
  new HumanMessage("What's the weather in San Francisco?"),
  new AIMessage({
    content: "",
    tool_calls: [{ id: "call_1", name: "get_weather", args: { city: "San Francisco" } }],
  }),
  new ToolMessage({
    content: "It's 72F and sunny in San Francisco.",
    tool_call_id: "call_1",
  }),
  new AIMessage("The weather in San Francisco is 72F and sunny."),
];

const result = await agent.invoke({
  messages: [new HumanMessage("What's the weather in San Francisco?")],
});

const evaluation = await trajectoryEvaluator({
  outputs: result.messages,
  referenceOutputs: referenceTrajectory,
});

expect(evaluation.score).toBe(true);
```

## LLM-as-Judge Evaluator

Use LLM-as-judge when multiple trajectories may be valid and you want qualitative scoring.

### Python Example

```python
from agentevals.trajectory.llm import (
    create_trajectory_llm_as_judge,
    TRAJECTORY_ACCURACY_PROMPT,
    TRAJECTORY_EFFICIENCY_PROMPT,
    TRAJECTORY_ACCURACY_PROMPT_WITH_REFERENCE,
)

accuracy_evaluator = create_trajectory_llm_as_judge(
    model="openai:o3-mini",
    prompt=TRAJECTORY_ACCURACY_PROMPT,
)

efficiency_evaluator = create_trajectory_llm_as_judge(
    model="openai:o3-mini",
    prompt=TRAJECTORY_EFFICIENCY_PROMPT,
)

with_reference_evaluator = create_trajectory_llm_as_judge(
    model="openai:o3-mini",
    prompt=TRAJECTORY_ACCURACY_PROMPT_WITH_REFERENCE,
)

result = agent.invoke({"messages": [{"role": "user", "content": "What's the weather in Seattle?"}]})

evaluation = accuracy_evaluator(outputs=result["messages"])
# If you have a reference trajectory, use reference_outputs:
# evaluation = with_reference_evaluator(outputs=result["messages"], reference_outputs=reference_trajectory)

assert evaluation["score"] is True
```

### TypeScript Example

```typescript
import {
  createTrajectoryLLMAsJudge,
  TRAJECTORY_ACCURACY_PROMPT,
  TRAJECTORY_EFFICIENCY_PROMPT,
  TRAJECTORY_ACCURACY_PROMPT_WITH_REFERENCE,
} from "agentevals";

const accuracyEvaluator = createTrajectoryLLMAsJudge({
  model: "openai:o3-mini",
  prompt: TRAJECTORY_ACCURACY_PROMPT,
});

const efficiencyEvaluator = createTrajectoryLLMAsJudge({
  model: "openai:o3-mini",
  prompt: TRAJECTORY_EFFICIENCY_PROMPT,
});

const withReferenceEvaluator = createTrajectoryLLMAsJudge({
  model: "openai:o3-mini",
  prompt: TRAJECTORY_ACCURACY_PROMPT_WITH_REFERENCE,
});

const result = await agent.invoke({
  messages: [{ role: "user", content: "What's the weather in Seattle?" }],
});

const evaluation = await accuracyEvaluator({
  outputs: result.messages,
});

// If you have a reference trajectory, pass referenceOutputs:
// const evaluation = await withReferenceEvaluator({
//   outputs: result.messages,
//   referenceOutputs: referenceTrajectory,
// });

expect(evaluation.score).toBe(true);
```

## Creating Datasets

### Dataset Structure

```json
{
  "examples": [
    {
      "inputs": {
        "messages": [
          { "role": "user", "content": "What's the weather in SF?" }
        ]
      },
      "outputs": {
        "messages": [
          { "role": "assistant", "content": "The weather in SF is 72F and sunny." }
        ]
      },
      "metadata": {
        "expected_tools": ["search"],
        "difficulty": "easy"
      }
    }
  ]
}
```

### Python SDK

```python
from langsmith import Client

client = Client()

dataset = client.create_dataset(
    dataset_name="weather_agent_trajectories",
    description="Test cases for weather agent trajectory evaluation",
)

examples = [
    {
        "inputs": {"messages": [{"role": "user", "content": "Weather in SF?"}]},
        "outputs": {"messages": [{"role": "assistant", "content": "72F and sunny."}]},
        "metadata": {"expected_tools": ["search"]},
    },
]

client.create_examples(dataset_id=dataset.id, examples=examples)
```

### TypeScript SDK

```typescript
import { Client } from "langsmith";

const client = new Client();

const dataset = await client.createDataset("weather_agent_trajectories", {
  description: "Test cases for weather agent trajectory evaluation",
});

const inputs = [{ messages: [{ role: "user", content: "Weather in SF?" }] }];
const outputs = [{ messages: [{ role: "assistant", content: "72F and sunny." }] }];
const metadata = [{ expected_tools: ["search"] }];

await client.createExamples({
  datasetId: dataset.id,
  inputs,
  outputs,
  metadata,
});
```

## Using AgentEvals with LangSmith

### Installation

```bash
# Python
uv pip install agentevals langsmith

# TypeScript
npm install agentevals langsmith
```

### Python Quick Start

```python
from agentevals.trajectory.llm import create_trajectory_llm_as_judge, TRAJECTORY_ACCURACY_PROMPT
from langsmith import Client

trajectory_evaluator = create_trajectory_llm_as_judge(
    model="openai:o3-mini",
    prompt=TRAJECTORY_ACCURACY_PROMPT,
)

def run_agent(inputs):
    result = agent.invoke(inputs)
    return result["messages"]

client = Client()
results = client.evaluate(
    run_agent,
    data="my_dataset",
    evaluators=[trajectory_evaluator],
    experiment_prefix="trajectory-eval",
)

print(results.experiment_name)
```

### TypeScript Quick Start

```typescript
import { createTrajectoryLLMAsJudge, TRAJECTORY_ACCURACY_PROMPT } from "agentevals";
import { evaluate } from "langsmith/evaluation";

const evaluator = createTrajectoryLLMAsJudge({
  model: "openai:o3-mini",
  prompt: TRAJECTORY_ACCURACY_PROMPT,
});

async function runAgent(inputs: any) {
  const result = await agent.invoke(inputs);
  return result.messages;
}

const results = await evaluate(runAgent, {
  data: "my_dataset",
  evaluators: [evaluator],
  experimentPrefix: "trajectory-eval",
});

console.log(results.experimentName);
```

### Custom Evaluation Criteria

```python
from agentevals.trajectory.llm import create_trajectory_llm_as_judge

CUSTOM_TRAJECTORY_PROMPT = """
Evaluate the trajectory using the following rubric:
1. Tool selection quality
2. Tool call order
3. Redundant steps
4. Final answer quality

Trajectory: {outputs}
Reference trajectory (optional): {reference_outputs}
"""

custom_evaluator = create_trajectory_llm_as_judge(
    model="openai:o3-mini",
    prompt=CUSTOM_TRAJECTORY_PROMPT,
)
```

## Best Practices

1. Start with deterministic trajectory matching for well-defined flows.
2. Use LLM-as-judge when multiple trajectories can be correct.
3. Prefer `openai:o3-mini` (or another current evaluator model) for judge evaluators.
4. Version trajectory datasets as your agent behavior evolves.
5. Include edge cases and failure paths in trajectory datasets.
6. Track evaluation cost and latency for LLM-as-judge runs.
7. Inspect failed runs in LangSmith traces, not just aggregate scores.
8. Keep evaluator prompts explicit and stable over time.

## Common Patterns

### Tool Call Validation

```python
def validate_tool_calls(run, example):
    """Custom evaluator for tool call validation."""
    messages = run.outputs.get("messages", [])
    expected_tools = example.metadata.get("expected_tools", [])

    actual_tools = []
    for msg in messages:
        for tool_call in msg.get("tool_calls", []):
            actual_tools.append(tool_call["name"])

    score = 1.0 if set(expected_tools).issubset(set(actual_tools)) else 0.0

    return {
        "key": "tool_call_validation",
        "score": score,
        "comment": f"Expected: {expected_tools}, Actual: {actual_tools}",
    }
```

### Compare Two Agent Versions

```python
from langsmith import Client

client = Client()

results_v1 = client.evaluate(agent_v1, data="test_dataset", evaluators=[evaluator], experiment_prefix="v1")
results_v2 = client.evaluate(agent_v2, data="test_dataset", evaluators=[evaluator], experiment_prefix="v2")

exp_v1 = client.read_project(project_name=results_v1.experiment_name, include_stats=True)
exp_v2 = client.read_project(project_name=results_v2.experiment_name, include_stats=True)

print("V1 feedback stats:", exp_v1.feedback_stats)
print("V2 feedback stats:", exp_v2.feedback_stats)
```
