# A/B Testing for LangGraph Agents

Guide to comparing agent versions and making data-driven decisions about changes.

## Table of Contents

- [When to A/B Test](#when-to-ab-test)
- [Offline A/B Testing](#offline-ab-testing)
- [Online A/B Testing](#online-ab-testing)
- [Statistical Significance](#statistical-significance)
- [Best Practices](#best-practices)

## When to A/B Test

Use A/B testing when:

1. **Comparing prompts** - Which prompt formulation works better?
2. **Model selection** - GPT-4 vs Claude vs Gemini
3. **Architecture changes** - New node structure vs old
4. **Parameter tuning** - Temperature, max tokens, etc.
5. **Tool selection** - Different tool implementations
6. **Before deployment** - Validate improvements before shipping

## Offline A/B Testing

Test both versions on the same dataset before deployment.

### Python Example

```python
from langsmith import Client

client = Client()

# Run both versions on same dataset
results_a = client.evaluate(
    agent_v1.invoke,
    data="test_dataset",
    evaluators=[accuracy_evaluator, latency_evaluator],
    experiment_prefix="agent_v1"
)

results_b = client.evaluate(
    agent_v2.invoke,
    data="test_dataset",
    evaluators=[accuracy_evaluator, latency_evaluator],
    experiment_prefix="agent_v2"
)

# Compare results
exp_a = client.read_project(project_name=results_a.experiment_name, include_stats=True)
exp_b = client.read_project(project_name=results_b.experiment_name, include_stats=True)

print("A/B Comparison:")
print(f"Version A accuracy: {exp_a.feedback_stats['accuracy']['avg']:.2%}")
print(f"Version B accuracy: {exp_b.feedback_stats['accuracy']['avg']:.2%}")

print(f"Version A latency: {exp_a.latency_p50:.2f}s")
print(f"Version B latency: {exp_b.latency_p50:.2f}s")
```

### Using compare_agents.py Script

```bash
# Compare two agent implementations
uv run scripts/compare_agents.py \
  agent_v1:run_agent \
  agent_v2:run_agent \
  test_dataset.json \
  --output comparison_report.json
```

### TypeScript Example

```typescript
import { Client } from "langsmith";
import { evaluate } from "langsmith/evaluation";

const client = new Client();

// Run both versions
const resultsA = await evaluate(agentV1.invoke.bind(agentV1), {
  data: "test_dataset",
  evaluators: [accuracyEvaluator, latencyEvaluator],
  experimentPrefix: "agent_v1",
});

const resultsB = await evaluate(agentV2.invoke.bind(agentV2), {
  data: "test_dataset",
  evaluators: [accuracyEvaluator, latencyEvaluator],
  experimentPrefix: "agent_v2",
});

// Compare
const expA = await client.readProject({
  projectName: resultsA.experimentName,
  includeStats: true,
});

const expB = await client.readProject({
  projectName: resultsB.experimentName,
  includeStats: true,
});

console.log("A/B Comparison:");
console.log(`Version A accuracy: ${expA.feedbackStats.accuracy.avg}`);
console.log(`Version B accuracy: ${expB.feedbackStats.accuracy.avg}`);
```

## Online A/B Testing

Test both versions with real traffic.

### Traffic Splitting

```python
from enum import Enum

class AgentVersion(Enum):
    V1 = "v1"
    V2 = "v2"

def select_agent_version(user_id: str, split_ratio: float = 0.5) -> AgentVersion:
    """
    Deterministic traffic split based on user_id.

    Args:
        user_id: Unique user identifier
        split_ratio: Percentage to route to V2 (0.0 to 1.0)
    """
    # Hash user_id for consistent assignment
    import hashlib
    hash_value = int(hashlib.md5(user_id.encode()).hexdigest(), 16)
    bucket = (hash_value % 100) / 100.0

    return AgentVersion.V2 if bucket < split_ratio else AgentVersion.V1

# Usage
def handle_request(user_id: str, inputs: dict):
    version = select_agent_version(user_id, split_ratio=0.2)  # 20% to V2

    if version == AgentVersion.V2:
        result = agent_v2.invoke(inputs)
    else:
        result = agent_v1.invoke(inputs)

    # Log version for analysis
    return result, version
```

### TypeScript Traffic Splitting

```typescript
import { createHash } from "crypto";

enum AgentVersion {
  V1 = "v1",
  V2 = "v2",
}

function selectAgentVersion(
  userId: string,
  splitRatio: number = 0.5
): AgentVersion {
  const hash = createHash("md5").update(userId).digest("hex");
  const hashValue = parseInt(hash.substring(0, 8), 16);
  const bucket = (hashValue % 100) / 100;

  return bucket < splitRatio ? AgentVersion.V2 : AgentVersion.V1;
}

// Usage
async function handleRequest(userId: string, inputs: any) {
  const version = selectAgentVersion(userId, 0.2); // 20% to V2

  const result =
    version === AgentVersion.V2
      ? await agentV2.invoke(inputs)
      : await agentV1.invoke(inputs);

  return { result, version };
}
```

### Tracking Online Experiments

```python
from langsmith import Client

client = Client()

def run_online_experiment(user_id: str, inputs: dict):
    """Run request and track experiment version."""
    version = select_agent_version(user_id, split_ratio=0.2)

    # Tag run with experiment version
    with client.trace(
        name="online_experiment",
        project_name="production",
        tags=["ab_test", f"version_{version.value}"],
        metadata={"experiment_version": version.value}
    ) as run:
        if version == AgentVersion.V2:
            result = agent_v2.invoke(inputs)
        else:
            result = agent_v1.invoke(inputs)

    return result

# Later, analyze results by version
runs_v1 = client.list_runs(
    project_name="production",
    filter='has(tags, "version_v1") and gt(start_time, "2025-01-01T00:00:00Z")',
)

runs_v2 = client.list_runs(
    project_name="production",
    filter='has(tags, "version_v2") and gt(start_time, "2025-01-01T00:00:00Z")',
)

# Compare metrics
print(f"V1: {len(list(runs_v1))} runs")
print(f"V2: {len(list(runs_v2))} runs")
```

## Statistical Significance

### Sample Size Calculation

```python
def calculate_sample_size(
    baseline_rate: float,
    minimum_detectable_effect: float,
    significance_level: float = 0.05,
    power: float = 0.8
) -> int:
    """
    Calculate required sample size for A/B test.

    Args:
        baseline_rate: Current success rate (e.g., 0.8 for 80%)
        minimum_detectable_effect: Smallest difference to detect (e.g., 0.05 for 5%)
        significance_level: Alpha (typically 0.05)
        power: 1 - Beta (typically 0.8)

    Returns:
        Required sample size per variant
    """
    from scipy.stats import norm

    # Z-scores for significance and power
    z_alpha = norm.ppf(1 - significance_level / 2)
    z_beta = norm.ppf(power)

    # Effect size
    effect = minimum_detectable_effect
    p1 = baseline_rate
    p2 = baseline_rate + effect

    # Pooled standard deviation
    p_pooled = (p1 + p2) / 2
    std_pooled = (2 * p_pooled * (1 - p_pooled)) ** 0.5

    # Sample size formula
    n = ((z_alpha + z_beta) ** 2 * std_pooled ** 2) / (effect ** 2)

    return int(n) + 1

# Example
sample_size = calculate_sample_size(
    baseline_rate=0.80,  # 80% current accuracy
    minimum_detectable_effect=0.05  # Want to detect 5% improvement
)

print(f"Required sample size: {sample_size} per variant")
print(f"Total requests needed: {sample_size * 2}")
```

### Significance Testing

```python
from scipy import stats

def test_significance(results_a: list, results_b: list) -> dict:
    """
    Test if difference between A and B is statistically significant.

    Args:
        results_a: List of scores for variant A
        results_b: List of scores for variant B

    Returns:
        Dictionary with test results
    """
    # Calculate means
    mean_a = sum(results_a) / len(results_a)
    mean_b = sum(results_b) / len(results_b)

    # T-test for difference in means
    t_stat, p_value = stats.ttest_ind(results_a, results_b)

    # Effect size (Cohen's d)
    pooled_std = (
        ((len(results_a) - 1) * stats.tstd(results_a) ** 2 +
         (len(results_b) - 1) * stats.tstd(results_b) ** 2) /
        (len(results_a) + len(results_b) - 2)
    ) ** 0.5

    cohens_d = (mean_b - mean_a) / pooled_std

    return {
        "mean_a": mean_a,
        "mean_b": mean_b,
        "difference": mean_b - mean_a,
        "p_value": p_value,
        "significant": p_value < 0.05,
        "cohens_d": cohens_d,
        "sample_size_a": len(results_a),
        "sample_size_b": len(results_b)
    }

# Usage
accuracy_scores_v1 = [0.8, 0.85, 0.82, 0.79, 0.88, ...]  # Collect from runs
accuracy_scores_v2 = [0.87, 0.89, 0.85, 0.91, 0.86, ...]

results = test_significance(accuracy_scores_v1, accuracy_scores_v2)

print(f"Version A mean: {results['mean_a']:.2%}")
print(f"Version B mean: {results['mean_b']:.2%}")
print(f"Difference: {results['difference']:+.2%}")
print(f"P-value: {results['p_value']:.4f}")
print(f"Significant: {results['significant']}")
print(f"Effect size (Cohen's d): {results['cohens_d']:.2f}")
```

### Confidence Intervals

```python
def calculate_confidence_interval(scores: list, confidence: float = 0.95) -> tuple:
    """Calculate confidence interval for mean score."""
    from scipy import stats

    mean = sum(scores) / len(scores)
    std_err = stats.sem(scores)
    ci = stats.t.interval(
        confidence,
        len(scores) - 1,
        loc=mean,
        scale=std_err
    )

    return ci

# Usage
ci_v1 = calculate_confidence_interval(accuracy_scores_v1)
ci_v2 = calculate_confidence_interval(accuracy_scores_v2)
mean_v1 = sum(accuracy_scores_v1) / len(accuracy_scores_v1)
mean_v2 = sum(accuracy_scores_v2) / len(accuracy_scores_v2)

print(f"V1: {mean_v1:.2%} (95% CI: {ci_v1[0]:.2%} to {ci_v1[1]:.2%})")
print(f"V2: {mean_v2:.2%} (95% CI: {ci_v2[0]:.2%} to {ci_v2[1]:.2%})")
```

## Best Practices

### 1. Define Success Criteria Upfront

```python
# Document before starting test
AB_TEST_CONFIG = {
    "test_name": "gpt4_vs_claude_sonnet",
    "start_date": "2025-02-06",
    "duration_days": 14,
    "primary_metric": "accuracy",
    "secondary_metrics": ["latency", "cost"],
    "minimum_detectable_effect": 0.05,
    "significance_level": 0.05,
    "required_sample_size": 1000,
    "stopping_criteria": {
        "min_samples": 1000,
        "min_duration_days": 7,
        "significance_threshold": 0.05
    }
}
```

### 2. Monitor Both Variants

```python
def monitor_ab_test(experiment_name: str):
    """Generate monitoring report for AB test."""
    from langsmith import Client

    client = Client()

    runs_v1 = list(client.list_runs(
        project_name=experiment_name,
        filter='has(tags, "version_v1")'
    ))

    runs_v2 = list(client.list_runs(
        project_name=experiment_name,
        filter='has(tags, "version_v2")'
    ))

    print("AB Test Monitoring Report")
    print("=" * 50)
    print(f"Version 1: {len(runs_v1)} runs")
    print(f"Version 2: {len(runs_v2)} runs")

    # Calculate metrics
    v1_errors = sum(1 for r in runs_v1 if r.error is not None)
    v2_errors = sum(1 for r in runs_v2 if r.error is not None)

    print(f"\nError Rates:")
    print(f"V1: {v1_errors / len(runs_v1):.2%}")
    print(f"V2: {v2_errors / len(runs_v2):.2%}")

    # Alert if severe degradation
    if v2_errors / len(runs_v2) > 2 * (v1_errors / len(runs_v1)):
        print("\n⚠️ WARNING: V2 error rate significantly higher!")
```

### 3. Consider Multiple Metrics

Don't optimize for a single metric - track trade-offs.

```python
def comprehensive_comparison(results_a: dict, results_b: dict) -> dict:
    """Compare on multiple dimensions."""
    return {
        "accuracy": {
            "v1": results_a["accuracy"],
            "v2": results_b["accuracy"],
            "winner": "v2" if results_b["accuracy"] > results_a["accuracy"] else "v1"
        },
        "latency": {
            "v1": results_a["latency"],
            "v2": results_b["latency"],
            "winner": "v2" if results_b["latency"] < results_a["latency"] else "v1"
        },
        "cost": {
            "v1": results_a["cost"],
            "v2": results_b["cost"],
            "winner": "v2" if results_b["cost"] < results_a["cost"] else "v1"
        },
        "overall_recommendation": "..."  # Based on business priorities
    }
```

### 4. Run for Sufficient Duration

Account for:
- Day-of-week effects
- Peak vs off-peak traffic
- Seasonal patterns

Minimum 1 week, preferably 2 weeks for online tests.

### 5. Avoid Peeking

Don't stop early based on intermediate results - increases false positive rate.

Use sequential testing if you need early stopping:

```python
def sequential_test(results_a: list, results_b: list, alpha: float = 0.05):
    """Sequential probability ratio test for early stopping."""
    # Implement SPRT or similar
    # See: https://en.wikipedia.org/wiki/Sequential_probability_ratio_test
    pass
```

### 6. Document Everything

```python
# ab_test_results.json
{
    "test_name": "prompt_v2_test",
    "date_range": "2025-02-01 to 2025-02-14",
    "variants": {
        "control": {
            "description": "Original prompt",
            "sample_size": 1250,
            "accuracy": 0.82,
            "latency_p50": 1.3,
            "cost_per_request": 0.015
        },
        "treatment": {
            "description": "Optimized prompt with examples",
            "sample_size": 1248,
            "accuracy": 0.87,
            "latency_p50": 1.5,
            "cost_per_request": 0.018
        }
    },
    "statistical_analysis": {
        "p_value": 0.003,
        "significant": true,
        "confidence_interval": [0.02, 0.08]
    },
    "decision": "Deploy treatment to 100%",
    "rationale": "5% accuracy improvement outweighs 20% cost increase"
}
```

## Example: Complete A/B Test

```python
# 1. Setup
from langsmith import Client
from datetime import datetime, timedelta

client = Client()

# 2. Run experiment
def run_ab_experiment():
    dataset = client.read_dataset(dataset_name="production_sample")

    results_v1 = client.evaluate(
        agent_v1.invoke,
        data="production_sample",
        evaluators=[accuracy_evaluator, latency_evaluator, cost_evaluator],
        experiment_prefix="ab_test_v1"
    )

    results_v2 = client.evaluate(
        agent_v2.invoke,
        data="production_sample",
        evaluators=[accuracy_evaluator, latency_evaluator, cost_evaluator],
        experiment_prefix="ab_test_v2"
    )

    return results_v1, results_v2

# 3. Analyze
results_v1, results_v2 = run_ab_experiment()

exp_v1 = client.read_project(project_name=results_v1.experiment_name, include_stats=True)
exp_v2 = client.read_project(project_name=results_v2.experiment_name, include_stats=True)

# 4. Test significance
accuracy_v1 = [...]  # Extract from runs
accuracy_v2 = [...]

significance = test_significance(accuracy_v1, accuracy_v2)

# 5. Make decision
if significance["significant"] and significance["difference"] > 0.03:
    print("✅ Deploy V2 to production")
else:
    print("❌ Keep V1, insufficient improvement")
```
