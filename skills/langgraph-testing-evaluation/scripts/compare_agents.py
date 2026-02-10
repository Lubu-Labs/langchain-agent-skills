#!/usr/bin/env python3
"""
Compare two agent implementations using A/B evaluation.

This script runs two agent versions on the same dataset and compares
their performance across multiple metrics. Useful for validating changes
before deployment.

Usage:
    python compare_agents.py <agent_a> <agent_b> <dataset> [options]

Examples:
    # Compare two agent versions
    python compare_agents.py my_agent:agent_v1 my_agent:agent_v2 test_dataset

    # Output comparison report
    python compare_agents.py agent_a:run agent_b:run dataset --output comparison_report.json
"""

import argparse
import asyncio
import importlib
import inspect
import json
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple


def load_agent_function(function_path: str) -> Tuple[Callable, str]:
    """Load agent function from module path."""
    try:
        if ":" not in function_path:
            print(
                "Error: Function path must be in format 'module.path:function_name'"
            )
            sys.exit(1)

        module_name, func_name = function_path.split(":", 1)

        # Import module
        module = importlib.import_module(module_name)

        # Get function
        if not hasattr(module, func_name):
            print(f"Error: Module '{module_name}' has no function '{func_name}'")
            sys.exit(1)

        func = getattr(module, func_name)
        return func, f"{module_name}:{func_name}"

    except Exception as e:
        print(f"Error loading agent function: {e}")
        sys.exit(1)


def load_dataset(dataset_path: str) -> List[Dict[str, Any]]:
    """Load dataset from file or LangSmith."""
    # Check if it's a file path
    path = Path(dataset_path)
    if path.exists():
        with open(path, "r") as f:
            data = json.load(f)
            if isinstance(data, list):
                return data
            elif isinstance(data, dict) and "examples" in data:
                return data["examples"]
            else:
                print(
                    "Error: Dataset file must contain a list of examples or a dict with 'examples' key"
                )
                sys.exit(1)

    # Try to load from LangSmith
    try:
        from langsmith import Client

        client = Client()
        dataset = client.read_dataset(dataset_name=dataset_path)
        examples = list(client.list_examples(dataset_id=dataset.id))
        return [{"inputs": ex.inputs, "outputs": ex.outputs} for ex in examples]
    except ImportError:
        print("Error: langsmith package not installed for loading remote datasets")
        print("Install with: uv pip install langsmith")
        sys.exit(1)
    except Exception as e:
        print(f"Error: Could not find dataset '{dataset_path}'")
        print(f"Details: {e}")
        sys.exit(1)


def evaluate_agent(
    agent_func: Callable,
    dataset: List[Dict],
    agent_name: str,
) -> Dict[str, Any]:
    """Evaluate a single agent on the dataset."""
    results = {
        "agent": agent_name,
        "examples": [],
        "metrics": {
            "total_examples": len(dataset),
            "successful": 0,
            "failed": 0,
            "total_latency": 0.0,
            "latencies": [],
        },
    }

    print(f"\nEvaluating {agent_name}...")

    for i, example in enumerate(dataset):
        print(f"  Example {i + 1}/{len(dataset)}...", end=" ")
        input_payload = example.get("inputs", example.get("input"))
        if input_payload is None:
            print("✗ Error: Missing 'inputs' in dataset example")
            results["metrics"]["failed"] += 1
            results["examples"].append(
                {
                    "index": i,
                    "error": "Missing 'inputs' in dataset example",
                    "success": False,
                }
            )
            continue

        example_result = {
            "index": i,
            "input": input_payload,
            "expected_output": example.get("outputs"),
        }

        try:
            # Measure latency
            start_time = time.time()
            output = invoke_agent(agent_func, input_payload)
            latency = time.time() - start_time

            example_result["output"] = output
            example_result["latency"] = latency
            example_result["success"] = True

            results["metrics"]["successful"] += 1
            results["metrics"]["total_latency"] += latency
            results["metrics"]["latencies"].append(latency)

            print(f"✓ ({latency:.2f}s)")

        except Exception as e:
            example_result["error"] = str(e)
            example_result["success"] = False
            results["metrics"]["failed"] += 1

            print(f"✗ Error: {e}")

        results["examples"].append(example_result)

    # Calculate summary metrics
    if results["metrics"]["latencies"]:
        latencies = results["metrics"]["latencies"]
        results["metrics"]["avg_latency"] = sum(latencies) / len(latencies)
        results["metrics"]["min_latency"] = min(latencies)
        results["metrics"]["max_latency"] = max(latencies)
        results["metrics"]["p50_latency"] = sorted(latencies)[len(latencies) // 2]
        results["metrics"]["p95_latency"] = sorted(latencies)[
            int(len(latencies) * 0.95)
        ]

    results["metrics"]["success_rate"] = (
        results["metrics"]["successful"] / results["metrics"]["total_examples"]
        if results["metrics"]["total_examples"] > 0
        else 0.0
    )

    return results


def invoke_agent(agent_func: Callable, inputs: Dict[str, Any]) -> Any:
    """Invoke an agent function that may be sync or async."""
    result = agent_func(inputs)
    if inspect.isawaitable(result):
        return asyncio.run(result)
    return result


def compare_results(
    results_a: Dict[str, Any], results_b: Dict[str, Any]
) -> Dict[str, Any]:
    """Compare results from two agents."""
    comparison = {
        "agent_a": results_a["agent"],
        "agent_b": results_b["agent"],
        "metrics_comparison": {},
        "winner": None,
    }

    metrics_a = results_a["metrics"]
    metrics_b = results_b["metrics"]

    # Compare success rate
    comparison["metrics_comparison"]["success_rate"] = {
        "agent_a": metrics_a["success_rate"],
        "agent_b": metrics_b["success_rate"],
        "difference": metrics_b["success_rate"] - metrics_a["success_rate"],
        "winner": (
            results_b["agent"]
            if metrics_b["success_rate"] > metrics_a["success_rate"]
            else results_a["agent"]
            if metrics_a["success_rate"] > metrics_b["success_rate"]
            else "tie"
        ),
    }

    # Compare latency (lower is better)
    if "avg_latency" in metrics_a and "avg_latency" in metrics_b:
        comparison["metrics_comparison"]["avg_latency"] = {
            "agent_a": metrics_a["avg_latency"],
            "agent_b": metrics_b["avg_latency"],
            "difference": metrics_b["avg_latency"] - metrics_a["avg_latency"],
            "winner": (
                results_a["agent"]
                if metrics_a["avg_latency"] < metrics_b["avg_latency"]
                else results_b["agent"]
                if metrics_b["avg_latency"] < metrics_a["avg_latency"]
                else "tie"
            ),
        }

        comparison["metrics_comparison"]["p95_latency"] = {
            "agent_a": metrics_a["p95_latency"],
            "agent_b": metrics_b["p95_latency"],
            "difference": metrics_b["p95_latency"] - metrics_a["p95_latency"],
            "winner": (
                results_a["agent"]
                if metrics_a["p95_latency"] < metrics_b["p95_latency"]
                else results_b["agent"]
                if metrics_b["p95_latency"] < metrics_a["p95_latency"]
                else "tie"
            ),
        }

    # Determine overall winner (success rate weighted more heavily)
    success_winner = comparison["metrics_comparison"]["success_rate"]["winner"]
    latency_winner = comparison["metrics_comparison"].get("avg_latency", {}).get(
        "winner", "tie"
    )

    if success_winner == latency_winner:
        comparison["winner"] = success_winner
    elif success_winner == "tie":
        comparison["winner"] = latency_winner
    elif latency_winner == "tie":
        comparison["winner"] = success_winner
    else:
        # Success rate wins in tie-breaker
        comparison["winner"] = success_winner

    return comparison


def print_comparison_report(comparison: Dict[str, Any]):
    """Print a formatted comparison report."""
    print("\n" + "=" * 70)
    print("AGENT COMPARISON REPORT")
    print("=" * 70)

    print(f"\nAgent A: {comparison['agent_a']}")
    print(f"Agent B: {comparison['agent_b']}")

    print("\n" + "-" * 70)
    print("METRICS COMPARISON")
    print("-" * 70)

    for metric, data in comparison["metrics_comparison"].items():
        print(f"\n{metric.upper().replace('_', ' ')}:")
        print(f"  Agent A: {data['agent_a']:.4f}")
        print(f"  Agent B: {data['agent_b']:.4f}")
        print(f"  Difference: {data['difference']:+.4f}")
        print(f"  Winner: {data['winner']}")

    print("\n" + "=" * 70)
    print(f"OVERALL WINNER: {comparison['winner']}")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Compare two agent implementations")
    parser.add_argument(
        "agent_a", help="First agent function path (e.g., my_agent:agent_v1)"
    )
    parser.add_argument(
        "agent_b", help="Second agent function path (e.g., my_agent:agent_v2)"
    )
    parser.add_argument(
        "dataset", help="Dataset name (LangSmith) or file path (JSON)"
    )
    parser.add_argument(
        "--output",
        "-o",
        help="Output file for detailed comparison results (JSON)",
    )

    args = parser.parse_args()

    # Load agents
    print("Loading agents...")
    agent_a_func, agent_a_name = load_agent_function(args.agent_a)
    agent_b_func, agent_b_name = load_agent_function(args.agent_b)

    # Load dataset
    print(f"Loading dataset: {args.dataset}")
    dataset = load_dataset(args.dataset)
    print(f"Loaded {len(dataset)} examples")

    # Evaluate agents
    results_a = evaluate_agent(agent_a_func, dataset, agent_a_name)
    results_b = evaluate_agent(agent_b_func, dataset, agent_b_name)

    # Compare results
    comparison = compare_results(results_a, results_b)

    # Print report
    print_comparison_report(comparison)

    # Save detailed results
    if args.output:
        output_data = {
            "comparison": comparison,
            "agent_a_results": results_a,
            "agent_b_results": results_b,
        }

        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(output_data, f, indent=2, default=str)

        print(f"\n✅ Detailed results saved to: {output_path}")


if __name__ == "__main__":
    main()
