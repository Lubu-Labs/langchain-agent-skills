#!/usr/bin/env python3
"""
Run evaluations with LangSmith's evaluate() API.

This script follows current LangSmith evaluation patterns:
- Target functions accept `inputs: dict` and return outputs (preferably dict)
- Evaluators use named arguments such as `inputs`, `outputs`, `reference_outputs`, and `run`
- Evaluation runs through `langsmith.evaluate(...)`

Usage:
    uv run evaluate_with_langsmith.py <target_function> <dataset> [options]

Examples:
    uv run evaluate_with_langsmith.py my_agent:run_agent my_dataset
    uv run evaluate_with_langsmith.py my_agent:run_agent my_dataset --evaluators accuracy,relevance
    uv run evaluate_with_langsmith.py my_agent:run_agent --create-dataset examples.json --dataset-name my_dataset
"""

import argparse
import asyncio
import importlib
import inspect
import json
import math
import sys
from typing import Any, Callable, Dict, List, Optional


def die(message: str) -> None:
    """Print an error and exit."""
    print(f"Error: {message}")
    sys.exit(1)


def load_target_function(function_path: str) -> Callable[[Dict[str, Any]], Any]:
    """Load target function from `module.path:function_name`."""
    if ":" not in function_path:
        die("Function path must be in format 'module.path:function_name'")

    module_name, func_name = function_path.split(":", 1)

    try:
        module = importlib.import_module(module_name)
    except Exception as exc:
        die(f"Failed to import module '{module_name}': {exc}")

    if not hasattr(module, func_name):
        die(f"Module '{module_name}' has no function '{func_name}'")

    return getattr(module, func_name)


def _load_examples_file(examples_file: str) -> List[Dict[str, Any]]:
    """Load examples from JSON file (list or {'examples': [...]}) format."""
    try:
        with open(examples_file, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except FileNotFoundError:
        die(f"Examples file not found: {examples_file}")
    except json.JSONDecodeError as exc:
        die(f"Invalid JSON in examples file: {exc}")

    if isinstance(payload, list):
        return payload

    if isinstance(payload, dict) and isinstance(payload.get("examples"), list):
        return payload["examples"]

    die("Examples file must contain a list or an object with an 'examples' list")


def create_dataset_from_file(dataset_name: str, examples_file: str) -> str:
    """Create a LangSmith dataset from a local JSON examples file."""
    try:
        from langsmith import Client
    except ImportError:
        die("langsmith package not installed. Install with: uv pip install -U langsmith")

    client = Client()
    examples = _load_examples_file(examples_file)

    formatted_examples = []
    for item in examples:
        if not isinstance(item, dict):
            continue

        inputs = item.get("inputs", item.get("input", {}))
        outputs = item.get("outputs", item.get("output", {}))

        if not isinstance(inputs, dict):
            # Keep dataset schema aligned with LangSmith's key-value input format.
            inputs = {"input": inputs}

        if outputs is None:
            outputs = {}
        elif not isinstance(outputs, dict):
            outputs = {"output": outputs}

        formatted_examples.append({"inputs": inputs, "outputs": outputs})

    if not formatted_examples:
        die("No valid examples found in input file")

    print(f"Creating dataset '{dataset_name}' with {len(formatted_examples)} examples...")

    try:
        dataset = client.create_dataset(dataset_name=dataset_name)
        client.create_examples(dataset_id=dataset.id, examples=formatted_examples)
    except Exception as exc:
        die(f"Failed to create dataset '{dataset_name}': {exc}")

    print(f"✅ Dataset created: {dataset_name}")
    return dataset_name


def _normalize_target_output(output: Any) -> Dict[str, Any]:
    """Ensure target output is a dict, matching LangSmith evaluator conventions."""
    if isinstance(output, dict):
        return output
    return {"output": output}


def invoke_target(target_func: Callable[[Dict[str, Any]], Any], inputs: Dict[str, Any]) -> Dict[str, Any]:
    """Invoke sync or async target function and normalize output to dict."""
    result = target_func(inputs)
    if inspect.isawaitable(result):
        result = asyncio.run(result)
    return _normalize_target_output(result)


def _extract_primary_value(value: Any) -> Any:
    """Extract a stable comparison payload from common output dict shapes."""
    if isinstance(value, dict):
        for key in ("output", "answer", "response", "messages"):
            if key in value:
                return value[key]
    return value


def accuracy_evaluator(
    inputs: Dict[str, Any],
    outputs: Dict[str, Any],
    reference_outputs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Exact match between model output and reference output."""
    actual = _extract_primary_value(outputs)
    expected = _extract_primary_value(reference_outputs or {})

    score = 1.0 if actual == expected else 0.0
    return {"key": "accuracy", "score": score}


def relevance_evaluator(
    inputs: Dict[str, Any],
    outputs: Dict[str, Any],
    reference_outputs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Simple lexical overlap heuristic relative to reference outputs."""
    _ = inputs  # Not used by this heuristic.

    actual = str(_extract_primary_value(outputs) or "").lower()
    expected = str(_extract_primary_value(reference_outputs or {}) or "").lower()

    expected_tokens = {tok for tok in expected.split() if tok}
    if not expected_tokens:
        return {"key": "relevance", "score": 0.5, "comment": "No reference text"}

    actual_tokens = {tok for tok in actual.split() if tok}
    overlap = len(expected_tokens.intersection(actual_tokens))
    score = overlap / len(expected_tokens)
    return {"key": "relevance", "score": score}


def latency_evaluator(run: Any = None) -> Dict[str, Any]:
    """Score latency using run timing metadata when present."""
    start_time = getattr(run, "start_time", None) or getattr(run, "startTime", None)
    end_time = getattr(run, "end_time", None) or getattr(run, "endTime", None)

    if start_time is None or end_time is None:
        return {"key": "latency", "score": 0.5, "comment": "No timing data"}

    try:
        if hasattr(start_time, "timestamp") and hasattr(end_time, "timestamp"):
            latency = end_time.timestamp() - start_time.timestamp()
        else:
            latency = float(end_time) - float(start_time)
    except Exception:
        return {"key": "latency", "score": 0.0, "comment": "Invalid timing data"}

    if math.isnan(latency) or math.isinf(latency):
        return {"key": "latency", "score": 0.0, "comment": "Invalid timing data"}

    if latency < 1.0:
        score = 1.0
    elif latency < 5.0:
        score = 0.5
    else:
        score = 0.0

    return {"key": "latency", "score": score, "comment": f"{latency:.2f}s"}


def create_evaluators(names: List[str]) -> List[Callable[..., Dict[str, Any]]]:
    """Map evaluator names to evaluator callables."""
    registry: Dict[str, Callable[..., Dict[str, Any]]] = {
        "accuracy": accuracy_evaluator,
        "relevance": relevance_evaluator,
        "latency": latency_evaluator,
    }

    evaluators: List[Callable[..., Dict[str, Any]]] = []
    for name in names:
        evaluator = registry.get(name)
        if evaluator is None:
            print(f"Warning: Unknown evaluator '{name}'")
            continue
        evaluators.append(evaluator)

    return evaluators


def run_evaluation(
    target_func: Callable[[Dict[str, Any]], Dict[str, Any]],
    dataset: str,
    evaluators: List[Callable[..., Dict[str, Any]]],
    experiment_prefix: str,
    metadata: Optional[Dict[str, Any]],
    max_concurrency: int,
    upload_results: bool,
) -> Any:
    """Run evaluation via `langsmith.evaluate(...)`."""
    try:
        from langsmith import evaluate
    except ImportError:
        die("langsmith package not installed. Install with: uv pip install -U langsmith")

    print(f"\nRunning evaluation on dataset '{dataset}'...")
    print(f"Evaluators: {len(evaluators)}")

    try:
        return evaluate(
            target_func,
            data=dataset,
            evaluators=evaluators,
            experiment_prefix=experiment_prefix,
            metadata=metadata or {},
            max_concurrency=max_concurrency,
            upload_results=upload_results,
        )
    except Exception as exc:
        die(f"Failed to run evaluation: {exc}")


def print_evaluation_results(results: Any) -> None:
    """Print a compact summary of evaluation outputs."""
    print("\n" + "=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)

    experiment_name = getattr(results, "experiment_name", None)
    if experiment_name:
        print(f"\nExperiment: {experiment_name}")

    print("\nView detailed results in LangSmith:")
    print("https://smith.langchain.com")
    print("\n✅ Evaluation complete")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run evaluations using LangSmith")
    parser.add_argument(
        "target_function",
        help="Target function path (e.g., my_agent:run_agent)",
    )
    parser.add_argument(
        "dataset",
        nargs="?",
        help="Dataset name in LangSmith",
    )
    parser.add_argument(
        "--create-dataset",
        help="Create dataset from examples file (JSON)",
    )
    parser.add_argument(
        "--dataset-name",
        help="Name for newly created dataset (required with --create-dataset)",
    )
    parser.add_argument(
        "--evaluators",
        default="accuracy",
        help="Comma-separated list of evaluators (accuracy,relevance,latency)",
    )
    parser.add_argument(
        "--experiment-prefix",
        default="evaluation",
        help="Prefix for experiment name",
    )
    parser.add_argument(
        "--metadata",
        help="JSON string with experiment metadata",
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=4,
        help="Max concurrent evaluation workers",
    )
    parser.add_argument(
        "--no-upload",
        action="store_true",
        help="Run locally without uploading results to LangSmith",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.max_concurrency < 1:
        die("--max-concurrency must be >= 1")

    dataset_name = args.dataset
    if args.create_dataset:
        if not args.dataset_name:
            die("--dataset-name is required with --create-dataset")
        dataset_name = create_dataset_from_file(args.dataset_name, args.create_dataset)

    if not dataset_name:
        die("dataset name or --create-dataset is required")

    print(f"Loading target function: {args.target_function}")
    user_target = load_target_function(args.target_function)

    def eval_target(inputs: Dict[str, Any]) -> Dict[str, Any]:
        return invoke_target(user_target, inputs)

    evaluator_names = [item.strip() for item in args.evaluators.split(",") if item.strip()]
    evaluators = create_evaluators(evaluator_names)
    if not evaluators:
        die("No valid evaluators specified")

    print(f"Using evaluators: {', '.join(evaluator_names)}")

    metadata = None
    if args.metadata:
        try:
            metadata = json.loads(args.metadata)
        except json.JSONDecodeError as exc:
            die(f"Invalid JSON in --metadata: {exc}")

    results = run_evaluation(
        target_func=eval_target,
        dataset=dataset_name,
        evaluators=evaluators,
        experiment_prefix=args.experiment_prefix,
        metadata=metadata,
        max_concurrency=args.max_concurrency,
        upload_results=not args.no_upload,
    )

    print_evaluation_results(results)


if __name__ == "__main__":
    main()
