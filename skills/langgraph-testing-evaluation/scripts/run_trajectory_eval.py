#!/usr/bin/env python3
"""
Run trajectory evaluation on a LangGraph agent.

This script evaluates agent execution trajectories using either trajectory
matching or LLM-as-judge evaluation. It uses LangSmith's evaluation runner
and supports local-only runs with --no-langsmith (upload_results=False).

Usage:
    uv run run_trajectory_eval.py <target_function> <dataset> [options]

Examples:
    # Evaluate with trajectory matching
    uv run run_trajectory_eval.py my_agent:run_agent my_dataset --method match --reference-trajectory reference.json

    # Evaluate with LLM-as-judge
    uv run run_trajectory_eval.py my_agent:run_agent my_dataset --method llm-judge --model openai:o3-mini

    # Evaluate from local dataset file
    uv run run_trajectory_eval.py my_agent:run_agent dataset.json --method llm-judge --no-langsmith
"""

import argparse
import importlib
import inspect
import json
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple


def load_target_function(function_path: str) -> Callable:
    """Load target function from module path."""
    try:
        if ":" not in function_path:
            print("Error: Function path must be in format 'module.path:function_name'")
            sys.exit(1)

        module_name, func_name = function_path.split(":", 1)
        module = importlib.import_module(module_name)

        if not hasattr(module, func_name):
            print(f"Error: Module '{module_name}' has no function '{func_name}'")
            sys.exit(1)

        return getattr(module, func_name)

    except Exception as e:
        print(f"Error loading target function: {e}")
        sys.exit(1)


def load_dataset_data(dataset_path: str) -> Any:
    """
    Load dataset from local JSON file or pass through LangSmith dataset name.

    - If dataset_path is a local file, returns list[dict] examples.
    - Otherwise, returns the dataset name string for LangSmith to resolve.
    """
    dataset_file = Path(dataset_path)
    if dataset_file.exists():
        with open(dataset_file, "r") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
        if isinstance(data, dict) and "examples" in data:
            return data["examples"]

        print("Error: Dataset file must contain a list of examples or a dict with 'examples' key")
        sys.exit(1)

    return dataset_path


def invoke_target(target_func: Callable, inputs: Dict[str, Any]) -> Any:
    """Invoke sync or async target function."""
    result = target_func(inputs)
    if inspect.isawaitable(result):
        import asyncio

        result = asyncio.run(result)
    return normalize_target_output(result)


def normalize_target_output(output: Any) -> Any:
    """
    Normalize target output for trajectory evaluators.

    Trajectory evaluators expect a list of messages. If the target returns a
    dict with a messages key, unwrap it.
    """
    if isinstance(output, dict) and "messages" in output:
        return output["messages"]
    return output


def extract_outputs_and_reference(
    outputs: Any = None,
    reference_outputs: Any = None,
    run: Any = None,
    example: Any = None,
) -> Tuple[Any, Any]:
    """
    Normalize evaluator inputs from named LangSmith evaluator arguments.

    Supports either direct `outputs`/`reference_outputs` or `run`/`example`.
    """
    if outputs is None and run is not None:
        outputs = getattr(run, "outputs", None)
    if reference_outputs is None and example is not None:
        reference_outputs = getattr(example, "outputs", None)

    return normalize_target_output(outputs), normalize_target_output(reference_outputs)


def create_trajectory_matcher(
    reference_trajectory: Optional[List[Dict[str, Any]]],
    match_mode: str,
) -> Callable:
    """Create a trajectory matching evaluator using AgentEvals."""
    try:
        from agentevals.trajectory.match import create_trajectory_match_evaluator

        base_evaluator = create_trajectory_match_evaluator(
            trajectory_match_mode=match_mode
        )

        def evaluate_trajectory(
            inputs: Optional[Dict[str, Any]] = None,
            outputs: Any = None,
            reference_outputs: Any = None,
            run: Any = None,
            example: Any = None,
        ) -> Dict[str, Any]:
            _ = inputs
            outputs, reference_outputs = extract_outputs_and_reference(
                outputs=outputs,
                reference_outputs=reference_outputs,
                run=run,
                example=example,
            )
            reference = reference_outputs if reference_outputs else reference_trajectory

            if reference is None:
                return {
                    "key": f"trajectory_{match_mode}_match",
                    "score": 0.0,
                    "comment": "No reference trajectory provided",
                }

            return base_evaluator(outputs=outputs, reference_outputs=reference)

        return evaluate_trajectory

    except ImportError:
        print("Error: agentevals package not installed")
        print("Install with: uv pip install agentevals")
        sys.exit(1)


def create_llm_judge_evaluator(
    model: str = "openai:o3-mini",
    reference_trajectory: Optional[List[Dict[str, Any]]] = None,
) -> Callable:
    """Create an LLM-as-judge evaluator using AgentEvals."""
    try:
        from agentevals.trajectory.llm import (
            create_trajectory_llm_as_judge,
            TRAJECTORY_ACCURACY_PROMPT,
        )

        base_evaluator = create_trajectory_llm_as_judge(
            model=model,
            prompt=TRAJECTORY_ACCURACY_PROMPT,
        )

        def evaluate_trajectory(
            inputs: Optional[Dict[str, Any]] = None,
            outputs: Any = None,
            reference_outputs: Any = None,
            run: Any = None,
            example: Any = None,
        ) -> Dict[str, Any]:
            _ = inputs
            outputs, reference_outputs = extract_outputs_and_reference(
                outputs=outputs,
                reference_outputs=reference_outputs,
                run=run,
                example=example,
            )
            reference = reference_outputs if reference_outputs else reference_trajectory

            if reference is not None:
                return base_evaluator(outputs=outputs, reference_outputs=reference)
            return base_evaluator(outputs=outputs)

        return evaluate_trajectory

    except ImportError:
        print("Error: agentevals package not installed")
        print("Install with: uv pip install agentevals")
        sys.exit(1)


def run_evaluation(
    target_func: Callable,
    dataset_data: Any,
    evaluators: List[Callable],
    experiment_prefix: str = "trajectory_eval",
    upload_results: bool = True,
    max_concurrency: int = 4,
):
    """Run evaluation via LangSmith SDK evaluate()."""
    try:
        from langsmith import evaluate

        print("\nRunning evaluation...")
        return evaluate(
            target_func,
            data=dataset_data,
            evaluators=evaluators,
            experiment_prefix=experiment_prefix,
            upload_results=upload_results,
            max_concurrency=max_concurrency,
        )

    except ImportError:
        print("Error: langsmith package not installed")
        print("Install with: uv pip install langsmith")
        sys.exit(1)
    except Exception as e:
        print(f"Error running evaluation: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


def print_summary(results: Any):
    """Print evaluation summary."""
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)

    experiment_name = getattr(results, "experiment_name", None)
    if experiment_name:
        print(f"\nExperiment: {experiment_name}")
        print("View in LangSmith: https://smith.langchain.com")

    print("\n✅ Evaluation complete")


def main():
    parser = argparse.ArgumentParser(
        description="Run trajectory evaluation on LangGraph agent"
    )
    parser.add_argument(
        "target_function", help="Target function path (e.g., my_agent:run_agent)"
    )
    parser.add_argument(
        "dataset", help="Dataset name (LangSmith) or file path (JSON)"
    )
    parser.add_argument(
        "--method",
        choices=["match", "llm-judge"],
        default="llm-judge",
        help="Evaluation method (default: llm-judge)",
    )
    parser.add_argument(
        "--model",
        default="openai:o3-mini",
        help="LLM model for judge evaluation (default: openai:o3-mini)",
    )
    parser.add_argument(
        "--reference-trajectory",
        help="Path to reference trajectory JSON (required for match unless provided in dataset outputs)",
    )
    parser.add_argument(
        "--trajectory-match-mode",
        choices=["strict", "unordered", "subset", "superset"],
        default="strict",
        help="Trajectory matching mode (default: strict)",
    )
    parser.add_argument(
        "--experiment-prefix",
        default="trajectory_eval",
        help="Prefix for LangSmith experiment name",
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=4,
        help="Max concurrent evaluation workers (default: 4)",
    )
    parser.add_argument(
        "--no-langsmith",
        action="store_true",
        help="Run locally without uploading results to LangSmith",
    )

    args = parser.parse_args()

    # Load target function
    print(f"Loading target function: {args.target_function}")
    user_target_func = load_target_function(args.target_function)

    def eval_target(inputs: Dict[str, Any]) -> Any:
        return invoke_target(user_target_func, inputs)

    # Load dataset input
    print(f"Loading dataset: {args.dataset}")
    dataset_data = load_dataset_data(args.dataset)
    if isinstance(dataset_data, list):
        print(f"Loaded {len(dataset_data)} examples from local file")
    else:
        print(f"Using LangSmith dataset: {dataset_data}")

    reference_trajectory = None
    if args.reference_trajectory:
        with open(args.reference_trajectory, "r") as f:
            reference_trajectory = json.load(f)

    evaluators: List[Callable] = []
    if args.method == "match":
        if reference_trajectory is None:
            print(
                "Warning: --reference-trajectory not provided. "
                "Reference outputs will be read from dataset examples."
            )

        evaluators.append(
            create_trajectory_matcher(
                reference_trajectory=reference_trajectory,
                match_mode=args.trajectory_match_mode,
            )
        )
        print(
            "Using trajectory matching evaluator "
            f"(mode={args.trajectory_match_mode})"
        )

    if args.method == "llm-judge":
        evaluators.append(
            create_llm_judge_evaluator(
                model=args.model,
                reference_trajectory=reference_trajectory,
            )
        )
        print(f"Using LLM-as-judge evaluator with {args.model}")

    results = run_evaluation(
        target_func=eval_target,
        dataset_data=dataset_data,
        evaluators=evaluators,
        experiment_prefix=args.experiment_prefix,
        upload_results=not args.no_langsmith,
        max_concurrency=args.max_concurrency,
    )

    print_summary(results)


if __name__ == "__main__":
    main()
