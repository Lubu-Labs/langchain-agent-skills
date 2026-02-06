#!/usr/bin/env python3
"""Generate a LangGraph node wrapper with retry logic and error classification.

Usage:
    uv run scripts/wrap_with_retry.py <node_name> [options]

Examples:
    uv run scripts/wrap_with_retry.py search_api --max-attempts 3
    uv run scripts/wrap_with_retry.py call_llm --max-attempts 5 --backoff-factor 2.0
    uv run scripts/wrap_with_retry.py execute_tool --with-fallback --with-llm-recovery

Generates Python code for a LangGraph node with:
- RetryPolicy configuration for transient errors
- LLM-recovery loop for recoverable errors
- Optional human-in-the-loop escalation
- Optional fallback strategy
"""

import argparse
import sys


def _literal_annotation(destinations: list[str]) -> str:
    quoted = ", ".join(f'"{destination}"' for destination in destinations)
    return f" -> Command[Literal[{quoted}]]"


def generate_retry_node(
    node_name: str,
    max_attempts: int = 3,
    initial_interval: float = 1.0,
    backoff_factor: float = 2.0,
    with_fallback: bool = False,
    with_llm_recovery: bool = False,
    with_human_escalation: bool = False,
) -> str:
    """Generate Python code for a LangGraph node with error handling."""
    uses_command = with_llm_recovery or with_fallback or with_human_escalation
    lines = []
    lines.append(f'"""Auto-generated error handling for {node_name} node."""')
    lines.append("")

    # Imports
    if uses_command:
        lines.append("from typing import Literal")
    lines.append("from langgraph.graph import StateGraph, MessagesState, START, END")
    if uses_command:
        lines.append("from langgraph.types import RetryPolicy, Command")
    else:
        lines.append("from langgraph.types import RetryPolicy")
    if with_human_escalation:
        lines.append("from langgraph.types import interrupt")
    lines.append("")

    # State
    lines.append("")
    lines.append("class State(MessagesState, total=False):")
    lines.append(f"    {node_name}_result: str")
    lines.append(f"    {node_name}_error: str")
    if with_llm_recovery:
        lines.append("    retry_count: int")
    lines.append("")

    if uses_command:
        lines.append("# Update this tuple to match your app's recoverable errors.")
        lines.append("RECOVERABLE_EXCEPTIONS = (ValueError, KeyError)")
        lines.append("")

    # Main node function
    lines.append("")
    if uses_command:
        main_destinations = ["__end__"]
        if with_llm_recovery:
            main_destinations.append(f"handle_{node_name}_error")
        elif with_fallback:
            main_destinations.append(f"{node_name}_fallback")
        lines.append(
            f"def {node_name}(state: State){_literal_annotation(main_destinations)}:"
        )
    else:
        lines.append(f"def {node_name}(state: State):")
    lines.append(f'    """Execute {node_name} with error handling."""')
    lines.append("    try:")
    lines.append(f"        # TODO: Implement {node_name} logic")
    lines.append(f'        result = "success"')
    if uses_command:
        lines.append("        return Command(")
        lines.append(
            f'            update={{"{node_name}_result": result, "{node_name}_error": ""}},'
        )
        lines.append("            goto=END,")
        lines.append("        )")
    else:
        lines.append(
            f'        return {{"{node_name}_result": result, "{node_name}_error": ""}}'
        )

    if with_llm_recovery:
        lines.append("    except RECOVERABLE_EXCEPTIONS as e:")
        lines.append("        # Store error for LLM recovery")
        lines.append("        return Command(")
        lines.append(
            f'            update={{"{node_name}_error": str(e), "retry_count": state.get("retry_count", 0)}},'
        )
        lines.append(f'            goto="handle_{node_name}_error"')
        lines.append("        )")
        lines.append("    # Let transient/fatal exceptions bubble up for RetryPolicy and visibility.")
    elif with_human_escalation:
        lines.append("    except RECOVERABLE_EXCEPTIONS as e:")
        lines.append("        # Escalate to human")
        lines.append("        response = interrupt({")
        lines.append(f'            "error": str(e),')
        lines.append(f'            "node": "{node_name}",')
        lines.append(f'            "question": "How should we handle this error?"')
        lines.append("        })")
        lines.append("        return Command(")
        lines.append(
            f'            update={{"{node_name}_result": str(response), "{node_name}_error": str(e)}},'
        )
        lines.append("            goto=END,")
        lines.append("        )")
        lines.append("    # Let transient/fatal exceptions bubble up for RetryPolicy and visibility.")
    elif with_fallback:
        lines.append("    except RECOVERABLE_EXCEPTIONS as e:")
        lines.append("        return Command(")
        lines.append(f'            update={{"{node_name}_error": str(e)}},')
        lines.append(f'            goto="{node_name}_fallback",')
        lines.append("        )")
        lines.append("    # Let transient/fatal exceptions bubble up for RetryPolicy and visibility.")
    else:
        lines.append("    except Exception as e:")
        lines.append(f'        return {{"{node_name}_result": "", "{node_name}_error": str(e)}}')
    lines.append("")

    # LLM recovery handler
    if with_llm_recovery:
        lines.append("")
        handler_destinations = [node_name, "__end__"]
        if with_fallback:
            handler_destinations.append(f"{node_name}_fallback")
        lines.append(
            f"def handle_{node_name}_error(state: State){_literal_annotation(handler_destinations)}:"
        )
        lines.append(f'    """LLM-based error recovery for {node_name}."""')
        lines.append(f'    error = state.get("{node_name}_error", "")')
        lines.append(f'    retry_count = state.get("retry_count", 0)')
        lines.append("")
        lines.append(f"    if retry_count >= {max_attempts}:")
        if with_human_escalation:
            lines.append("        # Escalate to human after max retries")
            lines.append("        response = interrupt({")
            lines.append(f'            "error": error,')
            lines.append(f'            "retries_exhausted": True,')
            if with_fallback:
                lines.append(
                    '            "question": "LLM recovery failed. Reply \\"fallback\\" to use fallback, or provide manual instructions."'
                )
            else:
                lines.append(
                    '            "question": "LLM recovery failed. Provide manual instructions."'
                )
            lines.append("        })")
            if with_fallback:
                lines.append(
                    '        if isinstance(response, str) and response.strip().lower() == "fallback":'
                )
                lines.append("            return Command(")
                lines.append(
                    f'                update={{"{node_name}_error": error, "retry_count": 0}},'
                )
                lines.append(f'                goto="{node_name}_fallback",')
                lines.append("            )")
            lines.append("        return Command(")
            lines.append(
                f'            update={{"{node_name}_result": str(response), "{node_name}_error": error, "retry_count": 0}},'
            )
            lines.append("            goto=END,")
            lines.append("        )")
        elif with_fallback:
            lines.append("        return Command(")
            lines.append(
                f'            update={{"{node_name}_error": error, "retry_count": 0}},'
            )
            lines.append(f'            goto="{node_name}_fallback",')
            lines.append("        )")
        else:
            lines.append("        return Command(")
            lines.append(
                f'            update={{"{node_name}_error": error, "retry_count": retry_count}},'
            )
            lines.append("            goto=END,")
            lines.append("        )")
        lines.append("")
        lines.append("    # TODO: Add LLM call to analyze error and suggest fix")
        lines.append("    return Command(")
        lines.append(f'        update={{"retry_count": retry_count + 1}},')
        lines.append(f'        goto="{node_name}",')
        lines.append("    )")
        lines.append("")

    # Fallback handler
    if with_fallback:
        lines.append("")
        lines.append(f"def {node_name}_fallback(state: State):")
        lines.append(f'    """Fallback strategy for {node_name}."""')
        lines.append(f'    # TODO: Implement fallback logic')
        lines.append(f'    return {{"{node_name}_result": "fallback_result", "{node_name}_error": ""}}')
        lines.append("")

    # Graph assembly
    lines.append("")
    lines.append("# --- Graph Assembly ---")
    lines.append("builder = StateGraph(State)")
    lines.append("")
    lines.append(f"builder.add_node(")
    lines.append(f'    "{node_name}",')
    lines.append(f"    {node_name},")
    lines.append(f"    retry_policy=RetryPolicy(")
    lines.append(f"        max_attempts={max_attempts},")
    lines.append(f"        initial_interval={initial_interval},")
    lines.append(f"        backoff_factor={backoff_factor},")
    lines.append(f"    ),")
    lines.append(f")")

    if with_llm_recovery:
        lines.append(f'builder.add_node("handle_{node_name}_error", handle_{node_name}_error)')

    if with_fallback:
        lines.append(f'builder.add_node("{node_name}_fallback", {node_name}_fallback)')

    lines.append("")
    lines.append(f'builder.add_edge(START, "{node_name}")')

    if not uses_command:
        lines.append(f'builder.add_edge("{node_name}", END)')
    else:
        lines.append(
            f"# {node_name} uses Command(goto=...) for routing; avoid static edges from this node."
        )
    if with_fallback:
        lines.append(f'builder.add_edge("{node_name}_fallback", END)')

    lines.append("")

    if with_human_escalation:
        lines.append("# Requires checkpointer for interrupt()")
        lines.append("from langgraph.checkpoint.memory import InMemorySaver")
        lines.append("graph = builder.compile(checkpointer=InMemorySaver())")
    else:
        lines.append("graph = builder.compile()")

    lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Generate a LangGraph node with retry and error handling."
    )
    parser.add_argument("node_name", help="Name of the node function")
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--initial-interval", type=float, default=1.0)
    parser.add_argument("--backoff-factor", type=float, default=2.0)
    parser.add_argument("--with-fallback", action="store_true")
    parser.add_argument("--with-llm-recovery", action="store_true")
    parser.add_argument("--with-human-escalation", action="store_true")
    parser.add_argument("--output", "-o", help="Output file (default: stdout)")
    args = parser.parse_args()

    if args.max_attempts < 1:
        parser.error("--max-attempts must be >= 1")
    if args.initial_interval <= 0:
        parser.error("--initial-interval must be > 0")
    if args.backoff_factor <= 0:
        parser.error("--backoff-factor must be > 0")

    code = generate_retry_node(
        node_name=args.node_name,
        max_attempts=args.max_attempts,
        initial_interval=args.initial_interval,
        backoff_factor=args.backoff_factor,
        with_fallback=args.with_fallback,
        with_llm_recovery=args.with_llm_recovery,
        with_human_escalation=args.with_human_escalation,
    )

    if args.output:
        with open(args.output, "w") as f:
            f.write(code)
        print(f"Generated: {args.output}")
    else:
        print(code)


if __name__ == "__main__":
    main()
