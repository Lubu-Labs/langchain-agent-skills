#!/usr/bin/env python3
"""Classify Python exceptions into LangGraph error handling categories.

Usage:
    uv run scripts/classify_error.py <module_path>:<exception_class> [--verbose]

Examples:
    uv run scripts/classify_error.py httpx:TimeoutException --verbose
    uv run scripts/classify_error.py openai:RateLimitError
    uv run scripts/classify_error.py ValueError

Categories:
    transient    - Network timeouts, rate limits, temporary failures → use RetryPolicy
    recoverable  - Tool errors, parsing failures → loop back to LLM
    user-fixable - Missing info, unclear instructions → use interrupt()
    fatal        - Programming errors, unrecoverable → let bubble up

Notes:
    LangGraph's default RetryPolicy behavior is selective. For custom exception
    classes, prefer explicit retry_on=<ExceptionClass>.
"""

import argparse
import importlib
import sys

# Known exception classifications
TRANSIENT_PATTERNS = {
    # HTTP / network
    "TimeoutError", "TimeoutException", "ConnectError", "ConnectionError",
    "ConnectionRefusedError", "ConnectionResetError", "ConnectionAbortedError",
    "BrokenPipeError",
    # Rate limits
    "RateLimitError", "TooManyRequestsError",
    # httpx
    "ReadTimeout", "WriteTimeout", "ConnectTimeout", "PoolTimeout",
    # requests
    "Timeout", "ConnectionError",
    # Server errors
    "InternalServerError", "ServiceUnavailableError", "BadGatewayError",
    "GatewayTimeoutError",
    # aiohttp
    "ServerDisconnectedError", "ClientConnectionError",
    # Generic retryable
    "RetryError", "TemporaryError",
}

RECOVERABLE_PATTERNS = {
    # Tool / parsing
    "ToolException", "ToolError",
    "OutputParserException", "JsonDecodeError", "JSONDecodeError",
    "ValidationError", "MarshallingError",
    "ValueError", "KeyError",
    # LLM-specific
    "ContentFilterError", "InvalidRequestError",
    "BadRequestError",
}

FATAL_PATTERNS = {
    # Programming errors
    "TypeError", "AttributeError", "ImportError", "ModuleNotFoundError",
    "NameError", "SyntaxError", "RuntimeError", "ReferenceError",
    "NotImplementedError", "RecursionError",
    # Included in default non-retry behavior in LangGraph docs
    "LookupError", "StopIteration", "StopAsyncIteration", "ArithmeticError",
    "OSError",
    # Auth
    "AuthenticationError", "PermissionError", "PermissionDeniedError",
    # Data integrity
    "ZeroDivisionError", "OverflowError",
}

USER_FIXABLE_PATTERNS = {
    "FileNotFoundError", "IndexError",
    "MissingInputError", "InvalidInputError",
}


def classify_exception_name(name: str) -> tuple[str, str]:
    """Classify an exception by its class name.

    Returns:
        Tuple of (category, strategy).
    """
    if name in TRANSIENT_PATTERNS:
        return "transient", "RetryPolicy(max_attempts=3)"
    if name in RECOVERABLE_PATTERNS:
        return "recoverable", "Store error in state, loop back to LLM"
    if name in FATAL_PATTERNS:
        return "fatal", "Let bubble up for debugging"
    if name in USER_FIXABLE_PATTERNS:
        return "user-fixable", "Pause with interrupt()"

    # Heuristic: check name substrings
    lower = name.lower()
    if any(p in lower for p in ("timeout", "ratelimit", "rate_limit", "throttl")):
        return "transient", "RetryPolicy(max_attempts=3)"
    if any(p in lower for p in ("parse", "validation", "tool", "format")):
        return "recoverable", "Store error in state, loop back to LLM"
    if any(p in lower for p in ("auth", "permission", "denied", "forbidden")):
        return "fatal", "Let bubble up for debugging"
    if any(p in lower for p in ("missing", "notfound", "not_found")):
        return "user-fixable", "Pause with interrupt()"

    return "unknown", "Inspect and classify manually"


def classify_exception_class(exc_class: type) -> tuple[str, str]:
    """Classify an actual exception class by checking its MRO."""
    # Check the class name and all parent names
    for cls in exc_class.__mro__:
        category, strategy = classify_exception_name(cls.__name__)
        if category != "unknown":
            return category, strategy
    return classify_exception_name(exc_class.__name__)


def format_exception_reference(exc_class: type | None, name: str) -> str:
    """Return a Python reference string for snippets."""
    if exc_class is None:
        return name
    if exc_class.__module__ == "builtins":
        return exc_class.__name__
    return f"{exc_class.__module__}.{exc_class.__name__}"


def resolve_exception(spec: str) -> tuple[str, type | None]:
    """Resolve 'module:ClassName' or 'ClassName' to (name, class_or_None)."""
    if ":" in spec:
        module_path, class_name = spec.rsplit(":", 1)
        try:
            mod = importlib.import_module(module_path)
            exc_class = getattr(mod, class_name)
            if isinstance(exc_class, type) and issubclass(exc_class, BaseException):
                return class_name, exc_class
            return class_name, None
        except (ImportError, AttributeError):
            return class_name, None
    else:
        # Try builtins
        builtin_exc = getattr(__builtins__, spec, None) if isinstance(__builtins__, dict) else getattr(__builtins__, spec, None)
        if builtin_exc and isinstance(builtin_exc, type) and issubclass(builtin_exc, BaseException):
            return spec, builtin_exc
        return spec, None


def main():
    parser = argparse.ArgumentParser(
        description="Classify exceptions into LangGraph error handling categories."
    )
    parser.add_argument(
        "exception",
        help="Exception to classify: 'module:ClassName' or 'ClassName'",
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    name, exc_class = resolve_exception(args.exception)

    if exc_class is not None:
        category, strategy = classify_exception_class(exc_class)
        if args.verbose:
            mro = " -> ".join(c.__name__ for c in exc_class.__mro__)
            print(f"Class:    {exc_class.__module__}.{name}")
            print(f"MRO:      {mro}")
    else:
        category, strategy = classify_exception_name(name)
        if args.verbose:
            print(f"Name:     {name} (class not importable)")

    exc_ref = format_exception_reference(exc_class, name)

    print(f"Category: {category}")
    print(f"Strategy: {strategy}")

    # Print LangGraph code suggestion
    print()
    if category == "transient":
        print("Suggested code:")
        print(f"  # Ensure this exception class is imported in your node module")
        print(f"  retry_policy=RetryPolicy(max_attempts=3, retry_on={exc_ref})")
    elif category == "recoverable":
        print("Suggested code:")
        print(f"  except {exc_ref} as e:")
        print(f'      return Command(update={{"error": str(e)}}, goto="agent")')
    elif category == "user-fixable":
        print("Suggested code:")
        print(f"  except {exc_ref} as e:")
        print(f'      response = interrupt({{"error": str(e), "need": "user input"}})')
        print(f'      # interrupt() requires compile(checkpointer=...) and thread_id')
    elif category == "fatal":
        print("Suggested code:")
        print(f"  # Let {exc_ref} bubble up - do not catch")

    return 0 if category != "unknown" else 1


if __name__ == "__main__":
    sys.exit(main())
