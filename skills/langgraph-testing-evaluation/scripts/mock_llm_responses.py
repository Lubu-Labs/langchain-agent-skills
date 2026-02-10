#!/usr/bin/env python3
"""
Mock LLM responses for deterministic testing.

This module provides utilities for mocking LLM responses in tests,
enabling fast, deterministic, and cost-free testing of agent logic.

Usage:
    from mock_llm_responses import MockLLM, SequenceMockLLM

    # Single response mock
    mock = MockLLM(response="Mocked response")

    # Sequence of responses
    mock = SequenceMockLLM(responses=["First", "Second", "Third"])

Can also be used as a CLI tool to generate mock response configurations:
    python mock_llm_responses.py create --output mocks.json
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


class MockLLM:
    """Mock LLM that returns a fixed response."""

    def __init__(
        self,
        response: str = "Mocked response",
        model: str = "mock-model",
        tool_calls: Optional[List[Dict]] = None,
    ):
        """
        Initialize mock LLM.

        Args:
            response: Text response to return
            model: Model name to report
            tool_calls: Optional list of tool calls to include in response
        """
        self.response = response
        self.model = model
        self.tool_calls = tool_calls or []
        self.call_count = 0
        self.call_history = []

    def invoke(self, messages: List[Dict], **kwargs) -> Dict[str, Any]:
        """Mock invoke method compatible with LangChain chat models."""
        self.call_count += 1
        self.call_history.append({"messages": messages, "kwargs": kwargs})

        response = {
            "role": "assistant",
            "content": self.response,
        }

        if self.tool_calls:
            response["tool_calls"] = self.tool_calls

        return response

    async def ainvoke(self, messages: List[Dict], **kwargs) -> Dict[str, Any]:
        """Async version of invoke."""
        return self.invoke(messages, **kwargs)

    def __call__(self, messages: List[Dict], **kwargs) -> Dict[str, Any]:
        """Allow calling the mock directly."""
        return self.invoke(messages, **kwargs)

    def reset(self):
        """Reset call history."""
        self.call_count = 0
        self.call_history = []


class SequenceMockLLM:
    """Mock LLM that returns a sequence of responses."""

    def __init__(
        self,
        responses: List[Union[str, Dict]],
        model: str = "mock-model",
        loop: bool = False,
    ):
        """
        Initialize sequence mock LLM.

        Args:
            responses: List of responses (strings or dicts with content/tool_calls)
            model: Model name to report
            loop: If True, loop back to start after last response
        """
        self.responses = responses
        self.model = model
        self.loop = loop
        self.current_index = 0
        self.call_count = 0
        self.call_history = []

    def invoke(self, messages: List[Dict], **kwargs) -> Dict[str, Any]:
        """Mock invoke method that returns responses in sequence."""
        self.call_count += 1
        self.call_history.append({"messages": messages, "kwargs": kwargs})

        # Get current response
        if self.current_index >= len(self.responses):
            if self.loop:
                self.current_index = 0
            else:
                raise ValueError(
                    f"Mock LLM ran out of responses (called {self.call_count} times, only {len(self.responses)} responses)"
                )

        response_data = self.responses[self.current_index]
        self.current_index += 1

        # Format response
        if isinstance(response_data, str):
            response = {"role": "assistant", "content": response_data}
        elif isinstance(response_data, dict):
            response = {"role": "assistant", **response_data}
        else:
            response = {"role": "assistant", "content": str(response_data)}

        return response

    async def ainvoke(self, messages: List[Dict], **kwargs) -> Dict[str, Any]:
        """Async version of invoke."""
        return self.invoke(messages, **kwargs)

    def __call__(self, messages: List[Dict], **kwargs) -> Dict[str, Any]:
        """Allow calling the mock directly."""
        return self.invoke(messages, **kwargs)

    def reset(self):
        """Reset to start of sequence."""
        self.current_index = 0
        self.call_count = 0
        self.call_history = []


class ConditionalMockLLM:
    """Mock LLM that returns different responses based on input conditions."""

    def __init__(
        self,
        response_map: Dict[str, Union[str, Dict]],
        default_response: str = "Default mock response",
        model: str = "mock-model",
    ):
        """
        Initialize conditional mock LLM.

        Args:
            response_map: Dict mapping input patterns to responses
            default_response: Response to use when no pattern matches
            model: Model name to report
        """
        self.response_map = response_map
        self.default_response = default_response
        self.model = model
        self.call_count = 0
        self.call_history = []

    def invoke(self, messages: List[Dict], **kwargs) -> Dict[str, Any]:
        """Mock invoke that returns response based on input."""
        self.call_count += 1
        self.call_history.append({"messages": messages, "kwargs": kwargs})

        # Extract last user message
        user_message = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                user_message = msg.get("content", "")
                break

        # Find matching response
        response_data = self.default_response
        for pattern, response in self.response_map.items():
            if pattern.lower() in user_message.lower():
                response_data = response
                break

        # Format response
        if isinstance(response_data, str):
            response = {"role": "assistant", "content": response_data}
        elif isinstance(response_data, dict):
            response = {"role": "assistant", **response_data}
        else:
            response = {"role": "assistant", "content": str(response_data)}

        return response

    async def ainvoke(self, messages: List[Dict], **kwargs) -> Dict[str, Any]:
        """Async version of invoke."""
        return self.invoke(messages, **kwargs)

    def __call__(self, messages: List[Dict], **kwargs) -> Dict[str, Any]:
        """Allow calling the mock directly."""
        return self.invoke(messages, **kwargs)

    def reset(self):
        """Reset call history."""
        self.call_count = 0
        self.call_history = []


def create_tool_call_mock(
    tool_name: str, tool_args: Dict[str, Any], content: str = ""
) -> Dict[str, Any]:
    """
    Helper to create a mock response with tool calls.

    Args:
        tool_name: Name of the tool to call
        tool_args: Arguments for the tool
        content: Optional text content

    Returns:
        Dict formatted for MockLLM
    """
    return {
        "content": content,
        "tool_calls": [
            {
                "name": tool_name,
                "args": tool_args,
                "id": f"call_{tool_name}",
            }
        ],
    }


def load_mock_config(config_path: str) -> Dict[str, Any]:
    """Load mock configuration from JSON file."""
    with open(config_path, "r") as f:
        return json.load(f)


def create_mock_from_config(config: Dict[str, Any]):
    """Create appropriate mock LLM from configuration."""
    mock_type = config.get("type", "single")

    if mock_type == "single":
        return MockLLM(
            response=config.get("response", "Mocked response"),
            tool_calls=config.get("tool_calls"),
        )

    elif mock_type == "sequence":
        return SequenceMockLLM(
            responses=config.get("responses", []),
            loop=config.get("loop", False),
        )

    elif mock_type == "conditional":
        return ConditionalMockLLM(
            response_map=config.get("response_map", {}),
            default_response=config.get("default_response", "Default mock response"),
        )

    else:
        raise ValueError(f"Unknown mock type: {mock_type}")


def main():
    """CLI for managing mock configurations."""
    parser = argparse.ArgumentParser(description="Mock LLM response utilities")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Create command
    create_parser = subparsers.add_parser("create", help="Create mock configuration")
    create_parser.add_argument(
        "--output", "-o", default="mock_config.json", help="Output file"
    )
    create_parser.add_argument(
        "--type",
        choices=["single", "sequence", "conditional"],
        default="single",
        help="Mock type",
    )

    # Validate command
    validate_parser = subparsers.add_parser(
        "validate", help="Validate mock configuration"
    )
    validate_parser.add_argument("config", help="Configuration file to validate")

    args = parser.parse_args()

    if args.command == "create":
        # Create example configuration
        if args.type == "single":
            config = {
                "type": "single",
                "response": "This is a mocked response",
                "tool_calls": [],
            }
        elif args.type == "sequence":
            config = {
                "type": "sequence",
                "responses": [
                    "First response",
                    "Second response",
                    "Third response",
                ],
                "loop": False,
            }
        elif args.type == "conditional":
            config = {
                "type": "conditional",
                "response_map": {
                    "weather": "The weather is sunny today.",
                    "time": "It is currently 3:00 PM.",
                },
                "default_response": "I don't understand the question.",
            }

        output_path = Path(args.output)
        with open(output_path, "w") as f:
            json.dump(config, f, indent=2)

        print(f"✅ Created mock configuration: {output_path}")
        print(f"   Type: {args.type}")
        print("\nEdit the configuration file and use it in your tests:")
        print(f"   config = load_mock_config('{args.output}')")
        print(f"   mock = create_mock_from_config(config)")

    elif args.command == "validate":
        try:
            config = load_mock_config(args.config)
            mock = create_mock_from_config(config)
            print(f"✅ Valid configuration: {args.config}")
            print(f"   Type: {config.get('type')}")
        except Exception as e:
            print(f"❌ Invalid configuration: {e}")
            sys.exit(1)

    else:
        parser.print_help()


if __name__ == "__main__":
    import sys

    main()
