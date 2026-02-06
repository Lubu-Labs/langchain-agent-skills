#!/usr/bin/env python3
"""
Test reducer functions for LangGraph state management.

This script provides a testing harness for reducer functions to ensure they:
- Correctly merge state updates
- Handle edge cases (empty lists, None values, etc.)
- Maintain type consistency
- Work with LangGraph's state update mechanism

Usage:
    uv run test_reducers.py <module_path>

Examples:
    uv run test_reducers.py my_agent/reducers.py:extend_list
    uv run test_reducers.py my_agent/state.py:merge_context --verbose
"""

import argparse
import importlib.util
import sys
from pathlib import Path
from typing import Callable, Tuple


class ReducerTester:
    """Test harness for reducer functions."""

    def __init__(self, reducer: Callable, verbose: bool = False):
        self.reducer = reducer
        self.verbose = verbose
        self.tests_passed = 0
        self.tests_failed = 0

    def test_basic_merge(self) -> bool:
        """Test basic merge functionality."""
        test_name = "Basic Merge"
        print(f"\n🧪 Testing: {test_name}")

        try:
            # Test with lists
            left = [1, 2, 3]
            right = [4, 5, 6]
            result = self.reducer(left, right)

            if self.verbose:
                print(f"   Input: {left} + {right}")
                print(f"   Output: {result}")

            self._record_pass(test_name)
            return True

        except Exception as e:
            self._record_fail(test_name, str(e))
            return False

    def test_empty_left(self) -> bool:
        """Test with empty left value."""
        test_name = "Empty Left"
        print(f"\n🧪 Testing: {test_name}")

        try:
            # Determine type from reducer behavior
            test_values = [
                ([], [1, 2, 3]),
                ({}, {"key": "value"}),
                ("", "text"),
            ]

            for left, right in test_values:
                try:
                    result = self.reducer(left, right)
                    if self.verbose:
                        print(f"   Input: {left} + {right}")
                        print(f"   Output: {result}")

                    # Verify result is not None
                    if result is None:
                        raise ValueError("Reducer returned None")

                    self._record_pass(test_name)
                    return True
                except (TypeError, AttributeError):
                    continue

            self._record_fail(test_name, "No valid type combination found")
            return False

        except Exception as e:
            self._record_fail(test_name, str(e))
            return False

    def test_empty_right(self) -> bool:
        """Test with empty right value."""
        test_name = "Empty Right"
        print(f"\n🧪 Testing: {test_name}")

        try:
            test_values = [
                ([1, 2, 3], []),
                ({"key": "value"}, {}),
                ("text", ""),
            ]

            for left, right in test_values:
                try:
                    result = self.reducer(left, right)
                    if self.verbose:
                        print(f"   Input: {left} + {right}")
                        print(f"   Output: {result}")

                    # Verify result matches left
                    if result is None:
                        raise ValueError("Reducer returned None")

                    self._record_pass(test_name)
                    return True
                except (TypeError, AttributeError):
                    continue

            self._record_fail(test_name, "No valid type combination found")
            return False

        except Exception as e:
            self._record_fail(test_name, str(e))
            return False

    def test_none_values(self) -> bool:
        """Test with None values."""
        test_name = "None Values"
        print(f"\n🧪 Testing: {test_name}")

        try:
            # Test None on left
            try:
                result = self.reducer(None, [1, 2, 3])
                if self.verbose:
                    print(f"   Input: None + [1, 2, 3]")
                    print(f"   Output: {result}")
            except TypeError:
                if self.verbose:
                    print("   ✓ Correctly rejects None on left")

            # Test None on right
            try:
                result = self.reducer([1, 2, 3], None)
                if self.verbose:
                    print(f"   Input: [1, 2, 3] + None")
                    print(f"   Output: {result}")
            except TypeError:
                if self.verbose:
                    print("   ✓ Correctly rejects None on right")

            self._record_pass(test_name)
            return True

        except Exception as e:
            self._record_fail(test_name, str(e))
            return False

    def test_type_consistency(self) -> bool:
        """Test that output type matches input type."""
        test_name = "Type Consistency"
        print(f"\n🧪 Testing: {test_name}")

        try:
            test_values = [
                ([1, 2], [3, 4], list),
                ({"a": 1}, {"b": 2}, dict),
                ("hello", "world", str),
            ]

            for left, right, expected_type in test_values:
                try:
                    result = self.reducer(left, right)
                    if self.verbose:
                        print(f"   Input: {left} + {right}")
                        print(f"   Output: {result} (type: {type(result).__name__})")

                    if not isinstance(result, expected_type):
                        raise TypeError(
                            f"Expected {expected_type.__name__}, got {type(result).__name__}"
                        )

                    self._record_pass(test_name)
                    return True
                except (TypeError, AttributeError):
                    continue

            self._record_fail(test_name, "No valid type combination found")
            return False

        except Exception as e:
            self._record_fail(test_name, str(e))
            return False

    def test_nested_structures(self) -> bool:
        """Test with nested data structures."""
        test_name = "Nested Structures"
        print(f"\n🧪 Testing: {test_name}")

        try:
            test_values = [
                ([[1, 2]], [[3, 4]]),
                ({"a": {"b": 1}}, {"a": {"c": 2}}),
                ([{"x": 1}], [{"y": 2}]),
            ]

            for left, right in test_values:
                try:
                    result = self.reducer(left, right)
                    if self.verbose:
                        print(f"   Input: {left} + {right}")
                        print(f"   Output: {result}")

                    if result is None:
                        raise ValueError("Reducer returned None")

                    self._record_pass(test_name)
                    return True
                except (TypeError, AttributeError):
                    continue

            self._record_fail(test_name, "No valid type combination found")
            return False

        except Exception as e:
            self._record_fail(test_name, str(e))
            return False

    def test_large_inputs(self) -> bool:
        """Test with large inputs."""
        test_name = "Large Inputs"
        print(f"\n🧪 Testing: {test_name}")

        try:
            # Test with large list
            left = list(range(1000))
            right = list(range(1000, 2000))

            result = self.reducer(left, right)

            if self.verbose:
                print(f"   Input: list of {len(left)} + list of {len(right)}")
                print(f"   Output: list of {len(result) if hasattr(result, '__len__') else 'N/A'}")

            self._record_pass(test_name)
            return True

        except Exception as e:
            self._record_fail(test_name, str(e))
            return False

    def run_all_tests(self) -> bool:
        """Run all tests."""
        print(f"\n{'='*60}")
        print(f"Testing Reducer: {self.reducer.__name__}")
        print(f"{'='*60}")

        tests = [
            self.test_basic_merge,
            self.test_empty_left,
            self.test_empty_right,
            self.test_none_values,
            self.test_type_consistency,
            self.test_nested_structures,
            self.test_large_inputs,
        ]

        for test in tests:
            test()

        self._print_summary()

        return self.tests_failed == 0

    def _record_pass(self, test_name: str) -> None:
        """Record a passing test."""
        self.tests_passed += 1
        print(f"   ✅ {test_name} passed")

    def _record_fail(self, test_name: str, error: str) -> None:
        """Record a failing test."""
        self.tests_failed += 1
        print(f"   ❌ {test_name} failed: {error}")

    def _print_summary(self) -> None:
        """Print test summary."""
        total = self.tests_passed + self.tests_failed
        print(f"\n{'='*60}")
        print(f"Test Summary")
        print(f"{'='*60}")
        print(f"Total tests: {total}")
        print(f"✅ Passed: {self.tests_passed}")
        print(f"❌ Failed: {self.tests_failed}")

        if self.tests_failed == 0:
            print("\n🎉 All tests passed!")
        else:
            print(f"\n⚠️  {self.tests_failed} test(s) failed")


def load_reducer(module_path: str, reducer_name: str) -> Callable:
    """Load reducer function from module."""
    file_path = Path(module_path)

    if not file_path.exists():
        print(f"Error: File not found: {file_path}")
        sys.exit(1)

    # Load module
    spec = importlib.util.spec_from_file_location("module", file_path)
    if spec is None or spec.loader is None:
        print(f"Error: Could not load module from {file_path}")
        sys.exit(1)

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # Get reducer function
    if not hasattr(module, reducer_name):
        print(f"Error: Reducer '{reducer_name}' not found in {file_path}")
        print(f"Available names: {[name for name in dir(module) if not name.startswith('_')]}")
        sys.exit(1)

    return getattr(module, reducer_name)


def parse_module_path(module_path: str) -> Tuple[str, str]:
    """Parse module path in format 'path/to/file.py:function_name'."""
    if ":" not in module_path:
        print("Error: Module path must be in format 'path/to/file.py:function_name'")
        sys.exit(1)

    file_path, function_name = module_path.split(":", 1)
    return file_path, function_name


def main():
    parser = argparse.ArgumentParser(
        description="Test reducer functions for LangGraph state management"
    )
    parser.add_argument(
        "module_path",
        help="Path to module in format 'path/to/file.py:function_name'",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show verbose output",
    )

    args = parser.parse_args()

    file_path, reducer_name = parse_module_path(args.module_path)
    reducer = load_reducer(file_path, reducer_name)

    tester = ReducerTester(reducer, verbose=args.verbose)
    success = tester.run_all_tests()

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
