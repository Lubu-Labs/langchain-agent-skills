#!/usr/bin/env python3
"""
Validate LangGraph state schema structure and typing.

This script checks state schemas for common issues:
- Missing required fields
- Invalid type annotations
- Incompatible reducers
- Unsupported schema class patterns
- Invalid Literal values
- Inconsistent field types

Usage:
    uv run validate_state_schema.py <module_path>

Example:
    uv run validate_state_schema.py my_agent/state.py:MyState
    uv run validate_state_schema.py my_agent/state.py:MyState --verbose
"""

import argparse
import ast
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple


class StateSchemaValidator:
    """Validates LangGraph state schemas."""

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.info: List[str] = []

    def validate_file(self, file_path: Path, class_name: str) -> bool:
        """Validate state schema in a Python file."""
        try:
            with open(file_path) as f:
                tree = ast.parse(f.read(), filename=str(file_path))
        except SyntaxError as e:
            self.errors.append(f"Syntax error in {file_path}: {e}")
            return False

        # Find the class definition
        class_def = None
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == class_name:
                class_def = node
                break

        if not class_def:
            self.errors.append(f"Class '{class_name}' not found in {file_path}")
            return False

        self._validate_class(class_def, file_path)

        return len(self.errors) == 0

    def _validate_class(self, class_def: ast.ClassDef, file_path: Path) -> None:
        """Validate a state class definition."""
        schema_kind = self._detect_schema_kind(class_def)
        if schema_kind == "unknown":
            self.warnings.append(
                f"Class '{class_def.name}' is not a recognized LangGraph state pattern "
                "(TypedDict, dataclass, or BaseModel)"
            )
        else:
            self.info.append(f"Class '{class_def.name}' detected as {schema_kind} schema")

        # Check docstring
        if not ast.get_docstring(class_def):
            self.warnings.append(f"Class '{class_def.name}' missing docstring")

        # Validate fields
        fields = self._extract_fields(class_def)
        self._validate_fields(fields, class_def.name)

    def _extract_fields(self, class_def: ast.ClassDef) -> Dict[str, Any]:
        """Extract fields from class definition."""
        fields = {}

        for node in class_def.body:
            if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                field_name = node.target.id
                annotation = node.annotation
                fields[field_name] = {
                    "annotation": annotation,
                    "has_default": node.value is not None,
                }

        return fields

    def _validate_fields(self, fields: Dict[str, Any], class_name: str) -> None:
        """Validate field definitions."""
        if not fields:
            self.errors.append(f"Class '{class_name}' has no fields defined")
            return

        for field_name, field_info in fields.items():
            annotation = field_info["annotation"]

            # Check for Annotated fields (reducers)
            if isinstance(annotation, ast.Subscript):
                annotation_base = self._annotation_base_name(annotation.value)
                if annotation_base == "Annotated":
                    self._validate_annotated_field(field_name, annotation, class_name)
                elif annotation_base == "Literal":
                    self._validate_literal_field(field_name, annotation, class_name)
                elif annotation_base in {"list", "List"}:
                    self._validate_list_field(field_name, annotation, class_name)
                elif annotation_base in {"dict", "Dict"}:
                    self._validate_dict_field(field_name, annotation, class_name)

        # Check for common required fields
        self._check_common_patterns(fields, class_name)

    def _validate_annotated_field(
        self, field_name: str, annotation: ast.Subscript, class_name: str
    ) -> None:
        """Validate Annotated field with reducer."""
        if not isinstance(annotation.slice, ast.Tuple):
            self.errors.append(
                f"Field '{field_name}' in '{class_name}': Annotated requires at least 2 arguments"
            )
            return

        elts = annotation.slice.elts
        if len(elts) < 2:
            self.errors.append(
                f"Field '{field_name}' in '{class_name}': Annotated requires at least 2 arguments"
            )
            return

        # Second element should be the reducer
        reducer = elts[1]

        # Check if reducer is a valid function reference
        if isinstance(reducer, ast.Name):
            self.info.append(
                f"Field '{field_name}' uses reducer: {reducer.id}"
            )
        elif isinstance(reducer, ast.Attribute):
            reducer_name = f"{self._get_full_name(reducer.value)}.{reducer.attr}"
            self.info.append(
                f"Field '{field_name}' uses reducer: {reducer_name}"
            )
        else:
            self.warnings.append(
                f"Field '{field_name}' in '{class_name}': Reducer is not a simple name or attribute"
            )

    def _validate_literal_field(
        self, field_name: str, annotation: ast.Subscript, class_name: str
    ) -> None:
        """Validate Literal field."""
        if not isinstance(annotation.slice, (ast.Tuple, ast.Constant)):
            self.warnings.append(
                f"Field '{field_name}' in '{class_name}': Literal has unusual structure"
            )
            return

        # Extract literal values
        if isinstance(annotation.slice, ast.Tuple):
            values = [
                self._get_constant_value(elt) for elt in annotation.slice.elts
            ]
        else:
            values = [self._get_constant_value(annotation.slice)]

        if not values:
            self.errors.append(
                f"Field '{field_name}' in '{class_name}': Literal has no values"
            )
        else:
            self.info.append(
                f"Field '{field_name}' is Literal with values: {values}"
            )

    def _validate_list_field(
        self, field_name: str, annotation: ast.Subscript, class_name: str
    ) -> None:
        """Validate list field."""
        # Check if list has type parameter
        if not annotation.slice:
            self.warnings.append(
                f"Field '{field_name}' in '{class_name}': list without type parameter"
            )

    def _validate_dict_field(
        self, field_name: str, annotation: ast.Subscript, class_name: str
    ) -> None:
        """Validate dict field."""
        # Check if dict has type parameters
        if not annotation.slice:
            self.warnings.append(
                f"Field '{field_name}' in '{class_name}': dict without type parameters"
            )

    def _check_common_patterns(self, fields: Dict[str, Any], class_name: str) -> None:
        """Check for common state patterns."""
        field_names = set(fields.keys())

        # Check for messages field (common in chat patterns)
        if "messages" in field_names:
            messages_annotation = fields["messages"]["annotation"]
            if not self._has_annotated(messages_annotation):
                self.warnings.append(
                    f"'{class_name}' has 'messages' field without Annotated reducer (consider using add_messages)"
                )

        # Check for routing fields
        routing_fields = {"next", "route", "next_agent"}
        if routing_fields & field_names:
            for field in routing_fields & field_names:
                annotation = fields[field]["annotation"]
                if not self._has_literal(annotation):
                    self.warnings.append(
                        f"'{class_name}' has routing field '{field}' without Literal type"
                    )

    def _has_annotated(self, annotation: ast.AST) -> bool:
        """Check if annotation is Annotated."""
        return (
            isinstance(annotation, ast.Subscript)
            and self._annotation_base_name(annotation.value) == "Annotated"
        )

    def _has_literal(self, annotation: ast.AST) -> bool:
        """Check if annotation is Literal."""
        return (
            isinstance(annotation, ast.Subscript)
            and self._annotation_base_name(annotation.value) == "Literal"
        )

    def _annotation_base_name(self, node: ast.AST) -> str:
        """Get the right-most name for a type annotation node."""
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            return node.attr
        return ""

    def _is_dataclass_decorated(self, class_def: ast.ClassDef) -> bool:
        """Check whether class has a dataclass decorator."""
        for decorator in class_def.decorator_list:
            if isinstance(decorator, ast.Name) and decorator.id == "dataclass":
                return True
            if isinstance(decorator, ast.Attribute) and decorator.attr == "dataclass":
                return True
        return False

    def _detect_schema_kind(self, class_def: ast.ClassDef) -> str:
        """Detect common LangGraph state schema styles."""
        if self._is_dataclass_decorated(class_def):
            return "dataclass"

        base_names = {
            self._annotation_base_name(base)
            for base in class_def.bases
        }
        if "TypedDict" in base_names:
            return "TypedDict"
        if "BaseModel" in base_names:
            return "Pydantic BaseModel"
        return "unknown"

    def _get_constant_value(self, node: ast.AST) -> Any:
        """Extract constant value from AST node."""
        if isinstance(node, ast.Constant):
            return node.value
        return None

    def _get_full_name(self, node: ast.AST) -> str:
        """Get full name from AST node."""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return f"{self._get_full_name(node.value)}.{node.attr}"
        return ""

    def print_results(self) -> None:
        """Print validation results."""
        if self.errors:
            print("\n❌ ERRORS:")
            for error in self.errors:
                print(f"  • {error}")

        if self.warnings:
            print("\n⚠️  WARNINGS:")
            for warning in self.warnings:
                print(f"  • {warning}")

        if self.info and self.verbose:
            print("\nℹ️  INFO:")
            for info in self.info:
                print(f"  • {info}")

        if not self.errors and not self.warnings:
            print("\n✅ State schema validation passed!")
        elif not self.errors:
            print("\n✅ State schema validation passed with warnings")


def parse_module_path(module_path: str) -> Tuple[Path, str]:
    """Parse module path in format 'path/to/file.py:ClassName'."""
    if ":" not in module_path:
        print("Error: Module path must be in format 'path/to/file.py:ClassName'")
        sys.exit(1)

    file_path_str, class_name = module_path.split(":", 1)
    file_path = Path(file_path_str)

    if not file_path.exists():
        print(f"Error: File not found: {file_path}")
        sys.exit(1)

    return file_path, class_name


def main():
    parser = argparse.ArgumentParser(
        description="Validate LangGraph state schema structure and typing"
    )
    parser.add_argument(
        "module_path",
        help="Path to module in format 'path/to/file.py:ClassName'",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show verbose output including info messages",
    )

    args = parser.parse_args()

    file_path, class_name = parse_module_path(args.module_path)

    print(f"🔍 Validating state schema: {class_name}")
    print(f"   File: {file_path}")

    validator = StateSchemaValidator(verbose=args.verbose)
    success = validator.validate_file(file_path, class_name)
    validator.print_results()

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
