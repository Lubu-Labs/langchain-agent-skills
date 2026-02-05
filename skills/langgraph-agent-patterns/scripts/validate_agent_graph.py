#!/usr/bin/env python3
"""
Validate LangGraph agent graph structure for common issues.

Checks for:
- Unreachable nodes
- Cycles without proper termination
- Missing edges
- Invalid conditional routing
- Isolated subgraphs

Uses public LangGraph APIs where possible:
- graph.get_graph() for drawable graph structure
- graph.nodes for compiled graph nodes

Usage:
    python3 validate_agent_graph.py <path_to_graph_module>
    python3 validate_agent_graph.py path/to/my_agent/graph.py:graph
"""

import argparse
import importlib.util
import sys
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple, Optional
from collections import defaultdict, deque


class GraphValidator:
    """Validator for LangGraph structures."""

    def __init__(self, graph: Any):
        """Initialize validator with a compiled graph.

        Args:
            graph: Compiled LangGraph object
        """
        self.graph = graph
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.nodes: Set[str] = set()
        self.edges: Dict[str, List[str]] = defaultdict(list)
        self.conditional_edges: Dict[str, List[str]] = defaultdict(list)
        self.entry_point: Optional[str] = None
        self._structure_warnings: List[str] = []

    def _node_name(self, node: Any) -> Optional[str]:
        """Extract a node name from different node representations."""
        if isinstance(node, str):
            return node
        if isinstance(node, dict):
            for key in ("id", "name", "key"):
                if key in node:
                    return str(node[key])
        for attr in ("id", "name", "key"):
            if hasattr(node, attr):
                return str(getattr(node, attr))
        return None

    def _extract_nodes_from(self, nodes_attr: Any) -> Set[str]:
        """Normalize nodes from common LangGraph graph structures."""
        if nodes_attr is None:
            return set()
        if isinstance(nodes_attr, dict):
            return set(str(k) for k in nodes_attr.keys())
        if isinstance(nodes_attr, (list, tuple, set)):
            names = []
            for node in nodes_attr:
                name = self._node_name(node)
                if name:
                    names.append(name)
            return set(names)
        name = self._node_name(nodes_attr)
        return {name} if name else set()

    def _add_edge(self, source: Optional[str], target: Optional[str], is_conditional: bool = False) -> None:
        """Add an edge to adjacency lists."""
        if not source or not target:
            return
        self.edges[source].append(target)
        if is_conditional:
            self.conditional_edges[source].append(target)

    def _extract_edges_from(self, edges_attr: Any) -> None:
        """Extract edges from common graph representations."""
        if edges_attr is None:
            return

        # Dict form: {source: [targets]}
        if isinstance(edges_attr, dict):
            for source, targets in edges_attr.items():
                if isinstance(targets, list):
                    for target in targets:
                        self._add_edge(str(source), str(target))
                else:
                    self._add_edge(str(source), str(targets))
            return

        # List/tuple form
        if isinstance(edges_attr, (list, tuple, set)):
            for edge in edges_attr:
                source = None
                target = None
                is_conditional = False

                if isinstance(edge, tuple) and len(edge) >= 2:
                    source, target = edge[0], edge[1]
                elif isinstance(edge, dict):
                    source = edge.get("source") or edge.get("from")
                    target = edge.get("target") or edge.get("to")
                    is_conditional = "condition" in edge or "label" in edge
                else:
                    for attr in ("source", "from_", "from", "src"):
                        if hasattr(edge, attr):
                            source = getattr(edge, attr)
                            break
                    for attr in ("target", "to", "dst"):
                        if hasattr(edge, attr):
                            target = getattr(edge, attr)
                            break
                    if hasattr(edge, "condition") or hasattr(edge, "label"):
                        is_conditional = True

                if source is not None and target is not None:
                    self._add_edge(str(source), str(target), is_conditional=is_conditional)
            return

    def _extract_graph_structure(self) -> bool:
        """Extract graph structure from compiled graph.

        Returns:
            True if structure was extracted successfully, False otherwise
        """
        try:
            graph_obj = self.graph

            # Prefer public drawable graph API if available
            if hasattr(self.graph, "get_graph"):
                try:
                    graph_obj = self.graph.get_graph()
                except Exception as e:
                    self._structure_warnings.append(
                        f"Could not call graph.get_graph(): {e}. Falling back to compiled graph."
                    )

            # Extract nodes (prefer drawable graph, fall back to compiled graph nodes)
            if hasattr(graph_obj, "nodes"):
                self.nodes = self._extract_nodes_from(graph_obj.nodes)
            if not self.nodes and hasattr(self.graph, "nodes"):
                self.nodes = self._extract_nodes_from(self.graph.nodes)

            if not self.nodes:
                self.errors.append("Cannot access graph nodes - unsupported graph structure")
                return False

            # Extract edges (best-effort)
            if hasattr(graph_obj, "edges"):
                self._extract_edges_from(graph_obj.edges)
            if not self.edges and hasattr(self.graph, "edges"):
                self._extract_edges_from(self.graph.edges)

            if not self.edges and not self.conditional_edges:
                self._structure_warnings.append(
                    "Could not extract edges from graph structure. "
                    "Some validation checks will be skipped."
                )

            # Find entry point (best-effort)
            if hasattr(graph_obj, "entrypoint"):
                self.entry_point = getattr(graph_obj, "entrypoint")
            elif "__start__" in self.nodes:
                self.entry_point = "__start__"
            else:
                # Infer entry point as node with no incoming edges
                incoming = defaultdict(int)
                for source, targets in self.edges.items():
                    for target in targets:
                        incoming[target] += 1
                entry_candidates = [n for n in self.nodes if incoming[n] == 0]
                if len(entry_candidates) == 1:
                    self.entry_point = entry_candidates[0]
                else:
                    self.warnings.append("Cannot determine entry point")

            if self._structure_warnings:
                self.warnings.extend(self._structure_warnings)

            return True

        except Exception as e:
            self.errors.append(f"Error extracting graph structure: {e}")
            return False

    def check_unreachable_nodes(self) -> None:
        """Check for nodes that cannot be reached from entry point."""
        if not self.entry_point:
            self.warnings.append("Cannot check reachability without entry point")
            return
        if not self.edges and not self.conditional_edges:
            self.warnings.append("Skipping reachability check because edges could not be extracted")
            return

        # BFS from entry point
        reachable = set()
        queue = deque([self.entry_point])

        while queue:
            node = queue.popleft()
            if node in reachable:
                continue

            reachable.add(node)

            # Add neighbors from regular edges
            if node in self.edges:
                queue.extend(self.edges[node])

            # Add neighbors from conditional edges
            if node in self.conditional_edges:
                queue.extend(self.conditional_edges[node])

        # Check for unreachable nodes (excluding special nodes)
        unreachable = self.nodes - reachable - {'__start__', '__end__', 'END'}

        if unreachable:
            self.errors.append(
                f"Unreachable nodes detected: {', '.join(sorted(unreachable))}. "
                "These nodes cannot be reached from the entry point."
            )

    def check_cycles(self) -> None:
        """Check for cycles that may cause infinite loops."""
        if not self.edges and not self.conditional_edges:
            self.warnings.append("Skipping cycle check because edges could not be extracted")
            return
        # Detect cycles using DFS
        visited = set()
        rec_stack = set()
        cycles_found = []

        def dfs(node: str, path: List[str]) -> None:
            """DFS to detect cycles."""
            visited.add(node)
            rec_stack.add(node)
            path.append(node)

            # Check all neighbors
            neighbors = []
            if node in self.edges:
                neighbors.extend(self.edges[node])
            if node in self.conditional_edges:
                neighbors.extend(self.conditional_edges[node])

            for neighbor in neighbors:
                # Skip END nodes
                if neighbor in {'__end__', 'END'}:
                    continue

                if neighbor not in visited:
                    dfs(neighbor, path.copy())
                elif neighbor in rec_stack:
                    # Cycle detected
                    cycle_start = path.index(neighbor)
                    cycle = path[cycle_start:] + [neighbor]
                    cycles_found.append(cycle)

            rec_stack.remove(node)

        # Start DFS from each unvisited node
        for node in self.nodes:
            if node not in visited and node not in {'__end__', 'END'}:
                dfs(node, [])

        if cycles_found:
            cycle_descriptions = []
            for cycle in cycles_found:
                cycle_descriptions.append(' -> '.join(cycle))

            self.warnings.append(
                f"Cycles detected in graph: {'; '.join(cycle_descriptions)}. "
                "Ensure there are proper exit conditions to prevent infinite loops."
            )

    def check_dead_ends(self) -> None:
        """Check for nodes with no outgoing edges (except explicit END)."""
        if not self.edges and not self.conditional_edges:
            self.warnings.append("Skipping dead-end check because edges could not be extracted")
            return
        dead_ends = []

        for node in self.nodes:
            # Skip special nodes
            if node in {'__start__', '__end__', 'END'}:
                continue

            # Check if node has any outgoing edges
            has_outgoing = (
                node in self.edges and self.edges[node] or
                node in self.conditional_edges and self.conditional_edges[node]
            )

            if not has_outgoing:
                dead_ends.append(node)

        if dead_ends:
            self.warnings.append(
                f"Dead-end nodes detected: {', '.join(sorted(dead_ends))}. "
                "These nodes have no outgoing edges. "
                "Add edges to continue the flow or route to END."
            )

    def check_conditional_routing(self) -> None:
        """Check conditional routing for potential issues."""
        if not self.conditional_edges:
            self.warnings.append("Skipping conditional routing check because conditional edges were not detected")
            return
        for node, targets in self.conditional_edges.items():
            # Check if conditional edges include END
            has_end = any(t in {'__end__', 'END'} for t in targets)

            # Check if all targets are valid nodes
            invalid_targets = [t for t in targets if t not in self.nodes and t not in {'__end__', 'END'}]

            if invalid_targets:
                self.errors.append(
                    f"Node '{node}' has conditional edges to invalid targets: {', '.join(invalid_targets)}"
                )

            if not has_end and len(targets) > 0:
                self.warnings.append(
                    f"Node '{node}' has conditional routing but no path to END. "
                    "Ensure there's a way to terminate the graph."
                )

    def check_isolated_subgraphs(self) -> None:
        """Check for isolated subgraphs disconnected from entry point."""
        if not self.entry_point:
            return
        if not self.edges and not self.conditional_edges:
            self.warnings.append("Skipping isolated subgraph check because edges could not be extracted")
            return

        # Find connected components
        visited = set()

        def dfs_component(node: str, component: Set[str]) -> None:
            """DFS to find connected component."""
            visited.add(node)
            component.add(node)

            # Check all neighbors (bidirectional)
            neighbors = []
            if node in self.edges:
                neighbors.extend(self.edges[node])
            if node in self.conditional_edges:
                neighbors.extend(self.conditional_edges[node])

            # Also check reverse edges
            for n, targets in {**self.edges, **self.conditional_edges}.items():
                if node in targets:
                    neighbors.append(n)

            for neighbor in neighbors:
                if neighbor not in visited and neighbor not in {'__start__', '__end__', 'END'}:
                    dfs_component(neighbor, component)

        # Find main component from entry point
        main_component: Set[str] = set()
        if self.entry_point in self.nodes:
            dfs_component(self.entry_point, main_component)

        # Check for nodes not in main component
        isolated = self.nodes - main_component - {'__start__', '__end__', 'END'}

        if isolated:
            self.warnings.append(
                f"Isolated nodes not connected to main flow: {', '.join(sorted(isolated))}"
            )

    def validate(self) -> bool:
        """Run all validation checks.

        Returns:
            True if validation passed (no errors), False otherwise
        """
        print("🔍 Analyzing graph structure...")

        if not self._extract_graph_structure():
            return False

        print(f"   Found {len(self.nodes)} nodes")
        print(f"   Entry point: {self.entry_point or 'Unknown'}")
        print()

        # Run validation checks
        print("🔍 Running validation checks...")
        self.check_unreachable_nodes()
        self.check_cycles()
        self.check_dead_ends()
        self.check_conditional_routing()
        self.check_isolated_subgraphs()

        # Display results
        print()
        if self.errors:
            print("❌ ERRORS:")
            for error in self.errors:
                print(f"   • {error}")
            print()

        if self.warnings:
            print("⚠️  WARNINGS:")
            for warning in self.warnings:
                print(f"   • {warning}")
            print()

        if not self.errors and not self.warnings:
            print("✅ Graph validation passed! No issues found.")
            return True
        elif not self.errors:
            print("✅ Graph validation passed with warnings.")
            return True
        else:
            print("❌ Graph validation failed.")
            return False


def load_graph_from_path(module_path: str) -> Optional[Any]:
    """Load graph from module path.

    Args:
        module_path: Path in format "path/to/module.py:graph_name"

    Returns:
        Loaded graph object or None if loading failed
    """
    if ':' not in module_path:
        print(f"❌ Invalid path format. Use: path/to/module.py:graph_name")
        return None

    file_path, graph_name = module_path.split(':', 1)
    file_path = Path(file_path)

    if not file_path.exists():
        print(f"❌ File not found: {file_path}")
        return None

    # Load module
    try:
        spec = importlib.util.spec_from_file_location("graph_module", file_path)
        if spec is None or spec.loader is None:
            print(f"❌ Cannot load module from {file_path}")
            return None

        module = importlib.util.module_from_spec(spec)
        sys.modules["graph_module"] = module
        spec.loader.exec_module(module)

        # Get graph object
        if not hasattr(module, graph_name):
            print(f"❌ Graph '{graph_name}' not found in {file_path}")
            return None

        graph = getattr(module, graph_name)
        return graph

    except Exception as e:
        print(f"❌ Error loading graph: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Validate LangGraph agent graph structure"
    )
    parser.add_argument(
        "graph_path",
        help="Path to graph module in format: path/to/module.py:graph_name"
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Treat warnings as errors"
    )

    args = parser.parse_args()

    print(f"🚀 Loading graph from {args.graph_path}")
    print()

    graph = load_graph_from_path(args.graph_path)
    if graph is None:
        return 1

    validator = GraphValidator(graph)
    success = validator.validate()

    if args.strict and validator.warnings:
        print("\n❌ Strict mode: treating warnings as errors")
        return 1

    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
