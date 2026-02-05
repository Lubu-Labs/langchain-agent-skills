#!/usr/bin/env python3
"""
Generate Mermaid diagram for LangGraph agent graphs.

Creates visual representations of graph structure including:
- Nodes
- Regular edges
- Conditional edges (when labels are available)
- Entry/exit points

Uses public LangGraph APIs where possible:
- graph.get_graph() for drawable graph structure
- graph.nodes for compiled graph nodes

Usage:
    python3 visualize_graph.py <path_to_graph_module> [--output diagram.md]
    python3 visualize_graph.py path/to/my_agent/graph.py:graph --output graph_diagram.md
"""

import argparse
import importlib.util
import sys
from pathlib import Path
from typing import Any, Dict, List, Set, Optional
from collections import defaultdict


class GraphVisualizer:
    """Generate Mermaid diagrams for LangGraph structures."""

    def __init__(self, graph: Any, graph_name: str = "graph"):
        """Initialize visualizer with a compiled graph.

        Args:
            graph: Compiled LangGraph object
            graph_name: Name of the graph for the diagram title
        """
        self.graph = graph
        self.graph_name = graph_name
        self.nodes: Set[str] = set()
        self.edges: Dict[str, List[str]] = defaultdict(list)
        self.conditional_edges: Dict[str, Dict[str, str]] = {}
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

    def _add_edge(self, source: Optional[str], target: Optional[str], condition: Optional[str] = None) -> None:
        """Add an edge to adjacency lists."""
        if not source or not target:
            return
        if condition is None:
            self.edges[source].append(target)
            return

        if source not in self.conditional_edges:
            self.conditional_edges[source] = {}

        condition_key = str(condition)
        if condition_key in self.conditional_edges[source]:
            suffix = 2
            while f"{condition_key}_{suffix}" in self.conditional_edges[source]:
                suffix += 1
            condition_key = f"{condition_key}_{suffix}"

        self.conditional_edges[source][condition_key] = target

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
                condition = None

                if isinstance(edge, tuple) and len(edge) >= 2:
                    source, target = edge[0], edge[1]
                elif isinstance(edge, dict):
                    source = edge.get("source") or edge.get("from")
                    target = edge.get("target") or edge.get("to")
                    condition = edge.get("condition") or edge.get("label")
                else:
                    for attr in ("source", "from_", "from", "src"):
                        if hasattr(edge, attr):
                            source = getattr(edge, attr)
                            break
                    for attr in ("target", "to", "dst"):
                        if hasattr(edge, attr):
                            target = getattr(edge, attr)
                            break
                    if hasattr(edge, "condition"):
                        condition = getattr(edge, "condition")
                    elif hasattr(edge, "label"):
                        condition = getattr(edge, "label")

                if source is not None and target is not None:
                    self._add_edge(str(source), str(target), condition=str(condition) if condition else None)
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
                print("❌ Cannot access graph nodes - unsupported graph structure")
                return False

            # Extract edges (best-effort)
            if hasattr(graph_obj, "edges"):
                self._extract_edges_from(graph_obj.edges)
            if not self.edges and not self.conditional_edges and hasattr(self.graph, "edges"):
                self._extract_edges_from(self.graph.edges)

            if not self.edges and not self.conditional_edges:
                self._structure_warnings.append(
                    "Could not extract edges from graph structure. "
                    "Diagram will include nodes only."
                )

            # Find entry point (best-effort)
            if hasattr(graph_obj, "entrypoint"):
                self.entry_point = getattr(graph_obj, "entrypoint")
            elif "__start__" in self.nodes:
                self.entry_point = "__start__"

            if self._structure_warnings:
                for warning in self._structure_warnings:
                    print(f"⚠️  {warning}")

            return True

        except Exception as e:
            print(f"❌ Error extracting graph structure: {e}")
            return False

    def _sanitize_node_id(self, node: str) -> str:
        """Sanitize node name for Mermaid ID.

        Args:
            node: Node name

        Returns:
            Sanitized node ID
        """
        return node.replace('-', '_').replace(' ', '_').replace('.', '_')

    def _get_node_style(self, node: str) -> str:
        """Get Mermaid node style based on node type.

        Args:
            node: Node name

        Returns:
            Mermaid node style string
        """
        if node in {'__start__', 'START'}:
            return f'{self._sanitize_node_id(node)}([START])'
        elif node in {'__end__', 'END'}:
            return f'{self._sanitize_node_id(node)}([END])'
        elif node == self.entry_point:
            return f'{self._sanitize_node_id(node)}[{node}]:::entry'
        else:
            return f'{self._sanitize_node_id(node)}[{node}]'

    def generate_mermaid(self) -> str:
        """Generate Mermaid diagram code.

        Returns:
            Mermaid diagram as string
        """
        if not self._extract_graph_structure():
            return ""

        lines = [
            "```mermaid",
            f"graph TD",
            f"    %% {self.graph_name}",
            ""
        ]

        # Add nodes
        lines.append("    %% Nodes")
        for node in sorted(self.nodes):
            node_id = self._sanitize_node_id(node)
            node_def = self._get_node_style(node)
            lines.append(f"    {node_def}")

        lines.append("")

        # Add regular edges
        if self.edges:
            lines.append("    %% Regular Edges")
            for source in sorted(self.edges.keys()):
                source_id = self._sanitize_node_id(source)
                for target in sorted(self.edges[source]):
                    target_id = self._sanitize_node_id(target)
                    lines.append(f"    {source_id} --> {target_id}")

            lines.append("")

        # Add conditional edges
        if self.conditional_edges:
            lines.append("    %% Conditional Edges")
            for source in sorted(self.conditional_edges.keys()):
                source_id = self._sanitize_node_id(source)
                conditions = self.conditional_edges[source]

                for condition, target in sorted(conditions.items()):
                    target_id = self._sanitize_node_id(target)
                    # Use dashed line for conditional edges with condition label
                    lines.append(f"    {source_id} -.{condition}.-> {target_id}")

            lines.append("")

        # Add styling
        lines.extend([
            "    %% Styling",
            "    classDef entry fill:#e1f5ff,stroke:#01579b,stroke-width:3px",
            "    classDef default fill:#f5f5f5,stroke:#666,stroke-width:2px",
        ])

        lines.append("```")

        return '\n'.join(lines)

    def generate_full_document(self) -> str:
        """Generate full markdown document with diagram and legend.

        Returns:
            Complete markdown document as string
        """
        doc = [
            f"# {self.graph_name.replace('_', ' ').title()} Graph",
            "",
            "## Graph Structure",
            "",
            self.generate_mermaid(),
            "",
            "## Legend",
            "",
            "- **Rectangle nodes**: Regular agent nodes",
            "- **Rounded nodes ([])**: START/END nodes",
            "- **Solid arrows (-->)**: Regular edges (unconditional flow)",
            "- **Dashed arrows (-. .->)**: Conditional edges (with condition label)",
            "- **Blue highlight**: Entry point node",
            "",
            "## Graph Statistics",
            "",
            f"- **Total nodes**: {len(self.nodes)}",
            f"- **Entry point**: {self.entry_point or 'Unknown'}",
            f"- **Regular edges**: {sum(len(targets) for targets in self.edges.values())}",
            f"- **Conditional edges**: {sum(len(conditions) for conditions in self.conditional_edges.values())}",
            "",
            "## Nodes",
            ""
        ]

        # List all nodes
        for node in sorted(self.nodes):
            outgoing = []

            if node in self.edges:
                outgoing.extend(f"{t}" for t in self.edges[node])

            if node in self.conditional_edges:
                outgoing.extend(f"{t} (condition: {c})" for c, t in self.conditional_edges[node].items())

            if outgoing:
                doc.append(f"- **{node}**")
                doc.append(f"  - Outgoing: {', '.join(outgoing)}")
            else:
                doc.append(f"- **{node}** (no outgoing edges)")

        return '\n'.join(doc)


def load_graph_from_path(module_path: str) -> tuple[Optional[Any], str]:
    """Load graph from module path.

    Args:
        module_path: Path in format "path/to/module.py:graph_name"

    Returns:
        Tuple of (loaded graph object or None, graph name)
    """
    if ':' not in module_path:
        print(f"❌ Invalid path format. Use: path/to/module.py:graph_name")
        return None, ""

    file_path, graph_name = module_path.split(':', 1)
    file_path = Path(file_path)

    if not file_path.exists():
        print(f"❌ File not found: {file_path}")
        return None, ""

    # Load module
    try:
        spec = importlib.util.spec_from_file_location("graph_module", file_path)
        if spec is None or spec.loader is None:
            print(f"❌ Cannot load module from {file_path}")
            return None, ""

        module = importlib.util.module_from_spec(spec)
        sys.modules["graph_module"] = module
        spec.loader.exec_module(module)

        # Get graph object
        if not hasattr(module, graph_name):
            print(f"❌ Graph '{graph_name}' not found in {file_path}")
            return None, ""

        graph = getattr(module, graph_name)
        return graph, graph_name

    except Exception as e:
        print(f"❌ Error loading graph: {e}")
        import traceback
        traceback.print_exc()
        return None, ""


def main():
    parser = argparse.ArgumentParser(
        description="Generate Mermaid diagram for LangGraph agent graphs"
    )
    parser.add_argument(
        "graph_path",
        help="Path to graph module in format: path/to/module.py:graph_name"
    )
    parser.add_argument(
        "--output",
        "-o",
        help="Output file path (default: print to stdout)"
    )
    parser.add_argument(
        "--diagram-only",
        action="store_true",
        help="Output only the Mermaid diagram without full document"
    )

    args = parser.parse_args()

    print(f"🚀 Loading graph from {args.graph_path}")

    graph, graph_name = load_graph_from_path(args.graph_path)
    if graph is None:
        return 1

    print(f"✅ Loaded graph: {graph_name}")
    print("🎨 Generating Mermaid diagram...")

    visualizer = GraphVisualizer(graph, graph_name)

    if args.diagram_only:
        output = visualizer.generate_mermaid()
    else:
        output = visualizer.generate_full_document()

    if args.output:
        output_path = Path(args.output)
        output_path.write_text(output)
        print(f"✅ Diagram saved to {output_path}")
    else:
        print()
        print(output)

    return 0


if __name__ == "__main__":
    exit(main())
