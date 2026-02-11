#!/usr/bin/env python3
"""
Initialize a new Deep Agent project with proper structure and configuration.

Usage:
    uv run init_deep_agent_project.py <project-name> [options]

Fallback:
    python3 init_deep_agent_project.py <project-name> [options]

Options:
    --language LANG      Language: python or javascript (default: python)
    --template TEMPLATE  Template: simple, with-subagents, or cli-config (default: simple)
    --path PATH          Output directory (default: current directory)
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Literal


def create_directory_structure(
    project_path: Path,
    language: Literal["python", "javascript"]
) -> None:
    """Create the Deep Agent project directory structure."""

    # Create main directory
    project_path.mkdir(parents=True, exist_ok=True)

    # Create subdirectories
    (project_path / "tools").mkdir(exist_ok=True)

    if language == "python":
        # Create __init__.py files for Python package
        (project_path / "__init__.py").write_text("")
        (project_path / "tools" / "__init__.py").write_text("")


def create_agent_file(
    project_path: Path,
    language: Literal["python", "javascript"],
    template: Literal["simple", "with-subagents", "cli-config"]
) -> None:
    """Create the main agent file based on language and template."""

    if language == "python":
        if template == "simple":
            content = get_simple_python_template()
        elif template == "with-subagents":
            content = get_subagents_python_template()
        else:  # cli-config
            content = get_cli_config_python_template()

        file_path = project_path / "agent.py"
    else:  # javascript
        if template == "simple":
            content = get_simple_javascript_template()
        elif template == "with-subagents":
            content = get_subagents_javascript_template()
        else:  # cli-config
            content = get_cli_config_javascript_template()

        file_path = project_path / "agent.js"

    file_path.write_text(content)


def create_tools_file(
    project_path: Path,
    language: Literal["python", "javascript"]
) -> None:
    """Create example tools file."""

    if language == "python":
        content = '''"""Example tools for the Deep Agent."""

from langchain_core.tools import tool


@tool
def search_web(query: str) -> str:
    """Search the web for information.

    Args:
        query: The search query

    Returns:
        Search results as a string
    """
    # TODO: Implement actual web search
    return f"Search results for: {query}"


@tool
def calculate(expression: str) -> str:
    """Perform a mathematical calculation.

    Args:
        expression: The mathematical expression to evaluate

    Returns:
        The result of the calculation
    """
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return str(result)
    except Exception as e:
        return f"Error: {str(e)}"


# List of all tools
tools = [search_web, calculate]
'''
        file_path = project_path / "tools" / "example_tools.py"
    else:  # javascript
        content = '''/**
 * Example tools for the Deep Agent.
 */

export function searchWeb(query) {
  // TODO: Implement actual web search
  return `Search results for: ${query}`;
}

export function calculate(expression) {
  try {
    // WARNING: eval is dangerous in production - use a proper math parser
    const result = eval(expression);
    return String(result);
  } catch (e) {
    return `Error: ${e.message}`;
  }
}

// List of all tools
export const tools = [searchWeb, calculate];
'''
        file_path = project_path / "tools" / "example_tools.js"

    file_path.write_text(content)


def create_env_file(project_path: Path) -> None:
    """Create .env.example file with common environment variables."""

    content = """# LangSmith Configuration (optional - for tracing and monitoring)
# LANGSMITH_API_KEY=your-api-key-here
# LANGSMITH_TRACING=true
# LANGSMITH_PROJECT=your-project-name

# LLM Provider API Keys (uncomment the one you need)
# OPENAI_API_KEY=your-openai-api-key-here
# ANTHROPIC_API_KEY=your-anthropic-api-key-here
# GOOGLE_API_KEY=your-google-api-key-here

# Other API Keys
# TAVILY_API_KEY=your-tavily-api-key-here
"""

    (project_path / ".env.example").write_text(content)


def create_readme(
    project_path: Path,
    project_name: str,
    language: Literal["python", "javascript"],
    template: Literal["simple", "with-subagents", "cli-config"]
) -> None:
    """Create README.md file."""

    if language == "python":
        setup_commands = """## Setup

1. Create a virtual environment:
   ```bash
   uv venv --python=3.12
   # or: python3 -m venv venv
   ```

2. Activate the environment:
   ```bash
   source .venv/bin/activate
   # or: source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   uv sync
   # or: pip install -e .
   ```

4. Configure environment variables:
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

5. Run the agent:
   ```bash
   uv run agent.py
   # or: python3 agent.py
   ```"""
    else:
        setup_commands = """## Setup

1. Install dependencies:
   ```bash
   npm install deepagents langchain @langchain/core @langchain/langgraph
   ```

2. Configure environment variables:
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

3. Run the agent:
   ```bash
   node agent.js
   ```"""

    template_description = {
        "simple": "A simple Deep Agent with basic tools",
        "with-subagents": "A Deep Agent with subagent delegation capabilities",
        "cli-config": "A Deep Agent configured for CLI-style usage"
    }

    content = f"""# {project_name}

{template_description[template]}

{setup_commands}

## Project Structure

- `agent.{('py' if language == 'python' else 'js')}` - Main Deep Agent implementation
- `tools/` - Custom tools for the agent
- `.env.example` - Environment variable template

## Documentation

- [Deep Agents Documentation](https://docs.langchain.com/oss/{'python' if language == 'python' else 'javascript'}/deepagents/overview)
- [LangChain Documentation](https://docs.langchain.com/)

## Next Steps

1. Customize the system prompt in `agent.{('py' if language == 'python' else 'js')}`
2. Add your own tools in the `tools/` directory
3. Configure middleware and backends as needed
4. Test with various inputs and scenarios
"""

    (project_path / "README.md").write_text(content)


def create_pyproject_toml(project_path: Path, project_name: str) -> None:
    """Create pyproject.toml for Python projects."""

    package_name = project_name.replace("-", "_")

    content = f"""[project]
name = "{package_name}"
version = "0.1.0"
description = "Deep Agent application"
requires-python = ">=3.11"
dependencies = [
    "deepagents",
    "langchain",
    "langchain-core",
    "langgraph",
]

[project.optional-dependencies]
openai = ["langchain-openai"]
anthropic = ["langchain-anthropic"]
google = ["langchain-google-genai"]

[build-system]
requires = ["setuptools>=61.0"]
build-backend = "setuptools.build_meta"
"""

    (project_path / "pyproject.toml").write_text(content)


def create_package_json(project_path: Path, project_name: str) -> None:
    """Create package.json for JavaScript projects."""

    config = {
        "name": project_name,
        "version": "0.1.0",
        "description": "Deep Agent application",
        "type": "module",
        "main": "agent.js",
        "scripts": {
            "start": "node agent.js"
        },
        "dependencies": {
            "deepagents": "^0.1.0",
            "langchain": "^1.0.0",
            "@langchain/core": "^1.0.0",
            "@langchain/langgraph": "^1.0.0"
        }
    }

    (project_path / "package.json").write_text(json.dumps(config, indent=2) + "\n")


def create_gitignore(project_path: Path, language: Literal["python", "javascript"]) -> None:
    """Create .gitignore file."""

    if language == "python":
        content = """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
dist/
*.egg-info/
.venv
venv/
env/

# Environment variables
.env
.env.local

# IDE
.vscode/
.idea/
*.swp
*~
"""
    else:
        content = """# Node
node_modules/
npm-debug.log*
yarn-debug.log*
yarn-error.log*

# Environment variables
.env
.env.local

# IDE
.vscode/
.idea/
*.swp
*~
"""

    (project_path / ".gitignore").write_text(content)


def get_simple_python_template() -> str:
    """Get simple Python Deep Agent template."""
    return '''"""Simple Deep Agent implementation."""

from deepagents import create_deep_agent
from tools.example_tools import tools


def main():
    """Initialize and run the Deep Agent."""

    # Create agent with automatic middleware (todos, filesystem, subagents)
    agent = create_deep_agent(
        model="claude-sonnet-4-5-20250929",  # or "openai:gpt-5", etc.
        tools=tools,
        system_prompt="""You are a helpful AI assistant powered by Deep Agents.

You have access to tools that help you search the web and perform calculations.
Break down complex tasks into steps using your built-in planning capabilities.""",
        debug=True,
    )

    # Example invocation
    result = agent.invoke({
        "messages": [{"role": "user", "content": "What is the weather in San Francisco?"}]
    })

    print("\\nAgent response:")
    print(result["messages"][-1].content)


if __name__ == "__main__":
    main()
'''


def get_subagents_python_template() -> str:
    """Get Python Deep Agent template with subagents."""
    return '''"""Deep Agent with subagent delegation."""

from deepagents import create_deep_agent
from tools.example_tools import tools


def main():
    """Initialize and run the Deep Agent with subagents."""

    # Define subagents for specialized tasks
    subagents = [
        {
            "name": "researcher",
            "description": "Research specialist for finding information",
            "tools": [tools[0]],  # search_web tool
            "system_prompt": "You are a research expert. Find accurate information.",
        },
        {
            "name": "calculator",
            "description": "Math specialist for calculations",
            "tools": [tools[1]],  # calculate tool
            "system_prompt": "You are a math expert. Perform precise calculations.",
        },
    ]

    # Create supervisor agent with subagents
    agent = create_deep_agent(
        model="claude-sonnet-4-5-20250929",
        tools=[],  # Supervisor has no direct tools, only delegates
        subagents=subagents,
        system_prompt="""You are a supervisor AI that coordinates specialized subagents.

When you receive a task:
1. Analyze what type of work is needed
2. Delegate to the appropriate subagent using the 'task' tool
3. Review the results and provide a comprehensive response

Available subagents:
- researcher: For finding information
- calculator: For mathematical calculations""",
        debug=True,
    )

    # Example invocation
    result = agent.invoke({
        "messages": [{
            "role": "user",
            "content": "Find the population of Tokyo and calculate how many buses would be needed if each bus holds 50 people."
        }]
    })

    print("\\nAgent response:")
    print(result["messages"][-1].content)


if __name__ == "__main__":
    main()
'''


def get_cli_config_python_template() -> str:
    """Get Python Deep Agent template with CLI configuration."""
    return '''"""Deep Agent with CLI-style configuration."""

import argparse
from deepagents import create_deep_agent
from deepagents.backends import StateBackend, StoreBackend, CompositeBackend
from langgraph.store.memory import InMemoryStore
from langgraph.checkpoint.memory import InMemorySaver
from tools.example_tools import tools


def create_agent(
    model: str = "claude-sonnet-4-5-20250929",
    use_memory: bool = False,
    use_checkpointer: bool = False,
    debug: bool = False
):
    """Create a Deep Agent with configurable options."""

    kwargs = {
        "model": model,
        "tools": tools,
        "system_prompt": "You are a helpful AI assistant with planning capabilities.",
        "debug": debug,
    }

    # Add memory store if requested
    if use_memory:
        store = InMemoryStore()
        kwargs["store"] = store
        kwargs["backend"] = lambda rt: CompositeBackend(
            default=StateBackend(rt),
            routes={"/memories/": StoreBackend(rt)}
        )

    # Add checkpointer if requested
    if use_checkpointer:
        kwargs["checkpointer"] = InMemorySaver()

    return create_deep_agent(**kwargs)


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Run Deep Agent with configuration")
    parser.add_argument(
        "--model",
        default="claude-sonnet-4-5-20250929",
        help="Model to use (default: claude-sonnet-4-5-20250929)"
    )
    parser.add_argument(
        "--memory",
        action="store_true",
        help="Enable long-term memory store"
    )
    parser.add_argument(
        "--checkpointer",
        action="store_true",
        help="Enable state checkpointing"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode"
    )
    parser.add_argument(
        "query",
        nargs="?",
        default="Hello! What can you help me with?",
        help="Query to send to the agent"
    )

    args = parser.parse_args()

    # Create agent with configuration
    agent = create_agent(
        model=args.model,
        use_memory=args.memory,
        use_checkpointer=args.checkpointer,
        debug=args.debug
    )

    # Run query
    config = {}
    if args.checkpointer:
        config["configurable"] = {"thread_id": "default"}

    result = agent.invoke(
        {"messages": [{"role": "user", "content": args.query}]},
        config=config
    )

    print("\\nAgent response:")
    print(result["messages"][-1].content)


if __name__ == "__main__":
    main()
'''


def get_simple_javascript_template() -> str:
    """Get simple JavaScript Deep Agent template."""
    return '''/**
 * Simple Deep Agent implementation.
 */

import { createDeepAgent } from "deepagents";
import { tools } from "./tools/example_tools.js";

async function main() {
  // Create agent with automatic middleware (todos, filesystem, subagents)
  const agent = await createDeepAgent({
    model: "claude-sonnet-4-5-20250929", // or "openai:gpt-5", etc.
    tools: tools,
    systemPrompt: `You are a helpful AI assistant powered by Deep Agents.

You have access to tools that help you search the web and perform calculations.
Break down complex tasks into steps using your built-in planning capabilities.`,
    debug: true,
  });

  // Example invocation
  const result = await agent.invoke({
    messages: [{ role: "user", content: "What is the weather in San Francisco?" }],
  });

  console.log("\\nAgent response:");
  console.log(result.messages[result.messages.length - 1].content);
}

main().catch(console.error);
'''


def get_subagents_javascript_template() -> str:
    """Get JavaScript Deep Agent template with subagents."""
    return '''/**
 * Deep Agent with subagent delegation.
 */

import { createDeepAgent } from "deepagents";
import { tools } from "./tools/example_tools.js";

async function main() {
  // Define subagents for specialized tasks
  const subagents = [
    {
      name: "researcher",
      description: "Research specialist for finding information",
      tools: [tools[0]], // searchWeb tool
      systemPrompt: "You are a research expert. Find accurate information.",
    },
    {
      name: "calculator",
      description: "Math specialist for calculations",
      tools: [tools[1]], // calculate tool
      systemPrompt: "You are a math expert. Perform precise calculations.",
    },
  ];

  // Create supervisor agent with subagents
  const agent = await createDeepAgent({
    model: "claude-sonnet-4-5-20250929",
    tools: [], // Supervisor has no direct tools, only delegates
    subagents: subagents,
    systemPrompt: `You are a supervisor AI that coordinates specialized subagents.

When you receive a task:
1. Analyze what type of work is needed
2. Delegate to the appropriate subagent using the 'task' tool
3. Review the results and provide a comprehensive response

Available subagents:
- researcher: For finding information
- calculator: For mathematical calculations`,
    debug: true,
  });

  // Example invocation
  const result = await agent.invoke({
    messages: [
      {
        role: "user",
        content:
          "Find the population of Tokyo and calculate how many buses would be needed if each bus holds 50 people.",
      },
    ],
  });

  console.log("\\nAgent response:");
  console.log(result.messages[result.messages.length - 1].content);
}

main().catch(console.error);
'''


def get_cli_config_javascript_template() -> str:
    """Get JavaScript Deep Agent template with CLI configuration."""
    return '''/**
 * Deep Agent with CLI-style configuration.
 */

import { createDeepAgent, StateBackend, StoreBackend, CompositeBackend } from "deepagents";
import { InMemoryStore, MemorySaver } from "@langchain/langgraph";
import { tools } from "./tools/example_tools.js";

async function createAgent(options = {}) {
  const {
    model = "claude-sonnet-4-5-20250929",
    useMemory = false,
    useCheckpointer = false,
    debug = false,
  } = options;

  const config = {
    model,
    tools,
    systemPrompt: "You are a helpful AI assistant with planning capabilities.",
    debug,
  };

  // Add memory store if requested
  if (useMemory) {
    const store = new InMemoryStore();
    config.store = store;
    config.backend = (rt) =>
      new CompositeBackend(
        new StateBackend(rt),
        { "/memories/": new StoreBackend(rt) },
      );
  }

  // Add checkpointer if requested
  if (useCheckpointer) {
    config.checkpointer = new MemorySaver();
  }

  return await createDeepAgent(config);
}

async function main() {
  // Parse command line arguments
  const args = process.argv.slice(2);
  const options = {
    model: "claude-sonnet-4-5-20250929",
    useMemory: args.includes("--memory"),
    useCheckpointer: args.includes("--checkpointer"),
    debug: args.includes("--debug"),
  };

  // Get query (everything after flags)
  const query = args.filter((arg) => !arg.startsWith("--")).join(" ") ||
    "Hello! What can you help me with?";

  // Create agent with configuration
  const agent = await createAgent(options);

  // Run query
  const config = {};
  if (options.useCheckpointer) {
    config.configurable = { thread_id: "default" };
  }

  const result = await agent.invoke(
    { messages: [{ role: "user", content: query }] },
    config
  );

  console.log("\\nAgent response:");
  console.log(result.messages[result.messages.length - 1].content);
}

main().catch(console.error);
'''


def main():
    parser = argparse.ArgumentParser(
        description="Initialize a new Deep Agent project"
    )
    parser.add_argument(
        "project_name",
        help="Name of the project (e.g., my-deep-agent)"
    )
    parser.add_argument(
        "--language",
        choices=["python", "javascript"],
        default="python",
        help="Programming language (default: python)"
    )
    parser.add_argument(
        "--template",
        choices=["simple", "with-subagents", "cli-config"],
        default="simple",
        help="Project template (default: simple)"
    )
    parser.add_argument(
        "--path",
        default=".",
        help="Output directory (default: current directory)"
    )

    args = parser.parse_args()

    # Create project directory
    base_path = Path(args.path).resolve()
    project_path = base_path / args.project_name

    if project_path.exists():
        print(f"❌ Error: Directory {project_path} already exists")
        sys.exit(1)

    print(f"🚀 Initializing Deep Agent project: {args.project_name}")
    print(f"   Language: {args.language}")
    print(f"   Template: {args.template}")
    print(f"   Location: {project_path}")
    print()

    try:
        # Create all files and directories
        create_directory_structure(project_path, args.language)
        create_agent_file(project_path, args.language, args.template)
        create_tools_file(project_path, args.language)
        create_env_file(project_path)
        create_readme(project_path, args.project_name, args.language, args.template)
        create_gitignore(project_path, args.language)

        if args.language == "python":
            create_pyproject_toml(project_path, args.project_name)
        else:
            create_package_json(project_path, args.project_name)

        print("✅ Created project structure")
        print("✅ Created agent file")
        print("✅ Created example tools")
        print("✅ Created .env.example")
        print("✅ Created README.md")
        print("✅ Created .gitignore")
        print(f"✅ Created {'pyproject.toml' if args.language == 'python' else 'package.json'}")
        print()
        print("📦 Next steps:")
        print(f"   1. cd {args.project_name}")

        if args.language == "python":
            print("   2. Create a virtual environment: uv venv --python=3.12")
            print("   3. Activate it: source .venv/bin/activate")
            print("   4. Install dependencies: uv sync")
        else:
            print("   2. Install dependencies: npm install")

        print("   5. Copy .env.example to .env and configure API keys")
        print(f"   6. Run the agent: {'uv run agent.py' if args.language == 'python' else 'node agent.js'}")
        print()
        print("🎯 Happy building!")

    except Exception as e:
        print(f"❌ Error creating project: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
