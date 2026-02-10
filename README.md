# Agent Skills for LangChain, LangGraph & LangSmith

A comprehensive collection of agent-optimized skills for AI coding assistants working in the LangChain ecosystem. These skills cover the complete development lifecycle from project setup to production deployment, monitoring, and debugging.

Each skill is a self-contained package with a `SKILL.md` entry point plus optional scripts, references, and templates.

## Available Skills

| Skill | Description |
| --- | --- |
| [langgraph-project-setup](./skills/langgraph-project-setup/) | Initialize and configure LangGraph projects (structure, `langgraph.json`, env vars, dependencies). |
| [langgraph-agent-patterns](./skills/langgraph-agent-patterns/) | Multi-agent coordination patterns: supervisor, router, orchestrator-worker, handoffs. |
| [langgraph-state-management](./skills/langgraph-state-management/) | State schemas, reducers, persistence, checkpoint inspection, and migration workflows. |
| [langgraph-error-handling](./skills/langgraph-error-handling/) | Retry strategies, LLM-based recovery loops, and human-in-the-loop escalation patterns. |
| [langgraph-testing-evaluation](./skills/langgraph-testing-evaluation/) | Test and evaluate LangGraph agents with unit/integration patterns, trajectory evaluation, LangSmith dataset evals, and A/B comparisons. |
| [langsmith-trace-analyzer](./skills/langsmith-trace-analyzer/) | Fetch, organize, and analyze LangSmith traces for debugging and performance optimization. |
| [langsmith-deployment](./skills/langsmith-deployment/) | Deploy, monitor, and manage LangGraph applications in production (Cloud, Hybrid, Standalone). |
| [skill-creator](./skills/skill-creator/) | Guidance for creating and maintaining skills in this repo. |

## Coverage

**7 production-ready skills** covering the complete LangGraph development lifecycle:

- 🚀 **Project Setup** - Initialize projects with proper structure and configuration
- 🤝 **Multi-Agent Patterns** - Supervisor, router, orchestrator, and handoff patterns
- 💾 **State Management** - Schemas, reducers, persistence, and checkpointing
- 🛡️ **Error Handling** - Retry policies, LLM recovery, and human-in-the-loop
- ✅ **Testing & Evaluation** - Unit/integration tests, trajectory evaluation, LangSmith evaluation, A/B testing
- 🔍 **Trace Analysis** - Debug with LangSmith traces and pattern detection
- 🌐 **Production Deployment** - Cloud, Hybrid, and Standalone deployment with monitoring

Plus the **skill-creator** meta-skill for extending this collection.

## Quick Start
Optional (recommended for contributors): create and sync a local environment.
```bash
uv venv --python=3.12
uv sync
```

1. Pick a skill folder in `skills/` and open its `SKILL.md`.
2. Use it directly in your assistant by pasting relevant sections or pointing the assistant to the file path.
3. For development tasks (creating, validating, packaging skills), follow the commands in `AGENTS.md`.

Example: initialize a new skill using the repo tooling.
```bash
uv run skills/skill-creator/scripts/init_skill.py <skill-name> --path skills/
```

## Install as Claude Code Plugin
1. Add the marketplace:
```
/plugin marketplace add Lubu-Labs/langchain-agent-skills
```
2. Install a skill:
```
/plugin install langsmith-deployment@lubu-labs-langchain-agent-skills
```
Or any other skill (e.g., `langgraph-project-setup`, `langgraph-agent-patterns`)
Or use the interactive menu:
```
/plugin menu
```
For local development:
```
claude --plugin-dir ./path/to/langchain-agent-skills
```
Once installed, Claude Code will automatically use these skills when relevant.

## Use in Other Repositories
These skills can be shared by copying a skill folder (for example `skills/langgraph-agent-patterns/`) into another repository or a supported assistant skills directory.

### OpenAI Codex CLI
Install via the Codex skill installer (replace with your repo path):
```
$skill-installer install langgraph-agent-patterns from Lubu-Labs/langchain-agent-skills
```
Or clone and copy manually:
```bash
git clone https://github.com/Lubu-Labs/langchain-agent-skills.git
cp -r langchain-agent-skills/skills/* ~/.codex/skills/
```
Restart Codex to pick up new skills.

### Cursor
Option 1: Remote rule (GitHub)
- Cursor Settings → Rules → Add Rule → Remote Rule (GitHub)
- Use: `https://github.com/Lubu-Labs/langchain-agent-skills.git`

Option 2: Local installation
```bash
# Project-level
git clone https://github.com/Lubu-Labs/langchain-agent-skills.git .cursor/skills/agent-skills

# User-level
git clone https://github.com/Lubu-Labs/langchain-agent-skills.git ~/.cursor/skills/agent-skills
```
Usage: type `/` in Agent chat to search and select skills by name.

### Other Assistants
If your assistant does not support skills directly, point it at the skill file:
```
Read skills/langsmith-deployment/SKILL.md for production deployment guidance
Read skills/langgraph-agent-patterns/SKILL.md for multi-agent patterns
```

## Repository Structure
- `skills/` - 8 skill packages (7 production skills + skill-creator)
- `.claude-plugin/marketplace.json` - Marketplace manifest for Claude Code
- Optional `*.skill` exports - Generated distribution artifacts
- `PLAN.md` - Roadmap and implementation ordering notes
- `AGENTS.md` - Complete development guidelines and workflow

## Contributing
Community contributions are welcome. We want to build great, comprehensive, and widely used LangChain agent skills together.

1. Open an issue for bugs, feature ideas, new skills, or improvements. Include context, expected behavior, and reproducible steps when possible.
2. Fork the repo and create a focused branch for your change.
3. Follow `AGENTS.md` while implementing:
   - Use `uv run` for Python scripts.
   - Keep skill instructions concise and move deep details to `references/`.
   - Validate skill structure with `uv run skills/skill-creator/scripts/quick_validate.py skills/<skill-name>/`.
4. Open a PR and include:
   - Clear summary of what changed and why.
   - Reference to related issue or `PLAN.md` item (if applicable).
   - Validation notes (script checks, `quick_validate.py` output, manual verification).
5. Request review and respond to feedback quickly so we can merge safely.

Keep changes focused and actionable. `AGENTS.md` is the single source of truth for workflow, tooling, and architecture rules.
