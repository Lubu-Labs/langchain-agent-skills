# Repository Guidelines

`AGENTS.md` is the single source of truth for contributor and agent guidance in this repository. All workflow, tooling, and architecture rules live here. `CLAUDE.md` only points to this document.

## Project Overview
This repository contains a collection of LangChain/LangGraph/LangSmith/Deep Agents skills for AI coding assistants. Skills are modular, self-contained packages that extend an agent's capabilities with specialized knowledge, workflows, and tools for the LangChain ecosystem.

## Repository Structure
```
skills/
├── skill-creator/              # Meta-skill for creating new skills
├── deepagents-setup-configuration/ # Skill for Deep Agents initialization, configuration, and validation
├── langgraph-project-setup/    # Skill for initializing LangGraph projects
├── langgraph-agent-patterns/   # Skill for multi-agent coordination patterns
├── langgraph-state-management/ # Skill for state schemas, reducers, persistence
├── langgraph-error-handling/   # Skill for retry, recovery, and escalation
├── langgraph-testing-evaluation/ # Skill for testing and evaluating agents
├── langsmith-trace-analyzer/   # Skill for fetching and analyzing LangSmith traces
├── langsmith-deployment/       # Skill for deploying agents to production
└── [2 planned skills]

Each skill follows this structure:
skill-name/
├── SKILL.md                 # Required: YAML frontmatter + markdown instructions
├── scripts/                 # Executable automation (Python/JavaScript/TypeScript)
├── references/              # Detailed documentation (loaded on-demand)
└── assets/                  # Templates, examples, schemas (optional)
```
- Package output (`*.skill`) is optional for distribution/export and should not be treated as source-of-truth.
- Marketplace/plugin paths should point to `skills/<skill-name>` directories in this repository.
- Roadmap and ordering are tracked in `PLAN.md`.

## Marketplace Configuration (Claude Code)
- Marketplace manifest: `.claude-plugin/marketplace.json`.
- This file lists installable skills, versions, and local paths for Claude Code Marketplace.
- Update it whenever you add/remove skills or bump versions.

## Development Environment

### Python Setup and Command Priority
**Always prioritize `uv` for all Python operations. Only fall back to `python3` if `uv` is not accessible.**

This applies to:
- Running scripts: `uv run script.py` (not `python3 script.py`)
- Managing environments: `uv venv`, `uv sync` (not `python -m venv`, `pip install`)
- Installing packages: prefer `uv add <package>` for project dependencies, and `uv pip install <package>` for ad-hoc environment installs

```bash
# Create virtual environment
uv venv --python=3.12

# Sync/install dependencies
uv sync

# Add dependency to the project
uv add pyyaml

# Or manually install specific packages in the current environment
uv pip install pyyaml
```

**Command Priority Order:**
1. **Preferred:** `uv run <script>` - Automatically manages environment
2. **Alternative:** Activate venv + `python3` - If uv unavailable
3. **Never use:** `python` - Always use `python3` or `uv run`

## Key Commands
**Initialize a new skill:**
```bash
# Preferred
uv run skills/skill-creator/scripts/init_skill.py <skill-name> --path skills/

# Fallback if uv not available
python3 skills/skill-creator/scripts/init_skill.py <skill-name> --path skills/
```

**Validate skill structure:**
```bash
# Preferred
uv run skills/skill-creator/scripts/quick_validate.py skills/<skill-name>/

# Fallback if uv not available
source .venv/bin/activate
python3 skills/skill-creator/scripts/quick_validate.py skills/<skill-name>/
```

**Package skill for distribution:**
```bash
# Preferred
uv run skills/skill-creator/scripts/package_skill.py skills/<skill-name>/ dist/

# Fallback if uv not available
source .venv/bin/activate
python3 skills/skill-creator/scripts/package_skill.py skills/<skill-name>/ dist/
```

## Skill Development Workflow
The canonical process for implementing skills is documented in `PLAN.md`. Follow this sequence:

1. **Initialize**: Run `uv run skills/skill-creator/scripts/init_skill.py <skill-name> --path skills/` to create skill directory structure
2. **Implement scripts**: Create Python and JavaScript versions of automation scripts
3. **Write references**: Create detailed documentation files for conditional loading
4. **Prepare assets**: Add templates, examples, or schemas
5. **Write SKILL.md**:
   - YAML frontmatter with comprehensive `description` (primary triggering mechanism)
   - Keep body under 500 lines
   - Link to reference files for detailed content
6. **Validate**: Run `uv run skills/skill-creator/scripts/quick_validate.py skills/<skill-name>/` to check structure
7. **Package**: Run `uv run skills/skill-creator/scripts/package_skill.py skills/<skill-name>/ dist/` to create optional `.skill` export artifacts
8. **Test**: Verify scripts execute correctly (use `uv run` for Python scripts) and skill triggers appropriately

## Skill Architecture Principles

### Progressive Disclosure
Skills use three-level loading to manage context efficiently:
1. **Metadata** (name + description) - Always in context
2. **SKILL.md body** - Loaded when skill triggers
3. **Bundled resources** - Loaded as needed by the agent

### Cross-Language Support
All skills must support both Python and JavaScript/TypeScript:
- Implement parallel script versions (`.py` and `.js`/`.ts`)
- Maintain consistent functionality across languages
- Provide language-specific examples in separate directories
- Document language differences in reference files

## File Organization Rules
**Scripts** (`scripts/`): Deterministic, fragile, or repetitive operations
- Test by actually running them
- Include error handling and validation
- Use similar CLI interfaces across languages (argparse in Python, commander/yargs in JS)

**References** (`references/`): Detailed documentation for conditional loading
- Split by topic/pattern for granular access
- Keep under 10k words per file (use grep patterns if larger)
- Organize by domain for skills with multiple areas

**Assets** (`assets/`): Files used in the agent's output
- Templates for boilerplate code
- Examples demonstrating subtle implementation details
- Schemas and configuration files

**Do NOT create:**
- README.md files at skill roots (use `SKILL.md` + `references/`; README files inside `assets/examples/` are acceptable)
- INSTALLATION_GUIDE.md or similar auxiliary documentation
- Generic development practice files

## SKILL.md Constraints
- **Line limit**: Keep under 500 lines
- **Description field**: Must comprehensively describe when to trigger the skill (primary triggering mechanism)
- **Content**: Only essential workflow guidance; move details to references
- **Format**: Use decision trees, comparison tables, and quick-start guides
- **Cross-references**: Clearly indicate when to load reference files

## Implementation Status

Skills have dependencies. Implementation order (see `PLAN.md` for details):

1. ✅ **langgraph-project-setup** - Initialize and configure LangGraph projects
2. ✅ **langgraph-agent-patterns** - Multi-agent coordination patterns
3. ✅ **langgraph-state-management** - State schemas, reducers, persistence
4. ✅ **langgraph-error-handling** - Retry logic, recovery, escalation
5. ✅ **langgraph-testing-evaluation** - Unit/integration tests, trajectory eval, LangSmith evaluation, A/B comparison
6. ✅ **langsmith-trace-analyzer** - Fetch and analyze LangSmith traces for debugging
7. ✅ **langsmith-deployment** - Deploy agents to LangSmith Cloud, Hybrid, or Standalone
8. ✅ **deepagents-setup-configuration** - Initialize, configure, validate, and troubleshoot Deep Agents projects

**Note:** langsmith-trace-analyzer, langsmith-deployment, and deepagents-setup-configuration were prioritized to complete production deployment/debugging and Deep Agents setup workflows.

## Testing Strategy
There is no centralized test suite. Use script validation as the primary check:

### Script Testing
Run scripts manually to verify functionality:
```bash
# Example: Test project initialization (using uv run)
cd /tmp
uv run /path/to/scripts/init_langgraph_project.py test-project
ls -la test-project/
rm -rf test-project

# Fallback without uv
python3 /path/to/scripts/init_langgraph_project.py test-project
```

### Skill Validation
- Structure: `uv run skills/skill-creator/scripts/quick_validate.py skills/<skill-name>/` checks required files and format
- Scripts: Execute with various inputs to ensure robustness
- Examples: Verify all examples work end-to-end

## Coding Style & Naming Conventions
- Skill folders use kebab-case (for example `langgraph-state-management`).
- Each `SKILL.md` starts with YAML frontmatter (`name`, `description`).
- Keep `SKILL.md` concise (target under 500 lines) and move deep details into `references/`.
- Avoid adding README-style files at skill roots (prefer `SKILL.md` + `references/`).

## Commit & Pull Request Guidelines
- Commits follow a Conventional Commits-style prefix (seen in history: `feat: ...`).
- PRs should include:
  - A short summary of the skill change or new skill.
  - Linked issue or roadmap reference when applicable (`PLAN.md`).
  - Validation notes (for example, `quick_validate.py` output) and any manual script checks.

## Project Context
This repository follows the skill-creator patterns documented in `skills/skill-creator/SKILL.md`. Key points:

- **Conciseness**: Challenge every piece of information - does the agent really need this?
- **Degrees of freedom**: Match specificity to task fragility
  - High freedom: Text instructions for flexible approaches
  - Medium freedom: Pseudocode/parameterized scripts
  - Low freedom: Specific scripts for fragile operations
- **Token efficiency**: Skills share the context window with system prompt, conversation history, and user requests

## LangChain Ecosystem Context
Skills in this repository target developers working with:
- **LangChain**: Framework for building LLM applications
- **LangGraph**: Low-level orchestration framework for stateful agents
- **LangSmith**: Managed hosting platform for deployment and monitoring
- **Deep Agents**: Agent harness for planning, filesystem context management, and subagent delegation

Focus on LangChain-specific knowledge that the agent doesn't have natively, such as:
- langgraph.json schema and configuration
- Multi-agent coordination patterns (supervisor, router, orchestrator, handoffs)
- State management and reducers
- Error handling strategies (retry policies, LLM recovery, human escalation)
- Testing and evaluation (unit tests, trajectory evaluation, LangSmith datasets)
- Trace analysis and debugging workflows (LangSmith trace fetching, filtering, analysis)
- Deployment to production (Cloud, Hybrid, Standalone infrastructure)
- Monitoring, alerts, and production operations
- CI/CD integration and deployment automation
- Deep Agents setup and configuration (create_deep_agent/createDeepAgent, middleware, backends, HITL, migration)
- LangGraph Studio setup and debugging
