#!/usr/bin/env node
/**
 * Initialize a new LangGraph JavaScript/TypeScript project.
 *
 * Usage:
 *   node init_langgraph_project.js <project-name> [options]
 *
 * Options:
 *   --path PATH          Output directory (default: current directory)
 *   --pattern PATTERN    Project pattern: simple or multiagent (default: simple)
 *   --graph-name NAME    Graph name for langgraph.json (default: agent)
 *   --typescript         Use TypeScript (default: false)
 */

const fs = require('fs');
const path = require('path');

function parseArgs() {
    const args = process.argv.slice(2);
    const parsed = {
        projectName: args[0],
        path: '.',
        pattern: 'simple',
        graphName: 'agent',
        typescript: false
    };

    for (let i = 1; i < args.length; i++) {
        if (args[i] === '--path' && i + 1 < args.length) {
            parsed.path = args[++i];
        } else if (args[i] === '--pattern' && i + 1 < args.length) {
            parsed.pattern = args[++i];
        } else if (args[i] === '--graph-name' && i + 1 < args.length) {
            parsed.graphName = args[++i];
        } else if (args[i] === '--typescript') {
            parsed.typescript = true;
        }
    }

    if (!parsed.projectName) {
        console.error('Error: Project name is required');
        console.error('Usage: node init_langgraph_project.js <project-name> [options]');
        process.exit(1);
    }

    return parsed;
}

function createDirectoryStructure(projectPath, pattern, useTypescript) {
    const ext = useTypescript ? 'ts' : 'js';
    const srcDir = path.join(projectPath, 'src');

    fs.mkdirSync(srcDir, { recursive: true });

    if (pattern === 'simple') {
        fs.writeFileSync(
            path.join(srcDir, `agent.${ext}`),
            getSimpleAgentTemplate(useTypescript)
        );
    } else {
        const utilsDir = path.join(srcDir, 'utils');
        fs.mkdirSync(utilsDir, { recursive: true });

        fs.writeFileSync(
            path.join(utilsDir, `state.${ext}`),
            getStateTemplate(useTypescript)
        );
        fs.writeFileSync(
            path.join(utilsDir, `nodes.${ext}`),
            getNodesTemplate(useTypescript)
        );
        fs.writeFileSync(
            path.join(utilsDir, `tools.${ext}`),
            getToolsTemplate(useTypescript)
        );
        fs.writeFileSync(
            path.join(srcDir, `agent.${ext}`),
            getMultiagentTemplate(useTypescript)
        );
    }
}

function createLanggraphJson(projectPath, graphName, useTypescript) {
    const ext = useTypescript ? 'ts' : 'js';
    const config = {
        dependencies: ["."],
        graphs: {
            [graphName]: `./src/agent.${ext}:graph`
        },
        env: ".env",
        node_version: "20"
    };

    fs.writeFileSync(
        path.join(projectPath, 'langgraph.json'),
        JSON.stringify(config, null, 2) + '\n'
    );
}

function createEnvFile(projectPath) {
    const content = `# LangSmith Configuration (optional - for tracing and monitoring)
# LANGSMITH_API_KEY=your-api-key-here
# LANGSMITH_TRACING=true
# LANGSMITH_PROJECT=your-project-name

# LLM Provider API Keys (uncomment the ones you need)
# OPENAI_API_KEY=your-openai-api-key-here
# ANTHROPIC_API_KEY=your-anthropic-api-key-here
# GOOGLE_API_KEY=your-google-api-key-here

# Other API Keys
# TAVILY_API_KEY=your-tavily-api-key-here
`;

    fs.writeFileSync(path.join(projectPath, '.env'), content);
}

function createPackageJson(projectPath, projectName, useTypescript) {
    const packageJson = {
        name: projectName,
        version: "0.1.0",
        description: "LangGraph agent application",
        type: "module",
        main: useTypescript ? "dist/agent.js" : "src/agent.js",
        scripts: {
            dev: "langgraph dev",
            ...(useTypescript && {
                build: "tsc",
                typecheck: "tsc --noEmit"
            })
        },
        dependencies: {
            "@langchain/langgraph": "^1.1.0",
            "@langchain/core": "^1.1.0",
            langchain: "^1.1.0"
        },
        devDependencies: {
            ...(useTypescript && {
                typescript: "^5.0.0",
                "@types/node": "^20.0.0"
            })
        }
    };

    fs.writeFileSync(
        path.join(projectPath, 'package.json'),
        JSON.stringify(packageJson, null, 2) + '\n'
    );
}

function createTsConfig(projectPath) {
    const tsconfig = {
        compilerOptions: {
            target: "ES2020",
            module: "ESNext",
            moduleResolution: "node",
            outDir: "./dist",
            rootDir: "./src",
            strict: true,
            esModuleInterop: true,
            skipLibCheck: true,
            forceConsistentCasingInFileNames: true,
            resolveJsonModule: true
        },
        include: ["src/**/*"],
        exclude: ["node_modules", "dist"]
    };

    fs.writeFileSync(
        path.join(projectPath, 'tsconfig.json'),
        JSON.stringify(tsconfig, null, 2) + '\n'
    );
}

function createGitignore(projectPath) {
    const content = `# Dependencies
node_modules/

# Build output
dist/
build/

# Environment
.env
.env.local

# IDE
.vscode/
.idea/
*.swp

# LangGraph
.langgraph/
`;

    fs.writeFileSync(path.join(projectPath, '.gitignore'), content);
}

function getSimpleAgentTemplate(useTypescript) {
    if (useTypescript) {
        return `import { StateGraph, START, END } from "@langchain/langgraph";
import { Annotation, messagesStateReducer } from "@langchain/langgraph";
import { BaseMessage } from "@langchain/core/messages";

// Define agent state
const AgentState = Annotation.Root({
    messages: Annotation<BaseMessage[]>({
        reducer: messagesStateReducer,
    })
});

async function callModel(state: typeof AgentState.State) {
    // TODO: Replace with your LLM provider
    // import { ChatOpenAI } from "@langchain/openai";
    // const model = new ChatOpenAI({ model: "gpt-4" });
    // const response = await model.invoke(state.messages);
    // return { messages: [response] };

    throw new Error("Please configure your LLM provider");
}

// Build graph
const graphBuilder = new StateGraph(AgentState)
    .addNode("call_model", callModel)
    .addEdge(START, "call_model")
    .addEdge("call_model", END);

export const graph = graphBuilder.compile();
`;
    } else {
        return `import { StateGraph, START, END } from "@langchain/langgraph";
import { Annotation, messagesStateReducer } from "@langchain/langgraph";

// Define agent state
const AgentState = Annotation.Root({
    messages: Annotation({
        reducer: messagesStateReducer,
    })
});

async function callModel(state) {
    // TODO: Replace with your LLM provider
    // import { ChatOpenAI } from "@langchain/openai";
    // const model = new ChatOpenAI({ model: "gpt-4" });
    // const response = await model.invoke(state.messages);
    // return { messages: [response] };

    throw new Error("Please configure your LLM provider");
}

// Build graph
const graphBuilder = new StateGraph(AgentState)
    .addNode("call_model", callModel)
    .addEdge(START, "call_model")
    .addEdge("call_model", END);

export const graph = graphBuilder.compile();
`;
    }
}

function getMultiagentTemplate(useTypescript) {
    const imports = useTypescript
        ? `import { StateGraph, START, END } from "@langchain/langgraph";
import { AgentState } from "./utils/state.js";
import { supervisor, worker1, worker2 } from "./utils/nodes.js";`
        : `import { StateGraph, START, END } from "@langchain/langgraph";
import { AgentState } from "./utils/state.js";
import { supervisor, worker1, worker2 } from "./utils/nodes.js";`;

    return `${imports}

function buildGraph() {
    const graphBuilder = new StateGraph(AgentState)
        .addNode("supervisor", supervisor)
        .addNode("worker_1", worker1)
        .addNode("worker_2", worker2)
        .addEdge(START, "supervisor")
        .addConditionalEdges(
            "supervisor",
            (state) => state.nextAgent || "END",
            {
                worker_1: "worker_1",
                worker_2: "worker_2",
                END: END
            }
        )
        .addEdge("worker_1", "supervisor")
        .addEdge("worker_2", "supervisor");

    return graphBuilder.compile();
}

export const graph = buildGraph();
`;
}

function getStateTemplate(useTypescript) {
    if (useTypescript) {
        return `import { Annotation, messagesStateReducer } from "@langchain/langgraph";
import { BaseMessage } from "@langchain/core/messages";

export const AgentState = Annotation.Root({
    messages: Annotation<BaseMessage[]>({
        reducer: messagesStateReducer,
    }),
    nextAgent: Annotation<"worker_1" | "worker_2" | "END">({
        reducer: (_, next) => next,
        default: () => "END"
    })
});
`;
    } else {
        return `import { Annotation, messagesStateReducer } from "@langchain/langgraph";

export const AgentState = Annotation.Root({
    messages: Annotation({
        reducer: messagesStateReducer,
    }),
    nextAgent: Annotation({
        reducer: (_, next) => next,
        default: () => "END"
    })
});
`;
    }
}

function getNodesTemplate(useTypescript) {
    const typeAnnotations = useTypescript ? ': typeof AgentState.State' : '';

    return `import { AIMessage } from "@langchain/core/messages";
${useTypescript ? 'import { AgentState } from "./state.js";' : ''}

export async function supervisor(state${typeAnnotations}) {
    const messages = state.messages;

    // TODO: Implement supervisor logic
    if (messages.length > 2) {
        return { nextAgent: "END" };
    }

    return { nextAgent: "worker_1" };
}

export async function worker1(state${typeAnnotations}) {
    // TODO: Implement worker 1 logic
    const response = new AIMessage({ content: "Worker 1 response" });
    return { messages: [response] };
}

export async function worker2(state${typeAnnotations}) {
    // TODO: Implement worker 2 logic
    const response = new AIMessage({ content: "Worker 2 response" });
    return { messages: [response] };
}
`;
}

function getToolsTemplate(useTypescript) {
    const typeAnnotations = useTypescript ? ': string' : '';

    return `import { tool } from "@langchain/core/tools";
import { z } from "zod";

export const exampleTool = tool(
    async (input${typeAnnotations}) => {
        return \`Tool received: \${input.query}\`;
    },
    {
        name: "example_tool",
        description: "Example tool that returns the input query",
        schema: z.object({
            query: z.string().describe("The input query string"),
        }),
    }
);

export const tools = [exampleTool];
`;
}

function main() {
    const args = parseArgs();
    const basePath = path.resolve(args.path);
    const projectPath = path.join(basePath, args.projectName);

    if (fs.existsSync(projectPath)) {
        console.error(`❌ Error: Directory ${projectPath} already exists`);
        process.exit(1);
    }

    console.log(`🚀 Initializing LangGraph project: ${args.projectName}`);
    console.log(`   Pattern: ${args.pattern}`);
    console.log(`   Language: ${args.typescript ? 'TypeScript' : 'JavaScript'}`);
    console.log(`   Location: ${projectPath}`);
    console.log();

    fs.mkdirSync(projectPath, { recursive: true });

    createDirectoryStructure(projectPath, args.pattern, args.typescript);
    createLanggraphJson(projectPath, args.graphName, args.typescript);
    createEnvFile(projectPath);
    createPackageJson(projectPath, args.projectName, args.typescript);
    if (args.typescript) {
        createTsConfig(projectPath);
    }
    createGitignore(projectPath);

    console.log('✅ Created project structure');
    console.log('✅ Created langgraph.json');
    console.log('✅ Created .env template');
    console.log('✅ Created package.json');
    if (args.typescript) {
        console.log('✅ Created tsconfig.json');
    }
    console.log('✅ Created .gitignore');
    console.log();
    console.log('📦 Next steps:');
    console.log(`   1. cd ${args.projectName}`);
    console.log('   2. Install dependencies: npm install');
    console.log('   3. Configure .env with your API keys');
    console.log('   4. Start development server: npm run dev');
    console.log();
    console.log('🎯 Happy building!');
}

main();
