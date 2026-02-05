# JavaScript/TypeScript Project Structure

Standard directory layouts for LangGraph JavaScript/TypeScript projects.

## Simple Agent Pattern

```
my-agent/
├── src/
│   └── agent.ts          # Graph definition
├── .env                  # Environment variables
├── .gitignore           # Git ignore rules
├── langgraph.json       # LangGraph config
├── package.json         # Dependencies
└── tsconfig.json        # TypeScript config (if using TS)
```

## Multi-Agent Pattern

```
my-agent/
├── src/
│   ├── utils/
│   │   ├── state.ts     # State definitions
│   │   ├── nodes.ts     # Node functions
│   │   └── tools.ts     # Tool definitions
│   └── agent.ts         # Graph builder
├── .env
├── .gitignore
├── langgraph.json
├── package.json
└── tsconfig.json
```

## Key Files

### agent.ts
Must export a compiled graph:

```typescript
import { StateGraph, START, END } from "@langchain/langgraph";

// ... build graph ...

export const graph = graphBuilder.compile();
```

### package.json

```json
{
  "name": "my-agent",
  "version": "0.1.0",
  "type": "module",
  "dependencies": {
    "@langchain/langgraph": "^0.2.0",
    "@langchain/core": "^0.3.0",
    "@langchain/openai": "^0.3.0"
  },
  "devDependencies": {
    "typescript": "^5.0.0",
    "@types/node": "^20.0.0"
  }
}
```

### tsconfig.json

```json
{
  "compilerOptions": {
    "target": "ES2020",
    "module": "ESNext",
    "moduleResolution": "node",
    "outDir": "./dist",
    "rootDir": "./src",
    "strict": true,
    "esModuleInterop": true
  }
}
```

## Installation

```bash
npm install
# or
yarn install
# or
pnpm install
```

## References

- Use the init script: `node scripts/init_langgraph_project.js my-agent --typescript`
