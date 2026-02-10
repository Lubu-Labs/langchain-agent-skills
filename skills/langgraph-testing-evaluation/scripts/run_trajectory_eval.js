#!/usr/bin/env node
/**
 * Run trajectory evaluation on a LangGraph agent.
 *
 * Usage:
 *   node run_trajectory_eval.js <target_function> <dataset> [options]
 *
 * Examples:
 *   node run_trajectory_eval.js ./my-agent.ts:runAgent my_dataset --method llm-judge
 */

import { program } from "commander";
import { Client } from "langsmith";
import { evaluate } from "langsmith/evaluation";
import { createTrajectoryLLMAsJudge, TRAJECTORY_ACCURACY_PROMPT } from "agentevals";
import * as fs from "fs";
import * as path from "path";

async function loadTargetFunction(functionPath) {
  const [moduleFile, funcName] = functionPath.split(":");

  if (!moduleFile || !funcName) {
    throw new Error("Function path must be in format: ./file.ts:functionName");
  }

  const absolutePath = path.resolve(moduleFile);
  const module = await import(absolutePath);

  if (!(funcName in module)) {
    throw new Error(`Module '${moduleFile}' has no export '${funcName}'`);
  }

  return module[funcName];
}

async function loadDataset(datasetPath, client) {
  // Check if it's a file path
  if (fs.existsSync(datasetPath)) {
    const data = JSON.parse(fs.readFileSync(datasetPath, "utf-8"));
    return Array.isArray(data) ? data : data.examples || [];
  }

  // Load from LangSmith
  try {
    const dataset = await client.readDataset({ datasetName: datasetPath });
    const examples = [];
    for await (const example of client.listExamples({ datasetId: dataset.id })) {
      examples.push({ inputs: example.inputs, outputs: example.outputs });
    }
    return examples;
  } catch (error) {
    throw new Error(`Could not find dataset '${datasetPath}': ${error.message}`);
  }
}

function createLLMJudgeEvaluator(model = "openai:gpt-4") {
  return createTrajectoryLLMAsJudge({
    model,
    prompt: TRAJECTORY_ACCURACY_PROMPT,
  });
}

async function main() {
  program
    .argument("<target-function>", "Target function path (e.g., ./my-agent.ts:runAgent)")
    .argument("<dataset>", "Dataset name (LangSmith) or file path (JSON)")
    .option("--method <type>", "Evaluation method", "llm-judge")
    .option("--model <model>", "LLM model for judge evaluation", "openai:gpt-4")
    .option("--experiment-prefix <prefix>", "Prefix for experiment name", "trajectory_eval")
    .option("--no-langsmith", "Run without LangSmith tracking")
    .parse();

  const options = program.opts();
  const [targetFunctionPath, datasetPath] = program.args;

  // Initialize LangSmith client
  const client = options.langsmith ? new Client() : null;

  if (client) {
    console.log("✅ Connected to LangSmith");
  }

  // Load target function
  console.log(`Loading target function: ${targetFunctionPath}`);
  const targetFunc = await loadTargetFunction(targetFunctionPath);

  // Load dataset
  console.log(`Loading dataset: ${datasetPath}`);
  const dataset = await loadDataset(datasetPath, client);
  console.log(`Loaded ${dataset.length} examples`);

  // Create evaluator
  const evaluator = createLLMJudgeEvaluator(options.model);
  console.log(`Using LLM-as-judge evaluator with ${options.model}`);

  // Run evaluation
  console.log("\nRunning evaluation...");

  const results = await evaluate(targetFunc, {
    data: dataset,
    evaluators: [evaluator],
    experimentPrefix: options.experimentPrefix,
  });

  console.log("\n" + "=".repeat(60));
  console.log("EVALUATION RESULTS");
  console.log("=".repeat(60));
  console.log(`\nExperiment: ${results.experimentName}`);
  console.log("View in LangSmith: https://smith.langchain.com");
  console.log("\n✅ Evaluation complete");
}

main().catch((error) => {
  console.error("Error:", error.message);
  process.exit(1);
});
