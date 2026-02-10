#!/usr/bin/env node
/**
 * Run evaluations using LangSmith's evaluation framework.
 *
 * Usage:
 *   node evaluate_with_langsmith.js <target_function> <dataset> [options]
 *
 * Examples:
 *   node evaluate_with_langsmith.js ./my-agent.ts:runAgent my_dataset
 */

import { program } from "commander";
import { Client } from "langsmith";
import { evaluate } from "langsmith/evaluation";
import * as fs from "fs";
import * as path from "path";
import { pathToFileURL } from "url";

async function loadTargetFunction(functionPath) {
  const [moduleFile, funcName] = functionPath.split(":");

  if (!moduleFile || !funcName) {
    throw new Error("Function path must be in format: ./file.ts:functionName");
  }

  const absolutePath = path.resolve(moduleFile);
  const module = await import(pathToFileURL(absolutePath).href);

  if (!(funcName in module)) {
    throw new Error(`Module '${moduleFile}' has no export '${funcName}'`);
  }

  return module[funcName];
}

async function createDatasetFromFile(client, datasetName, examplesFile) {
  const data = JSON.parse(fs.readFileSync(examplesFile, "utf-8"));
  const examples = Array.isArray(data) ? data : data.examples || [];

  console.log(`Creating dataset '${datasetName}' with ${examples.length} examples...`);

  const dataset = await client.createDataset(datasetName, {
    description: `Created from ${path.basename(examplesFile)}`,
  });

  const inputs = examples.map((example) => example.inputs || example.input || {});
  const outputs = examples.map((example) => example.outputs || example.output || {});
  await client.createExamples({ datasetId: dataset.id, inputs, outputs });

  console.log(`✅ Dataset created: ${datasetName}`);
  return datasetName;
}

function normalizeOutputs(value) {
  if (value && typeof value === "object") {
    for (const key of ["output", "answer", "response", "messages"]) {
      if (key in value) {
        return value[key];
      }
    }
  }
  return value;
}

function toComparable(value) {
  if (value === null || value === undefined) return value;
  if (typeof value === "string" || typeof value === "number" || typeof value === "boolean") {
    return value;
  }
  return JSON.stringify(value);
}

function normalizeEvaluatorArgs(args) {
  // Supports both:
  // 1) (run, example)
  // 2) ({ outputs, referenceOutputs })
  if (args.length === 1 && args[0] && typeof args[0] === "object" && "outputs" in args[0]) {
    return {
      actual: args[0].outputs,
      expected: args[0].referenceOutputs,
      run: null,
    };
  }
  if (args.length >= 2) {
    const [run, example] = args;
    return {
      actual: run?.outputs,
      expected: example?.outputs,
      run,
    };
  }
  return { actual: null, expected: null, run: null };
}

function createCustomEvaluator(evaluatorName) {
  if (evaluatorName === "accuracy") {
    return (...args) => {
      try {
        const { actual, expected } = normalizeEvaluatorArgs(args);
        const actualValue = toComparable(normalizeOutputs(actual));
        const expectedValue = toComparable(normalizeOutputs(expected));

        const score = actualValue === expectedValue ? 1.0 : 0.0;

        return { key: "accuracy", score };
      } catch (error) {
        return { key: "accuracy", score: 0.0, comment: error.message };
      }
    };
  }

  if (evaluatorName === "latency") {
    return (...args) => {
      try {
        const { run } = normalizeEvaluatorArgs(args);
        if (run.endTime && run.startTime) {
          const endMs =
            run.endTime instanceof Date ? run.endTime.getTime() : new Date(run.endTime).getTime();
          const startMs =
            run.startTime instanceof Date
              ? run.startTime.getTime()
              : new Date(run.startTime).getTime();
          const latency = (endMs - startMs) / 1000;

          if (!Number.isFinite(latency)) {
            return { key: "latency", score: 0.0, comment: "Invalid timing data" };
          }

          let score;
          if (latency < 1.0) {
            score = 1.0;
          } else if (latency < 5.0) {
            score = 0.5;
          } else {
            score = 0.0;
          }

          return {
            key: "latency",
            score,
            comment: `${latency.toFixed(2)}s`,
          };
        }

        return { key: "latency", score: 0.5, comment: "No timing data" };
      } catch (error) {
        return { key: "latency", score: 0.0, comment: error.message };
      }
    };
  }

  console.warn(`Warning: Unknown evaluator '${evaluatorName}'`);
  return null;
}

async function main() {
  program
    .argument("<target-function>", "Target function path")
    .argument("[dataset]", "Dataset name in LangSmith")
    .option("--create-dataset <file>", "Create dataset from examples file (JSON)")
    .option("--dataset-name <name>", "Name for newly created dataset")
    .option("--evaluators <list>", "Comma-separated list of evaluators", "accuracy")
    .option(
      "--experiment-prefix <prefix>",
      "Prefix for experiment name",
      "evaluation"
    )
    .option("--metadata <json>", "JSON string with experiment metadata")
    .option("--max-concurrency <n>", "Max concurrent evaluation workers", "4")
    .parse();

  const options = program.opts();
  const [targetFunctionPath, datasetArg] = program.args;

  // Initialize LangSmith client
  const client = new Client();
  console.log("✅ Connected to LangSmith");

  // Handle dataset creation
  let datasetName = datasetArg;
  if (options.createDataset) {
    if (!options.datasetName) {
      console.error("Error: --dataset-name required when using --create-dataset");
      process.exit(1);
    }

    datasetName = await createDatasetFromFile(
      client,
      options.datasetName,
      options.createDataset
    );
  }

  if (!datasetName) {
    console.error("Error: dataset name or --create-dataset required");
    process.exit(1);
  }

  // Load target function
  console.log(`\nLoading target function: ${targetFunctionPath}`);
  const targetFunc = await loadTargetFunction(targetFunctionPath);

  // Create evaluators
  const evaluatorNames = options.evaluators.split(",").map((e) => e.trim());
  const evaluators = evaluatorNames
    .map(createCustomEvaluator)
    .filter((e) => e !== null);

  if (evaluators.length === 0) {
    console.error("Error: No valid evaluators specified");
    process.exit(1);
  }

  console.log(`Using evaluators: ${evaluatorNames.join(", ")}`);

  // Parse metadata
  let metadata = {};
  if (options.metadata) {
    try {
      metadata = JSON.parse(options.metadata);
    } catch (error) {
      console.error("Error: Invalid JSON in --metadata");
      process.exit(1);
    }
  }

  // Run evaluation
  console.log("\nRunning evaluation...");
  const maxConcurrency = Number.parseInt(options.maxConcurrency, 10);
  if (!Number.isFinite(maxConcurrency) || maxConcurrency <= 0) {
    console.error("Error: --max-concurrency must be a positive integer");
    process.exit(1);
  }

  const results = await evaluate(targetFunc, {
    data: datasetName,
    evaluators,
    experimentPrefix: options.experimentPrefix,
    metadata,
    maxConcurrency,
  });

  console.log("\n" + "=".repeat(70));
  console.log("EVALUATION RESULTS");
  console.log("=".repeat(70));
  console.log(`\nExperiment: ${results.experimentName}`);
  console.log("🔗 View detailed results in LangSmith:");
  console.log("   https://smith.langchain.com");
  console.log("\n✅ Evaluation complete");
}

main().catch((error) => {
  console.error("Error:", error.message);
  process.exit(1);
});
