#!/usr/bin/env node
/**
 * Compare two agent implementations using A/B evaluation.
 *
 * Usage:
 *   node compare_agents.js <agent_a> <agent_b> <dataset> [options]
 *
 * Examples:
 *   node compare_agents.js ./agent-v1.ts:run ./agent-v2.ts:run test_dataset
 */

import { program } from "commander";
import * as fs from "fs";
import * as path from "path";
import { pathToFileURL } from "url";

async function loadAgentFunction(functionPath) {
  const [moduleFile, funcName] = functionPath.split(":");

  if (!moduleFile || !funcName) {
    throw new Error("Function path must be in format: ./file.ts:functionName");
  }

  const absolutePath = path.resolve(moduleFile);
  const module = await import(pathToFileURL(absolutePath).href);

  if (!(funcName in module)) {
    throw new Error(`Module '${moduleFile}' has no export '${funcName}'`);
  }

  return { func: module[funcName], name: functionPath };
}

async function loadDataset(datasetPath, client) {
  if (fs.existsSync(datasetPath)) {
    const data = JSON.parse(fs.readFileSync(datasetPath, "utf-8"));
    return Array.isArray(data) ? data : data.examples || [];
  }

  if (!client) {
    throw new Error(
      `Dataset '${datasetPath}' is not a file path. Remove --no-langsmith to load it from LangSmith.`
    );
  }

  const dataset = await client.readDataset({ datasetName: datasetPath });
  const examples = [];
  for await (const example of client.listExamples({ datasetId: dataset.id })) {
    examples.push({ inputs: example.inputs, outputs: example.outputs });
  }
  return examples;
}

async function evaluateAgent(agentFunc, dataset, agentName) {
  const results = {
    agent: agentName,
    examples: [],
    metrics: {
      totalExamples: dataset.length,
      successful: 0,
      failed: 0,
      latencies: [],
    },
  };

  console.log(`\nEvaluating ${agentName}...`);

  for (let i = 0; i < dataset.length; i++) {
    const example = dataset[i];
    process.stdout.write(`  Example ${i + 1}/${dataset.length}... `);
    const inputPayload = example.inputs ?? example.input;

    if (inputPayload === undefined) {
      results.metrics.failed++;
      results.examples.push({
        index: i,
        error: "Missing 'inputs' in dataset example",
        success: false,
      });
      console.log("✗ Error: Missing 'inputs' in dataset example");
      continue;
    }

    const exampleResult = {
      index: i,
      input: inputPayload,
      expectedOutput: example.outputs,
    };

    try {
      const startTime = Date.now();
      const output = await agentFunc(inputPayload);
      const latency = (Date.now() - startTime) / 1000;

      exampleResult.output = output;
      exampleResult.latency = latency;
      exampleResult.success = true;

      results.metrics.successful++;
      results.metrics.latencies.push(latency);

      console.log(`✓ (${latency.toFixed(2)}s)`);
    } catch (error) {
      exampleResult.error = error.message;
      exampleResult.success = false;
      results.metrics.failed++;

      console.log(`✗ Error: ${error.message}`);
    }

    results.examples.push(exampleResult);
  }

  // Calculate summary metrics
  if (results.metrics.latencies.length > 0) {
    const latencies = results.metrics.latencies;
    const sortedLatencies = [...latencies].sort((a, b) => a - b);
    results.metrics.avgLatency =
      latencies.reduce((a, b) => a + b, 0) / latencies.length;
    results.metrics.minLatency = Math.min(...latencies);
    results.metrics.maxLatency = Math.max(...latencies);
    results.metrics.p50Latency =
      sortedLatencies[Math.floor(sortedLatencies.length / 2)];
    results.metrics.p95Latency =
      sortedLatencies[Math.min(sortedLatencies.length - 1, Math.floor(sortedLatencies.length * 0.95))];
  }

  results.metrics.successRate =
    results.metrics.totalExamples > 0
      ? results.metrics.successful / results.metrics.totalExamples
      : 0;

  return results;
}

function compareResults(resultsA, resultsB) {
  const comparison = {
    agentA: resultsA.agent,
    agentB: resultsB.agent,
    metricsComparison: {},
  };

  const metricsA = resultsA.metrics;
  const metricsB = resultsB.metrics;

  // Compare success rate
  comparison.metricsComparison.successRate = {
    agentA: metricsA.successRate,
    agentB: metricsB.successRate,
    difference: metricsB.successRate - metricsA.successRate,
    winner:
      metricsB.successRate > metricsA.successRate
        ? resultsB.agent
        : metricsA.successRate > metricsB.successRate
          ? resultsA.agent
          : "tie",
  };

  // Compare latency
  if (metricsA.avgLatency && metricsB.avgLatency) {
    comparison.metricsComparison.avgLatency = {
      agentA: metricsA.avgLatency,
      agentB: metricsB.avgLatency,
      difference: metricsB.avgLatency - metricsA.avgLatency,
      winner:
        metricsA.avgLatency < metricsB.avgLatency
          ? resultsA.agent
          : metricsB.avgLatency < metricsA.avgLatency
            ? resultsB.agent
            : "tie",
    };
  }

  // Determine overall winner
  const successWinner = comparison.metricsComparison.successRate.winner;
  const latencyWinner = comparison.metricsComparison.avgLatency?.winner || "tie";

  if (successWinner === latencyWinner) {
    comparison.winner = successWinner;
  } else if (successWinner === "tie") {
    comparison.winner = latencyWinner;
  } else {
    comparison.winner = successWinner;
  }

  return comparison;
}

function printComparisonReport(comparison) {
  console.log("\n" + "=".repeat(70));
  console.log("AGENT COMPARISON REPORT");
  console.log("=".repeat(70));

  console.log(`\nAgent A: ${comparison.agentA}`);
  console.log(`Agent B: ${comparison.agentB}`);

  console.log("\n" + "-".repeat(70));
  console.log("METRICS COMPARISON");
  console.log("-".repeat(70));

  for (const [metric, data] of Object.entries(comparison.metricsComparison)) {
    console.log(`\n${metric.toUpperCase().replace(/_/g, " ")}:`);
    console.log(`  Agent A: ${data.agentA.toFixed(4)}`);
    console.log(`  Agent B: ${data.agentB.toFixed(4)}`);
    console.log(`  Difference: ${data.difference >= 0 ? "+" : ""}${data.difference.toFixed(4)}`);
    console.log(`  Winner: ${data.winner}`);
  }

  console.log("\n" + "=".repeat(70));
  console.log(`OVERALL WINNER: ${comparison.winner}`);
  console.log("=".repeat(70));
}

async function main() {
  program
    .argument("<agent-a>", "First agent function path")
    .argument("<agent-b>", "Second agent function path")
    .argument("<dataset>", "Dataset file path (JSON) or LangSmith dataset name")
    .option("-o, --output <file>", "Output file for detailed results (JSON)")
    .option("--no-langsmith", "Disable loading datasets from LangSmith")
    .parse();

  const options = program.opts();
  const [agentAPath, agentBPath, datasetPath] = program.args;
  let client = null;
  if (options.langsmith) {
    try {
      const { Client } = await import("langsmith");
      client = new Client();
      console.log("✅ Connected to LangSmith");
    } catch (error) {
      throw new Error(
        "LangSmith SDK is required for remote datasets. Install with: npm install langsmith"
      );
    }
  }

  // Load agents
  console.log("Loading agents...");
  const { func: agentAFunc, name: agentAName } = await loadAgentFunction(agentAPath);
  const { func: agentBFunc, name: agentBName } = await loadAgentFunction(agentBPath);

  // Load dataset
  console.log(`Loading dataset: ${datasetPath}`);
  const dataset = await loadDataset(datasetPath, client);
  console.log(`Loaded ${dataset.length} examples`);

  // Evaluate agents
  const resultsA = await evaluateAgent(agentAFunc, dataset, agentAName);
  const resultsB = await evaluateAgent(agentBFunc, dataset, agentBName);

  // Compare results
  const comparison = compareResults(resultsA, resultsB);

  // Print report
  printComparisonReport(comparison);

  // Save detailed results
  if (options.output) {
    const outputData = {
      comparison,
      agentAResults: resultsA,
      agentBResults: resultsB,
    };

    const outputPath = path.resolve(options.output);
    const outputDir = path.dirname(outputPath);
    if (!fs.existsSync(outputDir)) {
      fs.mkdirSync(outputDir, { recursive: true });
    }
    fs.writeFileSync(outputPath, JSON.stringify(outputData, null, 2));
    console.log(`\n✅ Detailed results saved to: ${outputPath}`);
  }
}

main().catch((error) => {
  console.error("Error:", error.message);
  process.exit(1);
});
