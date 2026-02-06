#!/usr/bin/env ts-node
/**
 * Download traces from LangSmith with flexible filtering and organization.
 *
 * Features:
 * - Query by trace ID, job_id, metadata, status, or time range
 * - Organize by outcome (passed/failed/error) automatically
 * - Create manifest with trace metadata
 * - Resume interrupted downloads
 *
 * Usage:
 *   # Download specific traces
 *   ts-node download_traces.ts --trace-ids id1 id2 id3 --output ./traces
 *
 *   # Download by metadata filter
 *   ts-node download_traces.ts --project my-project --filter 'job_id=abc123' --output ./traces
 *
 *   # Download recent traces
 *   ts-node download_traces.ts --project my-project --last-hours 24 --limit 50 --output ./traces
 */

import { Client } from "langsmith";
import { spawn } from "child_process";
import * as fs from "fs/promises";
import * as path from "path";

interface TraceInfo {
  trace_id: string;
  status?: string;
  metadata?: Record<string, any>;
  start_time?: string;
  end_time?: string;
}

interface DownloadResults {
  success: string[];
  failed: Array<{ trace_id: string; error: string }>;
  by_category: Record<string, number>;
}

async function ensureDirectory(dir: string): Promise<void> {
  await fs.mkdir(dir, { recursive: true });
}

async function queryTraces(
  projectName: string,
  filterQuery?: string,
  status?: string,
  sinceHours?: number,
  limit?: number
): Promise<TraceInfo[]> {
  const client = new Client();

  const filters: string[] = [];
  if (filterQuery) filters.push(filterQuery);
  if (status) filters.push(`eq(status, "${status}")`);

  const filterStr = filters.length > 1 ? `and(${filters.join(", ")})` : filters[0];

  const opts: any = {
    projectName,
    isRoot: true,
  };

  if (filterStr) opts.filter = filterStr;
  if (sinceHours !== undefined && sinceHours !== null) {
    const startTime = new Date(Date.now() - sinceHours * 60 * 60 * 1000);
    opts.startTime = startTime;
  }

  const traces: TraceInfo[] = [];
  for await (const run of client.listRuns(opts)) {
    traces.push({
      trace_id: String(run.id),
      status: run.status,
      metadata: run.metadata || {},
      start_time: run.start_time?.toISOString(),
      end_time: run.end_time?.toISOString(),
    });

    if (limit && traces.length >= limit) break;
  }

  return traces;
}

async function fetchSingleTrace(
  traceId: string,
  outputPath: string,
  retries = 3
): Promise<[boolean, string | null]> {
  await ensureDirectory(path.dirname(outputPath));
  let lastError: string | null = null;
  const primaryArgs = [
    "trace",
    traceId,
    "--format",
    "raw",
    "--include-metadata",
    "--file",
    outputPath,
  ];
  const fallbackArgs = [
    "trace",
    traceId,
    "--format",
    "raw",
    "--file",
    outputPath,
  ];

  const runFetch = async (args: string[]): Promise<void> =>
    new Promise<void>((resolve, reject) => {
      const proc = spawn("langsmith-fetch", args);

      let stderr = "";
      proc.stderr?.on("data", (data) => {
        stderr += data.toString();
      });

      const timer = setTimeout(() => {
        proc.kill();
        reject(new Error("Timeout"));
      }, 60000);

      proc.on("close", (code) => {
        clearTimeout(timer);
        if (code === 0) resolve();
        else reject(new Error(stderr || "Unknown error"));
      });
    });

  for (let attempt = 0; attempt < retries; attempt++) {
    try {
      try {
        await runFetch(primaryArgs);
      } catch (error: any) {
        const message = error?.message || "";
        const includeFlagUnsupported =
          message.includes("include-metadata") &&
          (message.includes("No such option") || message.toLowerCase().includes("unknown option"));

        if (!includeFlagUnsupported) {
          throw error;
        }

        // Backward/forward compatibility across langsmith-fetch CLI versions.
        await runFetch(fallbackArgs);
      }

      // Verify JSON
      const content = await fs.readFile(outputPath, "utf-8");
      JSON.parse(content);

      return [true, null];
    } catch (error: any) {
      lastError = error?.message || String(error) || "Unknown error";
      if (attempt < retries - 1) {
        await new Promise((resolve) => setTimeout(resolve, 2 ** attempt * 1000));
      }
    }
  }

  return [false, lastError || "Failed after retries"];
}

function extractTraceMetadata(traceData: any): Record<string, any> {
  if (traceData?.metadata && typeof traceData.metadata === "object") return traceData.metadata;
  if (traceData?.extra?.metadata && typeof traceData.extra.metadata === "object") return traceData.extra.metadata;
  return {};
}

function extractTraceMessages(traceData: any): Array<Record<string, any>> {
  if (Array.isArray(traceData?.messages)) return traceData.messages.filter((m: any) => m && typeof m === "object");
  if (Array.isArray(traceData?.inputs?.messages)) {
    return traceData.inputs.messages.filter((m: any) => m && typeof m === "object");
  }
  return [];
}

function categorizeTrace(traceData: any): [string, string | null] {
  const metadata = extractTraceMetadata(traceData);
  const status = String(traceData?.status || metadata?.status || (traceData?.error ? "error" : "unknown"));

  if (status === "success") {
    const customMeta = metadata.custom_metadata || {};
    const reward = customMeta.reward ?? metadata.reward;
    if (reward === 1 || reward === 1.0 || reward === true) return ["passed", null];
    if (reward === 0 || reward === 0.0 || reward === false) return ["failed", null];
    return ["passed", null];
  } else if (status === "error") {
    const messages = extractTraceMessages(traceData);
    const errorBlob = String(traceData?.error || "");
    let errorType = "unknown";
    if (errorBlob.includes("GraphRecursionError")) return ["error", "GraphRecursionError"];
    if (errorBlob.includes("TimeoutError") || errorBlob.includes("AgentTimeoutError")) return ["error", "TimeoutError"];
    if (errorBlob.includes("DaytonaError")) return ["error", "DaytonaError"];

    for (let i = messages.length - 1; i >= 0; i--) {
      const content = String(messages[i].content || "");
      if (content.includes("GraphRecursionError")) {
        errorType = "GraphRecursionError";
        break;
      } else if (content.includes("TimeoutError") || content.includes("AgentTimeoutError")) {
        errorType = "TimeoutError";
        break;
      } else if (content.includes("DaytonaError")) {
        errorType = "DaytonaError";
        break;
      }
    }

    return ["error", errorType];
  }

  return ["unknown", null];
}

async function fileExists(filePath: string): Promise<boolean> {
  try {
    await fs.access(filePath);
    return true;
  } catch {
    return false;
  }
}

async function findExistingTracePath(outputDir: string, traceId: string): Promise<string | null> {
  const directCandidates = [
    path.join(outputDir, "temp", `${traceId}.json`),
    path.join(outputDir, `${traceId}.json`),
  ];

  for (const candidate of directCandidates) {
    if (await fileExists(candidate)) return candidate;
  }

  const byOutcomeDir = path.join(outputDir, "by-outcome");
  if (!(await fileExists(byOutcomeDir))) return null;

  const categories = await fs.readdir(byOutcomeDir, { withFileTypes: true });
  for (const category of categories) {
    if (!category.isDirectory()) continue;
    const categoryDir = path.join(byOutcomeDir, category.name);
    const directFile = path.join(categoryDir, `${traceId}.json`);
    if (await fileExists(directFile)) return directFile;

    const subEntries = await fs.readdir(categoryDir, { withFileTypes: true });
    for (const subEntry of subEntries) {
      if (!subEntry.isDirectory()) continue;
      const subFile = path.join(categoryDir, subEntry.name, `${traceId}.json`);
      if (await fileExists(subFile)) return subFile;
    }
  }

  return null;
}

async function main() {
  const args = process.argv.slice(2);
  const config: any = {
    project: null,
    traceIds: [],
    filter: null,
    status: null,
    lastHours: null,
    limit: null,
    output: "./langsmith-traces",
    delay: 1000,
    organize: true,
  };

  if (args.includes("--help") || args.includes("-h")) {
    console.log(`Usage:
  ts-node download_traces.ts --trace-ids <id...> [--output <dir>]
  ts-node download_traces.ts --project <name> [--filter <query>] [--status success|error]
    [--last-hours <n>] [--limit <n>] [--output <dir>] [--delay <seconds>] [--no-organize]`);
    return 0;
  }

  // Parse command line args
  for (let i = 0; i < args.length; i++) {
    const arg = args[i];
    if (arg === "--project") config.project = args[++i];
    else if (arg === "--trace-ids") {
      while (i + 1 < args.length && !args[i + 1].startsWith("--")) {
        config.traceIds.push(args[++i]);
      }
    } else if (arg === "--filter") config.filter = args[++i];
    else if (arg === "--status") config.status = args[++i];
    else if (arg === "--last-hours") config.lastHours = parseInt(args[++i]);
    else if (arg === "--limit") config.limit = parseInt(args[++i]);
    else if (arg === "--output") config.output = args[++i];
    else if (arg === "--delay") config.delay = parseFloat(args[++i]) * 1000;
    else if (arg === "--organize") config.organize = true;
    else if (arg === "--no-organize") config.organize = false;
  }

  if (config.status && !["success", "error"].includes(config.status)) {
    console.error("Error: --status must be one of: success, error");
    process.exit(1);
  }

  await ensureDirectory(config.output);

  // Get traces
  let traces: TraceInfo[];
  if (config.traceIds.length > 0) {
    traces = config.traceIds.map((id: string) => ({ trace_id: id }));
    console.log(`Downloading ${traces.length} specified traces...`);
  } else {
    if (!config.project) {
      console.error("Error: --project required when not using --trace-ids");
      process.exit(1);
    }

    // Parse filter
    let filterQuery = config.filter;
    if (filterQuery && filterQuery.includes("=")) {
      const [key, value] = filterQuery.split("=", 2);
      filterQuery = `and(eq(metadata_key, "${key}"), eq(metadata_value, "${value}"))`;
    }

    console.log(`Querying traces from project: ${config.project}`);
    traces = await queryTraces(
      config.project,
      filterQuery,
      config.status,
      config.lastHours,
      config.limit
    );
    console.log(`Found ${traces.length} matching traces`);
  }

  if (traces.length === 0) {
    console.log("No traces to download");
    return 0;
  }

  const results: DownloadResults = {
    success: [],
    failed: [],
    by_category: {},
  };

  for (let i = 0; i < traces.length; i++) {
    const trace = traces[i];
    const traceId = trace.trace_id;

    // Determine output path
    const tempPath = config.organize
      ? path.join(config.output, "temp", `${traceId}.json`)
      : path.join(config.output, `${traceId}.json`);

    const existing = await findExistingTracePath(config.output, traceId);
    if (existing) {
      console.log(
        `[${i + 1}/${traces.length}] SKIP: ${traceId} (already exists at ${path.relative(config.output, existing)})`
      );
      results.success.push(traceId);
      continue;
    }

    console.log(`[${i + 1}/${traces.length}] Downloading: ${traceId}`);

    const [success, error] = await fetchSingleTrace(traceId, tempPath);

    if (success) {
      // Categorize and move if organizing
      if (config.organize) {
        const content = await fs.readFile(tempPath, "utf-8");
        const traceData = JSON.parse(content);

        const [category, subcategory] = categorizeTrace(traceData);
        results.by_category[category] = (results.by_category[category] || 0) + 1;

        const finalDir = subcategory
          ? path.join(config.output, "by-outcome", category, subcategory)
          : path.join(config.output, "by-outcome", category);

        await ensureDirectory(finalDir);
        const finalPath = path.join(finalDir, `${traceId}.json`);

        await fs.rename(tempPath, finalPath);
        console.log(`  -> ${path.relative(config.output, finalPath)}`);
      } else {
        console.log(`  -> ${path.relative(config.output, tempPath)}`);
      }

      results.success.push(traceId);
    } else {
      console.log(`  FAILED: ${error}`);
      results.failed.push({ trace_id: traceId, error: error || "Unknown" });
    }

    // Rate limiting
    if (i < traces.length - 1) {
      await new Promise((resolve) => setTimeout(resolve, config.delay));
    }
  }

  // Create manifest
  const manifest = {
    created_at: new Date().toISOString(),
    project: config.project,
    filter: config.filter,
    total_traces: traces.length,
    downloaded: results.success.length,
    failed: results.failed.length,
    by_category: results.by_category,
  };

  const manifestPath = path.join(config.output, "manifest.json");
  await fs.writeFile(manifestPath, JSON.stringify(manifest, null, 2));

  console.log("\n" + "=".repeat(60));
  console.log("DOWNLOAD COMPLETE");
  console.log("=".repeat(60));
  console.log(`Success: ${results.success.length}`);
  console.log(`Failed: ${results.failed.length}`);

  if (Object.keys(results.by_category).length > 0) {
    console.log("\nBy Category:");
    Object.entries(results.by_category)
      .sort()
      .forEach(([cat, count]) => {
        console.log(`  ${cat}: ${count}`);
      });
  }

  if (results.failed.length > 0) {
    console.log("\nFailed downloads:");
    results.failed.slice(0, 10).forEach((item) => {
      console.log(`  - ${item.trace_id}: ${item.error}`);
    });
  }

  console.log(`\nManifest written to: ${manifestPath}`);

  return results.failed.length > 0 ? 1 : 0;
}

main()
  .then((code) => process.exit(code))
  .catch((error) => {
    console.error(error instanceof Error ? error.message : String(error));
    process.exit(1);
  });
