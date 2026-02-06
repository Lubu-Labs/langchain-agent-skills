#!/usr/bin/env node
/**
 * Deploy LangGraph applications using the LangSmith Deployment control plane API.
 *
 * Default control-plane endpoints (from official docs):
 * - US: https://api.host.langchain.com
 * - EU: https://eu.api.host.langchain.com
 *
 * For self-hosted control planes, pass --control-plane-url with your /api-host URL.
 */

import * as fs from 'fs';

const DEFAULT_CONTROL_PLANE_US = 'https://api.host.langchain.com';
const DEFAULT_CONTROL_PLANE_EU = 'https://eu.api.host.langchain.com';
const DEFAULT_CLOUD_UI_URL = 'https://smith.langchain.com';

interface DeploymentConfig {
  display_name: string;
  config: {
    git_repo: {
      url: string;
      branch: string;
      api_config_relative_path: string;
    };
    env_vars: Record<string, string>;
    shareable: boolean;
  };
}

class LangSmithDeployer {
  private apiKey: string;
  private tenantId?: string;
  private baseUrl: string;

  constructor(apiKey?: string, controlPlaneUrl?: string, tenantId?: string) {
    this.apiKey = apiKey || process.env.LANGSMITH_API_KEY || '';
    if (!this.apiKey) {
      throw new Error('LANGSMITH_API_KEY not found. Set it via environment variable or --api-key');
    }

    this.baseUrl = (controlPlaneUrl || DEFAULT_CONTROL_PLANE_US).replace(/\/$/, '');
    this.tenantId = tenantId || process.env.LANGSMITH_TENANT_ID;
  }

  private get headers(): Record<string, string> {
    const headers: Record<string, string> = {
      'X-Api-Key': this.apiKey,
      'Content-Type': 'application/json',
    };

    if (this.tenantId) {
      headers['X-Tenant-Id'] = this.tenantId;
    }

    return headers;
  }

  private async parseResponse(response: Response): Promise<any> {
    const contentType = response.headers.get('content-type') || '';
    if (contentType.includes('application/json')) {
      return response.json();
    }
    return response.text();
  }

  private async extractError(response: Response): Promise<string> {
    const payload = await this.parseResponse(response);
    if (payload && typeof payload === 'object') {
      if ('detail' in payload && payload.detail) {
        return String(payload.detail);
      }
      if ('message' in payload && payload.message) {
        return String(payload.message);
      }
    }
    return typeof payload === 'string' ? payload : `HTTP ${response.status}`;
  }

  async validateLanggraphConfig(configPath: string): Promise<any> {
    if (!fs.existsSync(configPath)) {
      throw new Error(`Configuration file not found: ${configPath}`);
    }

    const content = fs.readFileSync(configPath, 'utf-8');
    const config = JSON.parse(content);

    if (!('graphs' in config)) {
      throw new Error(`Missing required fields in langgraph.json: graphs`);
    }

    const isJsProject = 'node_version' in config;
    if (!isJsProject && !('dependencies' in config)) {
      throw new Error(
        `Missing required fields in langgraph.json: dependencies (required for Python projects)`
      );
    }

    const graphCount = config.graphs && typeof config.graphs === 'object'
      ? Object.keys(config.graphs).length
      : 0;

    console.log(`✓ Configuration valid: ${graphCount} graph(s) defined`);
    return config;
  }

  async checkGithubIntegration(owner: string, repo: string): Promise<boolean> {
    const url = `${this.baseUrl}/api/v1/integrations/github/repos`;

    const response = await fetch(url, { headers: this.headers });
    if (!response.ok) {
      console.log(`Warning: Could not check GitHub integration: ${response.status}`);
      return false;
    }

    const payload = await this.parseResponse(response);
    if (!Array.isArray(payload)) {
      console.log('Warning: Unexpected integrations response format');
      return false;
    }

    const repoFullName = `${owner}/${repo}`;
    const integrated = payload.some((r: any) => r?.full_name === repoFullName);

    if (integrated) {
      console.log(`✓ GitHub repository ${repoFullName} is integrated`);
    } else {
      console.log(`✗ GitHub repository ${repoFullName} is not integrated`);
      console.log('  Add it in the LangSmith UI under GitHub integrations.');
    }

    return integrated;
  }

  async createDeployment(
    name: string,
    owner: string,
    repo: string,
    branch: string,
    configPath: string,
    envVars: Record<string, string> = {},
    shareable: boolean = false
  ): Promise<any> {
    const url = `${this.baseUrl}/api/v2/deployments`;

    const payload: DeploymentConfig = {
      display_name: name,
      config: {
        git_repo: {
          url: `https://github.com/${owner}/${repo}`,
          branch,
          api_config_relative_path: configPath,
        },
        env_vars: envVars,
        shareable,
      },
    };

    console.log(`\nCreating deployment '${name}'...`);
    console.log(`  Repository: ${owner}/${repo}`);
    console.log(`  Branch: ${branch}`);
    console.log(`  Config: ${configPath}`);

    const response = await fetch(url, {
      method: 'POST',
      headers: this.headers,
      body: JSON.stringify(payload),
    });

    if (response.status === 201) {
      const deployment = await this.parseResponse(response);
      console.log('\n✓ Deployment created successfully');
      console.log(`  Deployment ID: ${deployment?.id ?? 'unknown'}`);
      return deployment;
    }

    throw new Error(`Deployment failed: ${await this.extractError(response)}`);
  }

  async createRevision(
    deploymentId: string,
    branch?: string,
    configPath?: string,
    envVars?: Record<string, string>
  ): Promise<any> {
    const url = `${this.baseUrl}/api/v2/deployments/${deploymentId}/revisions`;

    const payload: Record<string, unknown> = {};
    if (branch) payload.git_branch = branch;
    if (configPath) payload.api_config_relative_path = configPath;
    if (envVars) payload.env_vars = envVars;

    console.log(`\nCreating revision for deployment ${deploymentId}...`);
    const response = await fetch(url, {
      method: 'POST',
      headers: this.headers,
      body: JSON.stringify(payload),
    });

    if (response.status === 201) {
      const revision = await this.parseResponse(response);
      console.log('✓ Revision created successfully');
      console.log(`  Revision ID: ${revision?.id ?? 'unknown'}`);
      return revision;
    }

    throw new Error(`Revision creation failed: ${await this.extractError(response)}`);
  }
}

function loadEnvFile(envFilePath: string): Record<string, string> {
  const envVars: Record<string, string> = {};

  if (!fs.existsSync(envFilePath)) {
    return envVars;
  }

  const content = fs.readFileSync(envFilePath, 'utf-8');
  const lines = content.split('\n');

  for (const line of lines) {
    const trimmed = line.trim();
    if (!trimmed || trimmed.startsWith('#') || !trimmed.includes('=')) {
      continue;
    }

    const [key, ...valueParts] = trimmed.split('=');
    let value = valueParts.join('=').trim();
    if ((value.startsWith('"') && value.endsWith('"')) || (value.startsWith("'") && value.endsWith("'"))) {
      value = value.slice(1, -1);
    }
    envVars[key.trim()] = value;
  }

  return envVars;
}

function getArg(args: string[], flag: string, defaultValue?: string): string | undefined {
  const index = args.indexOf(flag);
  return index !== -1 && args[index + 1] ? args[index + 1] : defaultValue;
}

function hasFlag(args: string[], flag: string): boolean {
  return args.includes(flag);
}

function resolveControlPlaneUrl(region: string, explicitUrl?: string): string {
  if (explicitUrl) return explicitUrl;
  if (process.env.LANGSMITH_CONTROL_PLANE_URL) return process.env.LANGSMITH_CONTROL_PLANE_URL;
  return region === 'eu' ? DEFAULT_CONTROL_PLANE_EU : DEFAULT_CONTROL_PLANE_US;
}

async function main() {
  const args = process.argv.slice(2);

  if (hasFlag(args, '--help') || hasFlag(args, '-h')) {
    console.log(`
Usage: deploy_to_langsmith.ts [options]

Options:
  --name <name>               Deployment name (required)
  --owner <owner>             GitHub owner (required)
  --repo <repo>               GitHub repository name (required)
  --branch <branch>           Git branch to deploy (default: main)
  --config <path>             Path to langgraph.json in repo (default: langgraph.json)
  --env-file <path>           Path to .env file with environment variables
  --env <KEY=VALUE>           Environment variable (can be used multiple times)
  --shareable                 Make deployment shareable in Studio
  --api-key <key>             LangSmith API key (or use LANGSMITH_API_KEY)
  --tenant-id <id>            Workspace ID for org-scoped API keys
  --region <us|eu>            Region for default control-plane host (default: us)
  --control-plane-url <url>   Override control-plane URL (self-hosted: https://<host>/api-host)
  --deployment-id <id>        Existing deployment ID (creates new revision)
  --validate-only             Only validate langgraph.json, don't deploy
  --skip-github-check         Skip GitHub integration check
  --ui-url <url>              Base UI URL for printed deployment link
  --help, -h                  Show this help message
`);
    process.exit(0);
  }

  const name = getArg(args, '--name');
  const owner = getArg(args, '--owner');
  const repo = getArg(args, '--repo');
  const branch = getArg(args, '--branch', 'main')!;
  const config = getArg(args, '--config', 'langgraph.json')!;
  const envFile = getArg(args, '--env-file');
  const apiKey = getArg(args, '--api-key');
  const tenantId = getArg(args, '--tenant-id');
  const deploymentId = getArg(args, '--deployment-id');
  const validateOnly = hasFlag(args, '--validate-only');
  const shareable = hasFlag(args, '--shareable');
  const skipGithubCheck = hasFlag(args, '--skip-github-check');
  const region = getArg(args, '--region', 'us')!;
  const controlPlaneUrl = resolveControlPlaneUrl(region, getArg(args, '--control-plane-url'));
  const uiUrl = (getArg(args, '--ui-url', DEFAULT_CLOUD_UI_URL) || DEFAULT_CLOUD_UI_URL).replace(/\/$/, '');

  if (!name || !owner || !repo) {
    console.error('Error: --name, --owner, and --repo are required');
    process.exit(1);
  }

  try {
    const deployer = new LangSmithDeployer(apiKey, controlPlaneUrl, tenantId);

    if (fs.existsSync(config)) {
      await deployer.validateLanggraphConfig(config);
      if (validateOnly) {
        console.log('\n✓ Validation complete');
        process.exit(0);
      }
    } else {
      console.log(`Note: Local ${config} not found. Using repository path at deploy time.`);
      if (validateOnly) {
        console.log('Warning: Cannot validate config file that does not exist locally.');
        process.exit(1);
      }
    }

    if (!skipGithubCheck) {
      await deployer.checkGithubIntegration(owner, repo);
    }

    let envVars: Record<string, string> = {};
    if (envFile) {
      envVars = loadEnvFile(envFile);
    }

    for (let i = 0; i < args.length; i++) {
      if (args[i] === '--env' && args[i + 1]) {
        const envPair = args[i + 1];
        if (!envPair.includes('=')) {
          console.log(`Warning: Ignoring invalid --env value '${envPair}' (expected KEY=VALUE)`);
          continue;
        }
        const [key, ...valueParts] = envPair.split('=');
        envVars[key.trim()] = valueParts.join('=').trim();
      }
    }

    if (deploymentId) {
      await deployer.createRevision(
        deploymentId,
        branch,
        config,
        Object.keys(envVars).length > 0 ? envVars : undefined
      );
    } else {
      const deployment = await deployer.createDeployment(
        name,
        owner,
        repo,
        branch,
        config,
        envVars,
        shareable
      );

      if (deployment?.id) {
        console.log('\n' + '='.repeat(60));
        console.log('Deployment initiated. Monitor progress at:');
        console.log(`${uiUrl}/deployments/${deployment.id}`);
        console.log('='.repeat(60));
      }
    }

    process.exit(0);
  } catch (error) {
    console.error(`\n✗ Error: ${error}`);
    process.exit(1);
  }
}

main();
