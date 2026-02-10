#!/usr/bin/env node
/**
 * Mock LLM responses for deterministic testing.
 *
 * Usage:
 *   import { MockLLM, SequenceMockLLM } from './mock_llm_responses.js';
 */
import { program } from "commander";
import * as fs from "fs";

export class MockLLM {
  constructor(response = "Mocked response", options = {}) {
    this.response = response;
    this.model = options.model || "mock-model";
    this.toolCalls = options.toolCalls || [];
    this.callCount = 0;
    this.callHistory = [];
  }

  async invoke(messages, options = {}) {
    this.callCount++;
    this.callHistory.push({ messages, options });

    const response = {
      role: "assistant",
      content: this.response,
    };

    if (this.toolCalls.length > 0) {
      response.tool_calls = this.toolCalls;
    }

    return response;
  }

  reset() {
    this.callCount = 0;
    this.callHistory = [];
  }
}

export class SequenceMockLLM {
  constructor(responses, options = {}) {
    this.responses = responses;
    this.model = options.model || "mock-model";
    this.loop = options.loop || false;
    this.currentIndex = 0;
    this.callCount = 0;
    this.callHistory = [];
  }

  async invoke(messages, options = {}) {
    this.callCount++;
    this.callHistory.push({ messages, options });

    if (this.currentIndex >= this.responses.length) {
      if (this.loop) {
        this.currentIndex = 0;
      } else {
        throw new Error(
          `Mock LLM ran out of responses (called ${this.callCount} times, only ${this.responses.length} responses)`
        );
      }
    }

    const responseData = this.responses[this.currentIndex];
    this.currentIndex++;

    if (typeof responseData === "string") {
      return { role: "assistant", content: responseData };
    } else if (typeof responseData === "object") {
      return { role: "assistant", ...responseData };
    } else {
      return { role: "assistant", content: String(responseData) };
    }
  }

  reset() {
    this.currentIndex = 0;
    this.callCount = 0;
    this.callHistory = [];
  }
}

export class ConditionalMockLLM {
  constructor(responseMap, defaultResponse = "Default mock response", options = {}) {
    this.responseMap = responseMap;
    this.defaultResponse = defaultResponse;
    this.model = options.model || "mock-model";
    this.callCount = 0;
    this.callHistory = [];
  }

  async invoke(messages, options = {}) {
    this.callCount++;
    this.callHistory.push({ messages, options });

    // Extract last user message
    let userMessage = "";
    for (let i = messages.length - 1; i >= 0; i--) {
      if (messages[i].role === "user") {
        userMessage = messages[i].content || "";
        break;
      }
    }

    // Find matching response
    let responseData = this.defaultResponse;
    for (const [pattern, response] of Object.entries(this.responseMap)) {
      if (userMessage.toLowerCase().includes(pattern.toLowerCase())) {
        responseData = response;
        break;
      }
    }

    // Format response
    if (typeof responseData === "string") {
      return { role: "assistant", content: responseData };
    } else if (typeof responseData === "object") {
      return { role: "assistant", ...responseData };
    } else {
      return { role: "assistant", content: String(responseData) };
    }
  }

  reset() {
    this.callCount = 0;
    this.callHistory = [];
  }
}

export function createToolCallMock(toolName, toolArgs, content = "") {
  return {
    content,
    tool_calls: [
      {
        name: toolName,
        args: toolArgs,
        id: `call_${toolName}`,
      },
    ],
  };
}

export function loadMockConfig(configPath) {
  return JSON.parse(fs.readFileSync(configPath, "utf-8"));
}

export function createMockFromConfig(config) {
  const mockType = config.type || "single";

  if (mockType === "single") {
    return new MockLLM(config.response || "Mocked response", {
      toolCalls: config.toolCalls || config.tool_calls || [],
    });
  }

  if (mockType === "sequence") {
    return new SequenceMockLLM(config.responses || [], {
      loop: config.loop || false,
    });
  }

  if (mockType === "conditional") {
    return new ConditionalMockLLM(
      config.responseMap || config.response_map || {},
      config.defaultResponse || config.default_response || "Default mock response"
    );
  }

  throw new Error(`Unknown mock type: ${mockType}`);
}

// CLI functionality
if (import.meta.url === `file://${process.argv[1]}`) {
  program
    .command("create")
    .description("Create mock configuration")
    .option("-o, --output <file>", "Output file", "mock_config.json")
    .option(
      "-t, --type <type>",
      "Mock type (single, sequence, conditional)",
      "single"
    )
    .action((options) => {
      let config;

      if (options.type === "single") {
        config = {
          type: "single",
          response: "This is a mocked response",
          toolCalls: [],
        };
      } else if (options.type === "sequence") {
        config = {
          type: "sequence",
          responses: ["First response", "Second response", "Third response"],
          loop: false,
        };
      } else if (options.type === "conditional") {
        config = {
          type: "conditional",
          responseMap: {
            weather: "The weather is sunny today.",
            time: "It is currently 3:00 PM.",
          },
          defaultResponse: "I don't understand the question.",
        };
      }

      fs.writeFileSync(options.output, JSON.stringify(config, null, 2));
      console.log(`✅ Created mock configuration: ${options.output}`);
      console.log(`   Type: ${options.type}`);
    });

  program
    .command("validate")
    .description("Validate mock configuration")
    .argument("<config>", "Path to configuration JSON file")
    .action((configPath) => {
      try {
        const config = loadMockConfig(configPath);
        createMockFromConfig(config);
        console.log(`✅ Valid configuration: ${configPath}`);
        console.log(`   Type: ${config.type || "single"}`);
      } catch (error) {
        console.error(`❌ Invalid configuration: ${error.message}`);
        process.exit(1);
      }
    });

  program.parse();
}
