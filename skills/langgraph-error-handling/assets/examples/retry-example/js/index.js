/**
 * LangGraph retry example with RetryPolicy and state-based error recovery.
 *
 * Demonstrates:
 * - RetryPolicy for transient errors (API calls)
 * - Recovery loop with Command routing from the agent node
 * - Retry count tracking in graph state
 */

import {
  StateGraph,
  StateSchema,
  MessagesValue,
  START,
  END,
  MemorySaver,
  Command,
} from "@langchain/langgraph";
import { AIMessage, HumanMessage } from "@langchain/core/messages";
import { z } from "zod/v4";

// --- State ---
const State = new StateSchema({
  messages: MessagesValue,
  searchResult: z.string().default(""),
  error: z.string().default(""),
  retryCount: z.number().default(0),
});

// --- Nodes ---
async function agent(state) {
  const error = state.error || "";
  const retryCount = state.retryCount || 0;

  if (retryCount >= 3) {
    return new Command({
      update: {
        messages: [
          new AIMessage({ content: `Search failed after 3 retries: ${error}` }),
        ],
      },
      goto: END,
    });
  }

  if (error) {
    // In a real app this node would call an LLM to adapt strategy.
    return new Command({
      update: {
        messages: [
          new AIMessage({ content: `Search failed (${error}), retrying with different query...` }),
        ],
      },
      goto: "search",
    });
  }

  if (state.searchResult) {
    return new Command({
      update: {
        messages: [new AIMessage({ content: `Found: ${state.searchResult}` })],
      },
      goto: END,
    });
  }

  return new Command({ goto: "search" });
}

async function search(state) {
  try {
    // Simulate search; replace with actual API call.
    const lastMessage = state.messages.at(-1);
    const query = typeof lastMessage?.content === "string" ? lastMessage.content : "default";
    const result = `Results for: ${query}`;
    return { searchResult: result, error: "" };
  } catch (e) {
    // Non-transient errors are surfaced in state for recovery decisions.
    return {
      error: String(e),
      retryCount: (state.retryCount || 0) + 1,
      searchResult: "",
    };
  }
}

// --- Graph ---
const builder = new StateGraph(State)
  .addNode("agent", agent, { ends: ["search", END] })
  .addNode("search", search, {
    // RetryPolicy handles transient errors (network, rate limits) automatically
    retryPolicy: { maxAttempts: 3, initialInterval: 1.0 },
  })
  .addEdge(START, "agent")
  .addEdge("search", "agent");
// `agent` uses Command for dynamic routing.

const graph = builder.compile({ checkpointer: new MemorySaver() });

// --- Usage ---
async function main() {
  const result = await graph.invoke(
    { messages: [new HumanMessage({ content: "Search for LangGraph tutorials" })] },
    { configurable: { thread_id: "demo-1" } }
  );
  for (const msg of result.messages) {
    console.log(`${msg.constructor.name}: ${msg.content}`);
  }
}

main();
