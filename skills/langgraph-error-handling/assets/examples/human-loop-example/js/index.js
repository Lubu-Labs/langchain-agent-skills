/**
 * LangGraph human-in-the-loop approval example.
 *
 * Demonstrates:
 * - interrupt() for human approval before sensitive operations
 * - new Command({ resume: ... }) for approve/reject flow
 * - Checkpointed pause/resume with thread_id
 */

import {
  StateGraph,
  StateSchema,
  MessagesValue,
  START,
  END,
  MemorySaver,
  Command,
  interrupt,
} from "@langchain/langgraph";
import { AIMessage, HumanMessage } from "@langchain/core/messages";
import { z } from "zod/v4";

// --- State ---
const State = new StateSchema({
  messages: MessagesValue,
  action: z.string().default(""),
  actionArgs: z.record(z.string(), z.unknown()).default({}),
  result: z.string().default(""),
});

// --- Nodes ---
async function planAction(state) {
  // Simulate planning; in practice this would be an LLM call.
  const latestUserMessage = String(state.messages.at(-1)?.content ?? "");
  if (latestUserMessage.toLowerCase().includes("inactive")) {
    return {
      action: "delete_records",
      actionArgs: { table: "users", filter: "inactive > 90 days" },
    };
  }
  return {
    action: "generate_report",
    actionArgs: { topic: "account-cleanup" },
  };
}

async function humanReview(state) {
  // Pause for human approval before executing sensitive action
  const isApproved = interrupt({
    question: "Do you want to proceed with this action?",
    action: state.action,
    args: state.actionArgs,
  });

  if (isApproved) {
    return new Command({ goto: "execute" });
  } else {
    return new Command({ goto: "cancel" });
  }
}

async function execute(state) {
  // Execute the approved action
  return {
    result: `Executed ${state.action} with ${JSON.stringify(state.actionArgs)}`,
    messages: [new AIMessage({ content: `Action completed: ${state.action}` })],
  };
}

async function cancel(state) {
  // Handle rejected action
  return {
    result: "cancelled",
    messages: [new AIMessage({ content: "Action was rejected by reviewer." })],
  };
}

// --- Graph ---
const builder = new StateGraph(State)
  .addNode("plan_action", planAction)
  .addNode("human_review", humanReview, { ends: ["execute", "cancel"] })
  .addNode("execute", execute)
  .addNode("cancel", cancel)
  .addEdge(START, "plan_action")
  .addEdge("plan_action", "human_review")
  // human_review uses Command for routing
  .addEdge("execute", END)
  .addEdge("cancel", END);

// A checkpointer is required for interrupt() pause/resume flows.
const graph = builder.compile({ checkpointer: new MemorySaver() });

// --- Usage ---
async function main() {
  const config = { configurable: { thread_id: "review-1" } };

  // Step 1: Run until interrupt
  const result = await graph.invoke(
    { messages: [new HumanMessage({ content: "Clean up inactive users" })] },
    config
  );
  console.log("Interrupted:", result.__interrupt__);

  // Step 2: Resume with approval
  const finalResult = await graph.invoke(
    new Command({ resume: true }),
    config
  );
  for (const msg of finalResult.messages) {
    console.log(`${msg.constructor.name}: ${msg.content}`);
  }
}

main();
