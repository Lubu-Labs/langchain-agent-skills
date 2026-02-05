import {
  Annotation,
  END,
  START,
  StateGraph,
  messagesStateReducer,
} from "@langchain/langgraph";
import { HumanMessage } from "@langchain/core/messages";

const SupervisorState = Annotation.Root({
  messages: Annotation({
    reducer: messagesStateReducer,
    default: () => [],
  }),
  next: Annotation({
    reducer: (_, right) => right,
    default: () => "researcher",
  }),
  currentAgent: Annotation({
    reducer: (_, right) => right,
    default: () => "",
  }),
  taskComplete: Annotation({
    reducer: (_, right) => right,
    default: () => false,
  }),
});

function supervisorNode(state) {
  if (state.taskComplete) {
    return { next: "FINISH", currentAgent: "supervisor" };
  }

  const last = state.messages.at(-1)?.content?.toLowerCase() ?? "";
  let next = "researcher";
  if (last.includes("research")) next = "researcher";
  else if (last.includes("write")) next = "writer";
  else if (last.includes("review")) next = "reviewer";

  return { next, currentAgent: "supervisor" };
}

function researcherNode() {
  return {
    messages: [new HumanMessage("Researcher: gathered quick notes.")],
    currentAgent: "researcher",
    taskComplete: true,
  };
}

function writerNode() {
  return {
    messages: [new HumanMessage("Writer: drafted a short response.")],
    currentAgent: "writer",
    taskComplete: true,
  };
}

function reviewerNode() {
  return {
    messages: [new HumanMessage("Reviewer: checked for clarity.")],
    currentAgent: "reviewer",
    taskComplete: true,
  };
}

const graph = new StateGraph(SupervisorState)
  .addNode("supervisor", supervisorNode)
  .addNode("researcher", researcherNode)
  .addNode("writer", writerNode)
  .addNode("reviewer", reviewerNode)
  .addEdge(START, "supervisor")
  .addConditionalEdges("supervisor", (state) => state.next, {
    researcher: "researcher",
    writer: "writer",
    reviewer: "reviewer",
    FINISH: END,
  })
  .addEdge("researcher", "supervisor")
  .addEdge("writer", "supervisor")
  .addEdge("reviewer", "supervisor")
  .compile();

async function main() {
  const result = await graph.invoke({
    messages: [new HumanMessage("Please research the topic.")],
    next: "researcher",
    currentAgent: "",
    taskComplete: false,
  });

  console.log("Final messages:");
  for (const message of result.messages) {
    console.log(`- ${message.content}`);
  }
}

main();
