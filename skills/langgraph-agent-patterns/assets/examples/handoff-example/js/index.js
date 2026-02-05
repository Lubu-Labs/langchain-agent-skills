import {
  Annotation,
  END,
  START,
  StateGraph,
  messagesStateReducer,
} from "@langchain/langgraph";
import { HumanMessage } from "@langchain/core/messages";

const HandoffState = Annotation.Root({
  messages: Annotation({
    reducer: messagesStateReducer,
    default: () => [],
  }),
  nextAgent: Annotation({
    reducer: (_, right) => right,
    default: () => "researcher",
  }),
  context: Annotation({
    reducer: (left, right) => ({ ...left, ...right }),
    default: () => ({}),
  }),
  currentAgent: Annotation({
    reducer: (_, right) => right,
    default: () => "",
  }),
});

function researcherNode() {
  return {
    messages: [new HumanMessage("Researcher: gathered key points.")],
    context: { research: "Key points" },
    nextAgent: "writer",
    currentAgent: "researcher",
  };
}

function writerNode(state) {
  const research = state.context?.research ?? "";
  return {
    messages: [new HumanMessage(`Writer: drafted using ${research}.`)],
    context: { draft: "Draft content" },
    nextAgent: "editor",
    currentAgent: "writer",
  };
}

function editorNode() {
  return {
    messages: [new HumanMessage("Editor: polished the draft.")],
    context: { final: "Final content" },
    nextAgent: "FINISH",
    currentAgent: "editor",
  };
}

const graph = new StateGraph(HandoffState)
  .addNode("researcher", researcherNode)
  .addNode("writer", writerNode)
  .addNode("editor", editorNode)
  .addEdge(START, "researcher")
  .addConditionalEdges("researcher", (state) => state.nextAgent, {
    writer: "writer",
    FINISH: END,
  })
  .addConditionalEdges("writer", (state) => state.nextAgent, {
    editor: "editor",
    FINISH: END,
  })
  .addConditionalEdges("editor", (state) => state.nextAgent, {
    FINISH: END,
  })
  .compile();

async function main() {
  const result = await graph.invoke({
    messages: [new HumanMessage("Create a short summary.")],
    nextAgent: "researcher",
    context: {},
    currentAgent: "",
  });

  console.log("Final context:");
  console.log(result.context);
  console.log("Messages:");
  for (const message of result.messages) {
    console.log(`- ${message.content}`);
  }
}

main();
