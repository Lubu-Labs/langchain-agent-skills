import {
  Annotation,
  END,
  Send,
  START,
  StateGraph,
} from "@langchain/langgraph";

const OrchestratorState = Annotation.Root({
  task: Annotation({
    reducer: (_, right) => right,
    default: () => "",
  }),
  subtasks: Annotation({
    reducer: (_, right) => right,
    default: () => [],
  }),
  results: Annotation({
    reducer: (left, right) => left.concat(right),
    default: () => [],
  }),
  expectedCount: Annotation({
    reducer: (_, right) => right,
    default: () => 0,
  }),
  finalResult: Annotation({
    reducer: (_, right) => right,
    default: () => "",
  }),
});

function orchestratorNode(state) {
  const subtasks = [
    { id: 1, query: `Research part A of: ${state.task}` },
    { id: 2, query: `Research part B of: ${state.task}` },
    { id: 3, query: `Research part C of: ${state.task}` },
  ];

  return { subtasks, expectedCount: subtasks.length };
}

function workerNode(state) {
  const subtask = state.subtask;
  const result = `Result for ${subtask.query}`;
  return { results: [{ id: subtask.id, result }] };
}

function aggregatorNode(state) {
  if (state.expectedCount && state.results.length < state.expectedCount) {
    return {};
  }

  const finalResult = state.results.map((r) => r.result).join("\n");
  return { finalResult };
}

function dispatchWorkers(state) {
  return state.subtasks.map((subtask) => new Send("worker", { subtask }));
}

const graph = new StateGraph(OrchestratorState)
  .addNode("orchestrator", orchestratorNode)
  .addNode("worker", workerNode)
  .addNode("aggregator", aggregatorNode)
  .addEdge(START, "orchestrator")
  .addConditionalEdges("orchestrator", dispatchWorkers)
  .addEdge("worker", "aggregator")
  .addEdge("aggregator", END)
  .compile();

async function main() {
  const result = await graph.invoke({
    task: "Prepare a short summary",
    subtasks: [],
    results: [],
    expectedCount: 0,
    finalResult: "",
  });

  console.log("Final result:");
  console.log(result.finalResult || "(no result)");
}

main();
