import {
  Annotation,
  END,
  START,
  StateGraph,
  messagesStateReducer,
} from "@langchain/langgraph";
import { HumanMessage } from "@langchain/core/messages";

const RouterState = Annotation.Root({
  messages: Annotation({
    reducer: messagesStateReducer,
    default: () => [],
  }),
  route: Annotation({
    reducer: (_, right) => right,
    default: () => "support",
  }),
});

function routerNode(state) {
  const query = state.messages[0]?.content?.toLowerCase() ?? "";

  let route = "support";
  if (["buy", "purchase", "price"].some((w) => query.includes(w))) {
    route = "sales";
  } else if (["help", "problem", "issue"].some((w) => query.includes(w))) {
    route = "support";
  } else if (["invoice", "payment", "billing"].some((w) => query.includes(w))) {
    route = "billing";
  }

  return { route };
}

function salesAgent() {
  return { messages: [new HumanMessage("Sales: Here's pricing info.")] };
}

function supportAgent() {
  return { messages: [new HumanMessage("Support: Let's troubleshoot.")] };
}

function billingAgent() {
  return { messages: [new HumanMessage("Billing: Here's your invoice.")] };
}

const graph = new StateGraph(RouterState)
  .addNode("router", routerNode)
  .addNode("sales", salesAgent)
  .addNode("support", supportAgent)
  .addNode("billing", billingAgent)
  .addEdge(START, "router")
  .addConditionalEdges("router", (state) => state.route, {
    sales: "sales",
    support: "support",
    billing: "billing",
  })
  .addEdge("sales", END)
  .addEdge("support", END)
  .addEdge("billing", END)
  .compile();

async function main() {
  const result = await graph.invoke({
    messages: [new HumanMessage("Can I see pricing?")],
    route: "support",
  });

  console.log("Final messages:");
  for (const message of result.messages) {
    console.log(`- ${message.content}`);
  }
}

main();
