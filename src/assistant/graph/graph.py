from langgraph.graph import StateGraph, END

from assistant.graph.state import AgentState
from assistant.agents.retriever_agent import retriever_node
from assistant.agents.writer_agent import writer_node
from assistant.agents.verifier_agent import verifier_node


def build_graph():

    graph = StateGraph(AgentState)

    graph.add_node("retriever", retriever_node)
    graph.add_node("writer", writer_node)
    graph.add_node("verifier", verifier_node)

    graph.set_entry_point("retriever")

    graph.add_edge("retriever", "writer")
    graph.add_edge("writer", "verifier")

    def check_verified(state):

        if state["verified"]:
            return END
        else:
            return "retriever"

    graph.add_conditional_edges(
        "verifier",
        check_verified
    )

    return graph.compile()