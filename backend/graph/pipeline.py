from langgraph.graph import END, StateGraph

from .nodes import llm_response_node, transcribe_node, tts_node
from .state import AgentState


def build_pipeline():
    g = StateGraph(AgentState)

    g.add_node("transcribe_node", transcribe_node)
    g.add_node("llm_response_node", llm_response_node)
    g.add_node("tts_node", tts_node)

    g.set_entry_point("transcribe_node")
    g.add_edge("transcribe_node", "llm_response_node")
    g.add_edge("llm_response_node", "tts_node")
    g.add_edge("tts_node", END)

    return g.compile()


pipeline = build_pipeline()
