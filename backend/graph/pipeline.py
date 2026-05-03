from langgraph.graph import END, StateGraph

from .nodes import (
    feedback_node,
    grammar_check_node,
    llm_response_node,
    transcribe_node,
    tts_node,
)
from .state import AgentState


def _route_after_llm(state: AgentState) -> str:
    return "feedback_node" if state.get("grammar_error") else "tts_node"


def build_pipeline():
    g = StateGraph(AgentState)

    g.add_node("transcribe_node", transcribe_node)
    g.add_node("grammar_check_node", grammar_check_node)
    g.add_node("llm_response_node", llm_response_node)
    g.add_node("feedback_node", feedback_node)
    g.add_node("tts_node", tts_node)

    g.set_entry_point("transcribe_node")
    g.add_edge("transcribe_node", "grammar_check_node")
    g.add_edge("grammar_check_node", "llm_response_node")
    g.add_conditional_edges(
        "llm_response_node",
        _route_after_llm,
        {"feedback_node": "feedback_node", "tts_node": "tts_node"},
    )
    g.add_edge("feedback_node", "tts_node")
    g.add_edge("tts_node", END)

    return g.compile()


pipeline = build_pipeline()
