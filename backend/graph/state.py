from typing import Annotated, Optional
from typing_extensions import TypedDict


def _merge_dicts(a: dict, b: dict) -> dict:
    return {**a, **b}


class AgentState(TypedDict):
    # Full conversation history stored in / restored from Redis
    messages: list[dict]
    # Current user text input
    current_input: str
    # Whether a grammar error was detected
    grammar_error: bool
    # Friendly correction text (set only when grammar_error is True)
    feedback_text: Optional[str]
    # LLM response text
    response_text: Optional[str]
    # Per-node elapsed seconds — Annotated so parallel nodes can both update it
    node_timings: Annotated[dict[str, float], _merge_dicts]
    # Redis session identifier
    session_id: str
    # Base64-encoded WAV audio from TTS (None if TTS was skipped/failed)
    audio_output: Optional[str]
    # Set True by the WebSocket handler to skip in-graph TTS (handled externally)
    skip_tts: bool
    # Long-term cross-session facts about the user, loaded from Redis by the
    # WebSocket handler. Must be declared here or LangGraph drops it from state
    # before it reaches llm_response_node.
    user_facts: list[str]
