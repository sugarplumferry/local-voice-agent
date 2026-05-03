from typing import Optional
from typing_extensions import TypedDict


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
    # Per-node elapsed seconds, e.g. {"transcribe_node": 0.002, ...}
    node_timings: dict[str, float]
    # Redis session identifier
    session_id: str
    # Base64-encoded WAV audio from TTS (None if TTS was skipped/failed)
    audio_output: Optional[str]
    # Set True by the WebSocket handler to skip in-graph TTS (handled externally)
    skip_tts: bool
