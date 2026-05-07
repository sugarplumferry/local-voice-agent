import time

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_ollama import ChatOllama

from config import settings
from services.speaches import SpeachesService
from .state import AgentState

_speaches = SpeachesService()

# Marker the LLM appends when it detects a grammar error.
# Must match the detection string used in ws.py.
GRAMMAR_MARKER = "\nGRAMMAR:"

SYSTEM_PROMPT = (
    "You are a friendly English speaking practice partner. "
    "Respond naturally and keep sentences short and conversational.\n\n"
    "If the user's message contains a clear grammar error, append a correction "
    "at the very end of your reply on a new line formatted EXACTLY as:\n"
    'GRAMMAR: By the way, you could say: "<corrected version>"\n'
    "If there are no grammar errors, do not include a GRAMMAR line."
)


def _get_llm(config: RunnableConfig, *, temperature: float, num_predict: int | None = None, streaming: bool = False):
    cfg = config.get("configurable", {})
    if cfg.get("llm_provider") == "openai":
        from langchain_openai import ChatOpenAI
        kw: dict = {
            "model": cfg.get("llm_model", "gpt-4o-mini"),
            "api_key": cfg.get("openai_api_key"),
            "temperature": temperature,
            "streaming": streaming,
        }
        if num_predict:
            kw["max_tokens"] = num_predict
        return ChatOpenAI(**kw)
    kw = {
        "model": settings.ollama_model,
        "base_url": settings.ollama_base_url,
        "temperature": temperature,
        "streaming": streaming,
        "keep_alive": -1,  # keep model in VRAM indefinitely between requests
    }
    if num_predict:
        kw["num_predict"] = num_predict
    return ChatOllama(**kw)


def _build_chat_messages(state: AgentState) -> list:
    msgs = [SystemMessage(content=SYSTEM_PROMPT)]
    for m in state["messages"]:  # pre-filtered by RedisMemory.get_relevant_messages
        if m["role"] == "user":
            msgs.append(HumanMessage(content=m["content"]))
        elif m["role"] == "assistant":
            msgs.append(AIMessage(content=m["content"]))
    msgs.append(HumanMessage(content=state["current_input"]))
    return msgs


# ---------------------------------------------------------------------------
# Node 1 – transcribe_node
# ---------------------------------------------------------------------------
async def transcribe_node(state: AgentState) -> dict:
    if "transcribe_node" in state["node_timings"]:
        return {}
    start = time.perf_counter()
    return {
        "node_timings": {
            **state["node_timings"],
            "transcribe_node": round(time.perf_counter() - start, 4),
        }
    }


# ---------------------------------------------------------------------------
# Node 2 – llm_response_node
# Single LLM call: generates the conversational reply and, if a grammar error
# is present, appends a GRAMMAR: correction line that ws.py strips before TTS.
# ---------------------------------------------------------------------------
async def llm_response_node(state: AgentState, config: RunnableConfig) -> dict:
    start = time.perf_counter()
    llm = _get_llm(config, temperature=0.7, streaming=True)
    chunks: list[str] = []
    async for chunk in llm.astream(_build_chat_messages(state), config=config):
        if chunk.content:
            chunks.append(chunk.content)

    full = "".join(chunks)

    if GRAMMAR_MARKER in full:
        idx = full.index(GRAMMAR_MARKER)
        response_text = full[:idx].strip()
        feedback_text = full[idx + len(GRAMMAR_MARKER):].strip()
    else:
        response_text = full.strip()
        feedback_text = None

    return {
        "response_text": response_text,
        "feedback_text": feedback_text,
        "grammar_error": feedback_text is not None,
        "node_timings": {
            **state["node_timings"],
            "llm_response_node": round(time.perf_counter() - start, 4),
        },
    }


# ---------------------------------------------------------------------------
# Node 3 – tts_node
# No-op when skip_tts=True (ws.py handles TTS per-sentence externally).
# ---------------------------------------------------------------------------
async def tts_node(state: AgentState) -> dict:
    if state.get("skip_tts"):
        return {"audio_output": None}

    start = time.perf_counter()
    audio_b64: str | None = None
    try:
        audio_bytes = await _speaches.text_to_speech(state.get("response_text") or "")
        import base64
        audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
    except Exception:
        pass
    return {
        "audio_output": audio_b64,
        "node_timings": {
            **state["node_timings"],
            "tts_node": round(time.perf_counter() - start, 4),
        },
    }
