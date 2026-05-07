import base64
import time

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_ollama import ChatOllama

from config import settings
from services.speaches import SpeachesService
from .state import AgentState

_speaches = SpeachesService()

SYSTEM_PROMPT = (
    "You are a friendly English speaking practice partner. "
    "Respond naturally and keep sentences short and conversational. "
    "Grammar corrections are handled separately by the system, "
    "so focus only on natural, fluent dialogue."
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
# If the caller pre-timed Whisper (e.g. ws.py streaming STT), the elapsed
# seconds are already in state["node_timings"]["transcribe_node"] — keep them.
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
# Node 2 – grammar_check_node
# Asks the LLM to decide YES/NO; sets grammar_error flag.
# ---------------------------------------------------------------------------
async def grammar_check_node(state: AgentState, config: RunnableConfig) -> dict:
    start = time.perf_counter()
    has_error = False
    try:
        llm = _get_llm(config, temperature=0, num_predict=30)
        prompt = (
            "Does this English sentence contain clear grammar errors? "
            "Answer YES or NO only, no explanation.\n"
            f'Sentence: "{state["current_input"]}"'
        )
        response = await llm.ainvoke([HumanMessage(content=prompt)])
        has_error = response.content.strip().upper().startswith("YES")
    except Exception:
        has_error = False

    return {
        "grammar_error": has_error,
        "node_timings": {
            **state["node_timings"],
            "grammar_check_node": round(time.perf_counter() - start, 4),
        },
    }


# ---------------------------------------------------------------------------
# Node 3 – llm_response_node
# Generates the main conversational reply via Ollama (streaming internally).
# astream_events() at the graph level will surface on_chat_model_stream events.
# ---------------------------------------------------------------------------
async def llm_response_node(state: AgentState, config: RunnableConfig) -> dict:
    start = time.perf_counter()
    llm = _get_llm(config, temperature=0.7, streaming=True)
    chunks: list[str] = []
    async for chunk in llm.astream(_build_chat_messages(state), config=config):
        if chunk.content:
            chunks.append(chunk.content)
    return {
        "response_text": "".join(chunks),
        "node_timings": {
            **state["node_timings"],
            "llm_response_node": round(time.perf_counter() - start, 4),
        },
    }


# ---------------------------------------------------------------------------
# Node 4 – feedback_node  (only reached when grammar_error is True)
# Generates a one-sentence friendly grammar correction.
# ---------------------------------------------------------------------------
async def feedback_node(state: AgentState, config: RunnableConfig) -> dict:
    start = time.perf_counter()
    llm = _get_llm(config, temperature=0.3, num_predict=80)
    prompt = (
        f'The learner said: "{state["current_input"]}"\n'
        "Provide a brief, friendly grammar correction in ONE sentence. "
        'Start with "By the way, you could say:"'
    )
    response = await llm.ainvoke([HumanMessage(content=prompt)])
    return {
        "feedback_text": response.content.strip(),
        "node_timings": {
            **state["node_timings"],
            "feedback_node": round(time.perf_counter() - start, 4),
        },
    }


# ---------------------------------------------------------------------------
# Node 5 – tts_node
# Sends the final text to Speaches TTS and stores base64-encoded audio.
# When skip_tts=True (ws.py handles TTS externally), timing is injected by
# the caller after _tts_runner completes — don't overwrite it here.
# ---------------------------------------------------------------------------
async def tts_node(state: AgentState) -> dict:
    audio_b64: str | None = None
    if not state.get("skip_tts"):
        start = time.perf_counter()
        try:
            text = state.get("response_text") or ""
            if state.get("feedback_text"):
                text = f"{text} {state['feedback_text']}"
            audio_bytes = await _speaches.text_to_speech(text)
            audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
        except Exception:
            pass  # audio is optional; the text response is still usable
        return {
            "audio_output": audio_b64,
            "node_timings": {
                **state["node_timings"],
                "tts_node": round(time.perf_counter() - start, 4),
            },
        }

    return {"audio_output": audio_b64}
