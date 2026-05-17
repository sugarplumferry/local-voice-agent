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
    provider = cfg.get("llm_provider", "local")

    if provider == "openai":
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

    if provider == "groq":
        from langchain_groq import ChatGroq
        kw = {
            "model": cfg.get("llm_model", "llama-3.3-70b-versatile"),
            "api_key": cfg.get("groq_api_key"),
            "temperature": temperature,
            "streaming": streaming,
        }
        if num_predict:
            kw["max_tokens"] = num_predict
        return ChatGroq(**kw)

    # Local Ollama (default)
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


# -----------------------