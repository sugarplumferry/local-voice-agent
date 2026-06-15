"""
WebSocket endpoint — single persistent connection per client.

Protocol (all JSON except the binary WAV frames):

  Client → Server
    text  {type: "init_session", session_id: str, user_id: str, settings: {...}}
    bytes  <raw WAV audio for one utterance>

  Server → Client
    {type: "transcript_update", text: str}   — progressive segment
    {type: "transcript",        text: str}   — final confirmed transcript
    {type: "token",             content: str}
    {type: "feedback",          content: str}
    {type: "audio",             content: str}  — base64 WAV (one per sentence)
    {type: "stop_playback"}                    — barge-in: drop queued / playing audio
    {type: "done",              node_timings: dict}
    {type: "error",             message: str}
"""

import asyncio
import base64
import json
import logging
import re
import time

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from langchain_core.messages import HumanMessage, SystemMessage

from config import settings
from graph.nodes import GRAMMAR_MARKER, _get_llm
from graph.pipeline import pipeline
from graph.state import AgentState
from services.redis_memory import RedisMemory
from services.speaches import SpeachesService
from services.whisper_stt import whisper_service

logger = logging.getLogger(__name__)
router = APIRouter()
_memory = RedisMemory()
_speaches = SpeachesService()

# Sentence boundary — ends with . ! ? (optionally followed by quotes/parens/space)
_SENTENCE_END = re.compile(r'[.!?]["\')]*\s*$')


def _build_providers(user_settings: dict) -> dict:
    """Return {stt_service, tts_service, pipeline_config} based on user settings."""
    api_key = (user_settings.get("openai_api_key") or "").strip()
    groq_key = (user_settings.get("groq_api_key") or "").strip()

    # STT
    if user_settings.get("stt") == "openai" and api_key:
        from services.openai_stt import OpenAISTTService
        stt_service = OpenAISTTService(api_key)
    else:
        stt_service = whisper_service

    # TTS
    if user_settings.get("tts") == "openai" and api_key:
        from services.openai_tts import OpenAITTSService
        tts_service = OpenAITTSService(api_key, voice=user_settings.get("tts_voice", "alloy"))
    else:
        tts_service = _speaches

    # LLM — passed via LangGraph configurable
    pipeline_config: dict = {"metadata": {}}
    llm_choice = user_settings.get("llm", "local")
    if llm_choice == "openai" and api_key:
        pipeline_config["configurable"] = {
            "llm_provider": "openai",
            "openai_api_key": api_key,
            "llm_model": user_settings.get("llm_model", "gpt-4o-mini"),
        }
    elif llm_choice == "groq" and groq_key:
        pipeline_config["configurable"] = {
            "llm_provider": "groq",
            "groq_api_key": groq_key,
            "llm_model": user_settings.get("groq_model", "llama-3.3-70b-versatile"),
        }

    return {"stt": stt_service, "tts": tts_service, "pipeline_config": pipeline_config}


@router.websocket("/ws")
async def websocket_endpoint(ws: WebSocket) -> None:
    await ws.accept()
    session_id = "default"
    user_id = "default"       # long-term identity (fact scope) — distinct from session
    current_task: asyncio.Task | None = None
    providers = _build_providers({})   # default: all local
    # Per-connection cross-turn state.
    #   last_feedback — suppress the small-LLM tic of regurgitating a grammar
    #     correction from earlier history when the current turn has no error.
    #   user_speaking — frontend-reported VAD state; used to hold transcript
    #     finalization while the user is still mid-sentence after transcribe.
    session_state: dict = {"last_feedback": None, "user_speaking": False}
    logger.info("WebSocket connected")

    try:
        while True:
            msg = await ws.receive()

            # Raw ws.receive() does NOT raise WebSocketDisconnect — it returns a
            # {"type": "websocket.disconnect"} message. Detect it and break, or
            # the next receive() raises RuntimeError ("Cannot call receive once
            # a disconnect message has been received").
            if msg.get("type") == "websocket.disconnect":
                raise WebSocketDisconnect(msg.get("code", 1000))

            if msg.get("text"):
                try:
                    data = json.loads(msg["text"])
                except json.JSONDecodeError:
                    continue
                if data.get("type") == "voice_state":
                    session_state["user_speaking"] = bool(data.get("active"))
                    continue
                if data.get("type") == "init_session":
                    session_id = data.get("session_id", "default")
                    # Fall back to session_id when the client did not send a
                    # persistent user_id (older frontends). This preserves
                    # backwards compatibility but disables long-term memory.
                    user_id = data.get("user_id") or session_id
                    providers = _build_providers(data.get("settings", {}))
                    logger.info(
                        "Session init: session=%s user=%s  stt=%s tts=%s llm=%s",
                        session_id,
                        user_id,
                        type(providers["stt"]).__name__,
                        type(providers["tts"]).__name__,
                        providers["pipeline_config"].get("configurable", {}).get("llm_provider", "local"),
                    )

            elif msg.get("bytes"):
                # Barge-in: a new utterance arrived while the previous one is
                # still being processed. Cancel everything cleanly:
                #  1. tell the client to stop any audio currently playing
                #  2. cancel the in-flight task (aborts LLM stream + drains TTS queue)
                #  3. wait for it to fully unwind before starting the new task
                if current_task and not current_task.done():
                    await _send(ws, {"type": "stop_playback"})
                    current_task.cancel()
                    try:
                        await current_task
                    except asyncio.CancelledError:
                        pass
                    except Exception:
                        logger.exception("Previous task raised during barge-in cancel")

                current_task = asyncio.create_task(
                    _handle_utterance(ws, msg["bytes"], session_id, user_id, providers, session_state)
                )

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected (session=%s)", session_id)
        if current_task and not current_task.done():
            current_task.cancel()


async def _send(ws: WebSocket, payload: dict) -> None:
    """Send JSON, silently dropping if the socket is already closed."""
    try:
        await ws.send_json(payload)
    except Exception:
        pass


async def _wait_user_silent(state: dict, grace_s: float) -> None:
    """Block until `state['user_speaking']` has been False for `grace_s`
    consecutive seconds. Polls at 50 ms.

    No overall timeout — if the user genuinely never stops, the escape hatch
    is the barge-in path: a new utterance arriving cancels this task, and
    the await-sleep below makes the cancellation cooperative."""
    silent_since: float | None = None
    while True:
        if state.get("user_speaking"):
            silent_since = None
        elif silent_since is None:
            silent_since = time.monotonic()
        elif time.monotonic() - silent_since >= grace_s:
            return
        await asyncio.sleep(0.05)


# ─── Long-term fact extraction (background task) ───────────────────────────────

MIN_TURN_LEN_FOR_FACT_EXTRACTION = 15  # chars — skip ultra-short utterances
MAX_FACTS_PER_TURN = 3                 # cap per turn so a chatty extractor can't flood

FACT_EXTRACTION_SYSTEM_PROMPT = (
    "You analyze a SINGLE conversation turn between a learner and an English "
    "speaking-practice partner. Your job is to identify STABLE long-term facts "
    "about the learner that the partner should remember for future sessions.\n\n"
    "ALWAYS extract personal identity details the learner states about "
    "themselves: their name, job, location, age, hobbies, goals, family, "
    "English level, recurring mistakes. A learner's name is ALWAYS a good fact.\n\n"
    "Phrase EVERY fact as a complete third-person sentence about the learner, "
    "starting with \"Learner\". Examples:\n"
    "  - \"Learner's name is Charlie\"\n"
    "  - \"Learner works as a software engineer in Taipei\"\n"
    "  - \"Learner is a beginner who confuses past tense forms\"\n"
    "  - \"Learner is preparing for IELTS speaking\"\n"
    "Never output a bare word or a first-person sentence.\n\n"
    "Do NOT extract:\n"
    "  - Anything about the assistant, or about the conversation itself "
    "(NOT: \"learner's name was forgotten by the assistant\", "
    "\"learner mentioned this earlier\")\n"
    "  - Transient or in-the-moment states "
    "(NOT: \"is hungry now\", \"is tired today\", \"is not well prepared for the conversation\")\n"
    "  - Pure greetings, fillers, or politeness with no information\n"
    "EVERY fact you output MUST still be true a week from now. If unsure, "
    "leave it out.\n\n"
    "Output STRICT JSON: {\"facts\": [\"fact 1\", \"fact 2\"]}.\n"
    "If the turn reveals no stable facts, output {\"facts\": []}.\n"
    "Output JSON only — no commentary, no markdown fences."
)


FACT_DEDUP_SYSTEM_PROMPT = (
    "You maintain a short list of stable long-term facts about a learner. "
    "A NEW candidate fact has arrived. Decide EXACTLY ONE of:\n"
    "  ADD          — the candidate adds information not covered by any existing fact\n"
    "  SKIP         — the candidate is already covered (same meaning, less specific, or a near-duplicate)\n"
    "  REPLACE:<N>  — the candidate is a MORE SPECIFIC or MORE ACCURATE version of existing fact #N (1-based)\n\n"
    "Output EXACTLY one line containing only one of those three tokens. "
    "No prose, no punctuation, no explanation."
)


def _parse_facts_json(text: str) -> list[str]:
    """Extract the `facts` list from a JSON-ish LLM response. Tolerant of
    surrounding noise (markdown fences, prose) — finds the outermost {...}
    block and falls back to an empty list on any parse problem."""
    if not text:
        return []
    match = re.search(r'\{[^{}]*"facts"[^{}]*\}', text, re.DOTALL)
    if not match:
        match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return []
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return []
    facts = data.get("facts") if isinstance(data, dict) else None
    if not isinstance(facts, list):
        return []
    return [f.strip() for f in facts if isinstance(f, str) and f.strip()][:MAX_FACTS_PER_TURN]


async def _dedup_decision(
    existing: list[str], candidate: str, pipeline_config: dict
) -> tuple[str, int | None]:
    """Ask a small LLM whether `candidate` is new info, a duplicate, or a
    refinement of an existing fact. Returns one of:
        ("ADD", None) | ("SKIP", None) | ("REPLACE", zero_based_index)
    Defaults to ADD on any error so we never silently drop facts."""
    if not existing:
        return ("ADD", None)
    listing = "\n".join(f"{i + 1}. {f}" for i, f in enumerate(existing))
    user_msg = f"Existing facts:\n{listing}\n\nCandidate: {candidate}"
    try:
        llm = _get_llm(
            pipeline_config,
            temperature=0,
            num_predict=20,
            streaming=False,
        )
        resp = await llm.ainvoke(
            [
                SystemMessage(content=FACT_DEDUP_SYSTEM_PROMPT),
                HumanMessage(content=user_msg),
            ],
            config=pipeline_config,
        )
        text = (resp.content if hasattr(resp, "content") else str(resp)).strip().upper()
    except Exception:
        logger.exception("Fact dedup failed (non-fatal); defaulting to ADD")
        return ("ADD", None)

    if text.startswith("SKIP"):
        return ("SKIP", None)
    if text.startswith("REPLACE"):
        try:
            tail = text.split(":", 1)[1].strip().split()[0]
            idx = int(re.sub(r"[^0-9]", "", tail) or "0")
            if 1 <= idx <= len(existing):
                return ("REPLACE", idx - 1)
        except (IndexError, ValueError):
            pass
    return ("ADD", None)


async def _extract_and_store_facts(
    user_id: str,
    user_text: str,
    assistant_text: str,
    pipeline_config: dict,
) -> None:
    """Best-effort background: call the LLM to find stable user facts, then
    LLM-dedup each candidate against existing facts before persisting. Never
    blocks the reply path; all exceptions are logged and swallowed."""
    if len(user_text or "") < MIN_TURN_LEN_FOR_FACT_EXTRACTION:
        return
    try:
        llm = _get_llm(
            pipeline_config,
            temperature=0,
            num_predict=200,
            streaming=False,
        )
        turn_text = (
            f"USER: {user_text.strip()}\n"
            f"ASSISTANT: {assistant_text.strip()}"
        )
        resp = await llm.ainvoke(
            [
                SystemMessage(content=FACT_EXTRACTION_SYSTEM_PROMPT),
                HumanMessage(content=turn_text),
            ],
            config=pipeline_config,
        )
        text = resp.content if hasattr(resp, "content") else str(resp)
        facts = _parse_facts_json(text if isinstance(text, str) else str(text))

        # One-at-a-time so a REPLACE made earlier is visible to the next candidate.
        for fact in facts:
            existing = await _memory.get_user_facts(user_id)
            action, idx = await _dedup_decision(existing, fact, pipeline_config)
            if action == "SKIP":
                logger.info("Skipped duplicate fact for user=%s: %r", user_id, fact)
            elif action == "REPLACE" and idx is not None:
                if await _memory.replace_user_fact(user_id, idx, fact):
                    logger.info(
                        "Replaced fact #%d for user=%s: %r", idx + 1, user_id, fact
                    )
            else:
                if await _memory.add_user_fact(user_id, fact):
                    logger.info("Added fact for user=%s: %r", user_id, fact)
    except Exception:
        logger.exception("Fact extraction failed (non-fatal)")


async def _handle_utterance(
    ws: WebSocket,
    wav_bytes: bytes,
    session_id: str,
    user_id: str,
    providers: dict,
    session_state: dict,
) -> None:
    stt_service = providers["stt"]
    tts_service = providers["tts"]
    pipeline_config = providers["pipeline_config"]

    # ── 1. Streaming transcription ────────────────────────────────────────────
    full_text = ""
    transcribe_start = time.perf_counter()
    try:
        async for segment in stt_service.transcribe_streaming(
            wav_bytes, language=settings.whisper_language
        ):
            full_text += segment
            await _send(ws, {"type": "transcript_update", "text": full_text.strip()})
    except Exception as exc:
        logger.exception("Transcription error")
        await _send(ws, {"type": "error", "message": str(exc)})
        return
    transcribe_elapsed = round(time.perf_counter() - transcribe_start, 4)

    full_text = full_text.strip()
    if not full_text:
        await _send(ws, {"type": "done", "node_timings": {}})  # reset UI
        return

    # If the user is still mid-sentence when transcribe finishes, hold the
    # finalize step. A new utterance arriving meanwhile cancels this task
    # (barge-in) and the frontend will resend combined PCM so we transcribe
    # the full thought instead of splitting it.
    if session_state.get("user_speaking"):
        logger.info(
            "Holding transcript — user still speaking after transcribe (%ss)",
            round(transcribe_elapsed, 2),
        )
        await _wait_user_silent(session_state, grace_s=0.3)
        logger.info("Resumed (user paused)")

    await _send(ws, {"type": "transcript", "text": full_text})

    # ── 2. LangGraph pipeline + sentence-level TTS ────────────────────────────
    messages = await _memory.get_relevant_messages(session_id, full_text)
    user_facts = await _memory.get_user_facts(user_id)
    initial_state: AgentState = {
        "messages":      messages,
        "current_input": full_text,
        "grammar_error": False,
        "feedback_text": None,
        "response_text": None,
        "node_timings":  {"transcribe_node": transcribe_elapsed},
        "session_id":    session_id,
        "audio_output":  None,
        "skip_tts":      True,   # ws.py handles TTS per-sentence externally
        "user_facts":    user_facts,
    }

    run_config = {
        **pipeline_config,
        "metadata": {
            **pipeline_config.get("metadata", {}),
            "transcribe_s": transcribe_elapsed,
        },
    }

    accumulated: dict = {}
    sentence_buf = ""
    response_acc = ""      # full accumulator; tracks every token from the model
    committed_len = 0      # chars of response_acc already forwarded to the client
    reply_done = False     # True once the GRAMMAR: marker is fully detected
    tts_elapsed_total = 0.0

    # Ordered TTS queue: a single runner processes sentences in order so audio
    # chunks arrive at the client in the same sequence as the text.
    tts_queue: asyncio.Queue[str | None] = asyncio.Queue()

    async def _tts_runner():
        nonlocal tts_elapsed_total
        while True:
            text = await tts_queue.get()
            if text is None:
                break
            try:
                t0 = time.perf_counter()
                audio_bytes = await tts_service.text_to_speech(text)
                tts_elapsed_total += time.perf_counter() - t0
                b64 = base64.b64encode(audio_bytes).decode()
                await _send(ws, {"type": "audio", "content": b64})
            except Exception:
                logger.exception("TTS error for chunk: %r", text)

    tts_task = asyncio.create_task(_tts_runner())

    try:
        try:
            async for event in pipeline.astream_events(
                initial_state, version="v2", config=run_config
            ):
                kind = event["event"]
                node = event.get("metadata", {}).get("langgraph_node", "")

                if kind == "on_chat_model_stream" and node == "llm_response_node":
                    content = getattr(event["data"]["chunk"], "content", None)
                    if content and not reply_done:
                        response_acc += content

                        marker_pos = response_acc.find(GRAMMAR_MARKER)
                        if marker_pos >= 0:
                            # Marker fully present — commit only the clean prefix
                            reply_done = True
                            to_send = response_acc[committed_len:marker_pos]
                        else:
                            # Hold back the last (marker_len-1) chars; they might be
                            # the start of a marker that completes in the next chunk.
                            safe_end = max(
                                committed_len,
                                len(response_acc) - len(GRAMMAR_MARKER) + 1,
                            )
                            to_send = response_acc[committed_len:safe_end]
                            committed_len = safe_end

                        if to_send:
                            await _send(ws, {"type": "token", "content": to_send})
                            sentence_buf += to_send
                            if _SENTENCE_END.search(sentence_buf.rstrip()):
                                chunk = sentence_buf.strip()
                                sentence_buf = ""
                                if chunk:
                                    await tts_queue.put(chunk)

                        if reply_done:
                            # Trim any partial marker prefix left in sentence_buf
                            for trim in range(len(GRAMMAR_MARKER), 0, -1):
                                if sentence_buf.endswith(GRAMMAR_MARKER[:trim]):
                                    sentence_buf = sentence_buf[:-trim]
                                    break
                            if sentence_buf.strip():
                                await tts_queue.put(sentence_buf.strip())
                                sentence_buf = ""

                elif kind == "on_chain_end" and node:
                    output = event.get("data", {}).get("output", {})
                    if isinstance(output, dict):
                        accumulated.update(output)
                        # Send feedback when the single LLM node finishes.
                        # Suppress consecutive duplicates within the same
                        # connection — small LLMs sometimes re-emit a correction
                        # from earlier in the conversation history.
                        if node == "llm_response_node":
                            feedback = output.get("feedback_text")
                            if feedback and feedback != session_state.get("last_feedback"):
                                await _send(ws, {"type": "feedback", "content": feedback})
                                session_state["last_feedback"] = feedback
                            elif feedback:
                                logger.info("Suppressed duplicate feedback: %r", feedback)

        except Exception as exc:
            logger.exception("Pipeline error")
            await _send(ws, {"type": "error", "message": str(exc)})
            return

        # Release any chars held back in the safety buffer (no marker found)
        if not reply_done and committed_len < len(response_acc):
            leftover = response_acc[committed_len:]
            await _send(ws, {"type": "token", "content": leftover})
            sentence_buf += leftover

        # Flush any remaining response text that didn't end with punctuation
        remaining = sentence_buf.strip()
        if remaining and not reply_done:
            await tts_queue.put(remaining)

        # Signal TTS runner to stop and wait for all audio to be sent
        await tts_queue.put(None)
        await tts_task

        # ── 3. Post-pipeline ──────────────────────────────────────────────────
        response_text = accumulated.get("response_text", "")
        if response_text:
            await _memory.add_turn(session_id, full_text, response_text)
            # Background: ask the LLM to extract any stable long-term facts
            # about this user from the turn. Best-effort, fire-and-forget so
            # the reply isn't blocked. Survives across sessions via redis.
            asyncio.create_task(
                _extract_and_store_facts(user_id, full_text, response_text, pipeline_config)
            )

        node_timings = accumulated.get("node_timings", {})
        node_timings["tts_node"] = round(tts_elapsed_total, 4)

        await _send(ws, {
            "type":         "done",
            "node_timings": node_timings,
        })

    except asyncio.CancelledError:
        # Barge-in path: someone outside (the websocket loop) cancelled us.
        # Drain the TTS queue so the runner exits promptly instead of spinning
        # on the next pending sentence, then re-raise so the outer loop knows.
        logger.info("Utterance cancelled (barge-in)")
        while not tts_queue.empty():
            try:
                tts_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
        await tts_queue.put(None)
        raise
    finally:
        if not tts_task.done():
            tts_task.cancel()
            try:
                await tts_task
            except asyncio.CancelledError:
                pass
