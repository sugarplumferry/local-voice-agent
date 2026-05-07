"""
WebSocket endpoint — single persistent connection per client.

Protocol (all JSON except the binary WAV frames):

  Client → Server
    text  {type: "init_session", session_id: str, settings: {...}}
    bytes  <raw WAV audio for one utterance>

  Server → Client
    {type: "transcript_update", text: str}   — progressive segment
    {type: "transcript",        text: str}   — final confirmed transcript
    {type: "token",             content: str}
    {type: "feedback",          content: str}
    {type: "audio",             content: str}  — base64 WAV (one per sentence)
    {type: "done",              node_timings: dict}
    {type: "error",             message: str}

Settings schema (sent in init_session):
  {
    openai_api_key: str,
    stt:  "local" | "openai",
    tts:  "local" | "openai",
    tts_voice: "alloy" | "echo" | "fable" | "onyx" | "nova" | "shimmer",
    llm:  "local" | "openai",
    llm_model: str,   e.g. "gpt-4o-mini"
  }
"""

import asyncio
import base64
import json
import logging
import re
import time

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from config import settings
from graph.nodes import GRAMMAR_MARKER
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
    if user_settings.get("llm") == "openai" and api_key:
        pipeline_config["configurable"] = {
            "llm_provider": "openai",
            "openai_api_key": api_key,
            "llm_model": user_settings.get("llm_model", "gpt-4o-mini"),
        }

    return {"stt": stt_service, "tts": tts_service, "pipeline_config": pipeline_config}


@router.websocket("/ws")
async def websocket_endpoint(ws: WebSocket) -> None:
    await ws.accept()
    session_id = "default"
    current_task: asyncio.Task | None = None
    providers = _build_providers({})   # default: all local
    logger.info("WebSocket connected")

    try:
        while True:
            msg = await ws.receive()

            if msg.get("text"):
                try:
                    data = json.loads(msg["text"])
                except json.JSONDecodeError:
                    continue
                if data.get("type") == "init_session":
                    session_id = data.get("session_id", "default")
                    providers = _build_providers(data.get("settings", {}))
                    logger.info(
                        "Session init: %s  stt=%s tts=%s llm=%s",
                        session_id,
                        type(providers["stt"]).__name__,
                        type(providers["tts"]).__name__,
                        providers["pipeline_config"].get("configurable", {}).get("llm_provider", "local"),
                    )

            elif msg.get("bytes"):
                if current_task and not current_task.done():
                    current_task.cancel()
                    try:
                        await current_task
                    except asyncio.CancelledError:
                        pass

                current_task = asyncio.create_task(
                    _handle_utterance(ws, msg["bytes"], session_id, providers)
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


async def _handle_utterance(
    ws: WebSocket, wav_bytes: bytes, session_id: str, providers: dict
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

    await _send(ws, {"type": "transcript", "text": full_text})

    # ── 2. LangGraph pipeline + sentence-level TTS ────────────────────────────
    messages = await _memory.get_relevant_messages(session_id, full_text)
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
    response_acc = ""      # full accumulator for GRAMMAR: marker detection
    reply_done = False     # True once the GRAMMAR: marker is detected
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
                        prev_len = len(response_acc)
                        response_acc += content

                        marker_pos = response_acc.find(GRAMMAR_MARKER)
                        if marker_pos >= 0:
                            # GRAMMAR: marker found — send only the clean prefix
                            reply_done = True
                            safe_end = marker_pos          # absolute cut-off
                            new_safe = response_acc[prev_len:safe_end]
                            if new_safe:
                                await _send(ws, {"type": "token", "content": new_safe})
                                sentence_buf += new_safe
                            # Strip any partial marker prefix already in sentence_buf
                            for trim in range(len(GRAMMAR_MARKER), 0, -1):
                                if sentence_buf.endswith(GRAMMAR_MARKER[:trim]):
                                    sentence_buf = sentence_buf[:-trim]
                                    break
                            # Flush remaining clean response text to TTS
                            if sentence_buf.strip():
                                await tts_queue.put(sentence_buf.strip())
                                sentence_buf = ""
                        else:
                            await _send(ws, {"type": "token", "content": content})
                            sentence_buf += content
                            if _SENTENCE_END.search(sentence_buf.rstrip()):
                                chunk = sentence_buf.strip()
                                sentence_buf = ""
                                if chunk:
                                    await tts_queue.put(chunk)

                elif kind == "on_chain_end" and node:
                    output = event.get("data", {}).get("output", {})
                    if isinstance(output, dict):
                        accumulated.update(output)
                        # Send feedback when the single LLM node finishes
                        if node == "llm_response_node":
                            feedback = output.get("feedback_text")
                            if feedback:
                                await _send(ws, {"type": "feedback", "content": feedback})

        except Exception as exc:
            logger.exception("Pipeline error")
            await _send(ws, {"type": "error", "message": str(exc)})
            return

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

        node_timings = accumulated.get("node_timings", {})
        node_timings["tts_node"] = round(tts_elapsed_total, 4)

        await _send(ws, {
            "type":         "done",
            "node_timings": node_timings,
        })

    finally:
        if not tts_task.done():
            tts_task.cancel()
            try:
                await tts_task
            except asyncio.CancelledError:
                pass
