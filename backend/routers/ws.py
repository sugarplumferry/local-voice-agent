"""
WebSocket endpoint — single persistent connection per client.

Protocol (all JSON except the binary WAV frames):

  Client → Server
    text  {type: "init_session", session_id: str}
    bytes  <raw WAV audio for one utterance>

  Server → Client
    {type: "transcript_update", text: str}   — progressive segment
    {type: "transcript",        text: str}   — final confirmed transcript
    {type: "token",             content: str}
    {type: "feedback",          content: str}
    {type: "audio",             content: str}  — base64 WAV (one per sentence)
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

from config import settings
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


@router.websocket("/ws")
async def websocket_endpoint(ws: WebSocket) -> None:
    await ws.accept()
    session_id = "default"
    current_task: asyncio.Task | None = None
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
                    logger.info("Session init: %s", session_id)

            elif msg.get("bytes"):
                if current_task and not current_task.done():
                    current_task.cancel()
                    try:
                        await current_task
                    except asyncio.CancelledError:
                        pass

                current_task = asyncio.create_task(
                    _handle_utterance(ws, msg["bytes"], session_id)
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


async def _handle_utterance(ws: WebSocket, wav_bytes: bytes, session_id: str) -> None:
    # ── 1. Streaming transcription ────────────────────────────────────────────
    full_text = ""
    transcribe_start = time.perf_counter()
    try:
        async for segment in whisper_service.transcribe_streaming(
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

    accumulated: dict = {}
    sentence_buf = ""
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
                audio_bytes = await _speaches.text_to_speech(text)
                tts_elapsed_total += time.perf_counter() - t0
                b64 = base64.b64encode(audio_bytes).decode()
                await _send(ws, {"type": "audio", "content": b64})
            except Exception:
                logger.exception("TTS error for chunk: %r", text)

    tts_task = asyncio.create_task(_tts_runner())

    try:
        try:
            async for event in pipeline.astream_events(
                initial_state,
                version="v2",
                config={"metadata": {"transcribe_s": transcribe_elapsed}},
            ):
                kind = event["event"]
                node = event.get("metadata", {}).get("langgraph_node", "")

                if kind == "on_chat_model_stream" and node == "llm_response_node":
                    content = getattr(event["data"]["chunk"], "content", None)
                    if content:
                        await _send(ws, {"type": "token", "content": content})
                        sentence_buf += content
                        # Queue TTS for each completed sentence
                        if _SENTENCE_END.search(sentence_buf.rstrip()):
                            chunk = sentence_buf.strip()
                            sentence_buf = ""
                            if chunk:
                                await tts_queue.put(chunk)

                elif kind == "on_chain_end" and node:
                    output = event.get("data", {}).get("output", {})
                    if isinstance(output, dict):
                        accumulated.update(output)
                        # Send feedback immediately when feedback_node completes
                        if node == "feedback_node":
                            feedback = output.get("feedback_text")
                            if feedback:
                                await _send(ws, {"type": "feedback", "content": feedback})

        except Exception as exc:
            logger.exception("Pipeline error")
            await _send(ws, {"type": "error", "message": str(exc)})
            return

        # Flush any remaining text that didn't end with punctuation
        remaining = sentence_buf.strip()
        if remaining:
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
