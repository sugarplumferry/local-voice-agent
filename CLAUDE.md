# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Local, real-time voice English-practice agent: browser mic → FastAPI WebSocket → faster-whisper STT → LangGraph/LLM (reply + grammar correction) → sentence-level TTS → browser playback. Everything runs in Docker Compose (Ollama, Speaches/Kokoro TTS, Redis, backend, nginx frontend).

## Commands

All dev commands go through `dev.ps1` (`dev.bat <cmd>` is a wrapper). Run from the repo root.

| Command | When |
|---|---|
| `.\dev.ps1 up` / `down` / `ps` | Start / stop / status of all containers |
| `.\dev.ps1 logs` | Follow backend logs |
| `.\dev.ps1 rebuild` | After `backend/requirements.txt` or `Dockerfile` changes |
| `.\dev.ps1 restart` | After `.env` changes |
| `.\dev.ps1 frontend` | After `frontend/nginx.conf` changes |
| `.\dev.ps1 studio` | LangGraph Studio on :2024 (uses `backend/langgraph.json` → `graph.pipeline:pipeline`) |
| `.\dev.ps1 phone` | Containers + ngrok tunnel on :3000 (mobile needs HTTPS for mic) |

Hot reload: `backend/*.py` is volume-mounted and uvicorn runs with `--reload`; `frontend/*` is served straight from a volume, so just refresh the browser. App is at http://localhost:3000; backend directly at :8080.

First run: `docker exec voice-agent-ollama ollama pull llama3.1:8b`.

### Tests

Tests live in `backend/tests/` (pytest, `asyncio_mode = auto`, `pythonpath = .`). Every external service is mocked, so no live Ollama/Speaches/Redis is needed. `conftest.py` sets the service-URL env vars **before** importing `main` because `config.settings` is created at import time.

```powershell
.\dev.ps1 test                                            # docker exec … python -m pytest tests/ -v
docker exec voice-agent-backend python -m pytest tests/test_nodes.py::TestTranscribeNode -v   # single test
```

The Docker image installs only `requirements.txt`. The test dependencies (`pytest-asyncio`, `pytest-mock`, `fakeredis`) are in `requirements-test.txt`. If they are missing, install them with `docker exec voice-agent-backend pip install -r requirements-test.txt`, or run pytest from `backend/` in a local venv.

`frontend/` has no build step and no tests. After editing a JS file, at least run `node --check frontend/app.js`.

## Architecture

### The real flow lives in `backend/routers/ws.py`, not the graph

`graph/pipeline.py` is a linear 3-node graph: `transcribe_node` → `llm_response_node` → `tts_node`. Most of the real-time behaviour is orchestrated by `_handle_utterance` in `ws.py`:

1. **STT** runs *outside* the graph. `stt_service.transcribe_streaming()` yields segments, and each one is sent as `transcript_update`. `transcribe_node` only records the timing that ws.py passes in.
2. If the frontend reports `voice_state.active=true`, finalization waits (`_wait_user_silent`) until the user pauses. Then `transcript` is sent.
3. **History** is loaded with `RedisMemory.get_relevant_messages` (last 4 turns + up to 3 keyword-relevant older turns). **User facts** are loaded from `user:<user_id>:facts`. Both go into `AgentState`.
4. **The graph runs with `astream_events(version="v2")`.** ws.py intercepts `on_chat_model_stream` events from `llm_response_node` and does three things: streams `token` messages, cuts sentences on `_SENTENCE_END`, and pushes each sentence onto an ordered `tts_queue`. A single `_tts_runner` turns the queue into `audio` messages (base64 WAV), so TTS overlaps with LLM generation. `skip_tts=True` makes the in-graph `tts_node` a no-op.
5. After the stream ends: `add_turn` to Redis, then a fire-and-forget `_extract_and_store_facts` task. That task makes one LLM extraction call returning JSON facts, then an LLM dedup call per fact returning ADD / SKIP / REPLACE:N.

### Grammar correction is a marker in the single LLM reply

There is no separate grammar node any more, so the README diagram is outdated. `SYSTEM_PROMPT` in `graph/nodes.py` tells the model to append `\nGRAMMAR: …`. `GRAMMAR_MARKER` is shared by both files:
- `nodes.py` splits the final text into `response_text` and `feedback_text`.
- `ws.py` keeps back the last `len(GRAMMAR_MARKER)-1` streamed chars so a partial marker never reaches the chat bubble or TTS. It sends `feedback` when the node ends and suppresses a correction that repeats the previous one.

If you change the marker or the prompt format, update both sides.

### Providers are chosen per connection

The client sends `init_session {session_id, user_id, settings}`. `_build_providers` then picks:
- **STT:** `whisper_service` (local) or `OpenAISTTService`.
- **TTS:** `SpeachesService` or `OpenAITTSService`.
- **LLM config:** the choice is passed through LangGraph `configurable` (`llm_provider`, keys, `llm_model`). `nodes._get_llm` reads it and builds ChatOllama, ChatOpenAI or ChatGroq.

API keys come from the browser's localStorage and are never stored on the server. Any new STT/TTS service must implement `transcribe_streaming(bytes, language=)` or `text_to_speech(text) -> bytes`.

### Barge-in

When a new binary utterance arrives while a task is still running, ws.py:
1. sends `stop_playback`,
2. cancels the task and awaits it,
3. starts the new one.

`_handle_utterance` catches `CancelledError`, drains `tts_queue`, and re-raises. LLM retry (`_classify_llm_error`) treats `CancelledError` as `hard`, so it is never retried. Keep new awaits cancellation-safe.

### State / identity

- `SESSION_ID` is a new UUID on every page load and scopes `session:<id>:messages`: 50 turns, 7-day TTL.
- `USER_ID` is persisted in localStorage (`voice_agent_user_id`) and scopes long-term facts: 30 max, 90-day TTL.
- Any new field passed into the graph **must be declared in `graph/state.py` `AgentState`**, or LangGraph silently drops it. This already happened with `user_facts`.

### WebSocket protocol

The protocol is documented in the `ws.py` module docstring. The frontend handles it in the `switch (msg.type)` in `frontend/app.js`. VAD runs in an AudioWorklet (`vad-processor.js`) and is configured through `port.postMessage({type:"config", …})`. `audio-buffer.js` encodes the WAV that is sent as one binary frame per utterance. `routers/{chat,speak,transcribe,health}.py` are standalone REST endpoints; the voice UI only uses `/ws`.

## Gotchas

- **Files have been silently truncated before.** An NTFS/edit-sync issue cut off the ends of `nodes.py`, `ws.py` and `app.js` (see commits `69ddb02`, `2465f5c`). After editing a large file, check that its tail is intact, e.g. `node --check` for JS or `python -m py_compile` for Python.
- `config.py` defaults `ollama_model` to `qwen3:8b`, but `.env.example` sets `llama3.1:8b`. The value in `.env` wins.
- `_reference/` is gitignored, private prototype code. Don't modify it, and don't treat it as part of the app.
