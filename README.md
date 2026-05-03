# Local Voice Agent — English Speaking Practice

A fully **local**, real-time voice-based English practice system.  
Speak → get an instant AI reply + grammar correction → hear it spoken back.  
Everything runs on your own machine: no cloud API, no data leaving your network.

---

## Features

- **Voice Activity Detection** — AudioWorklet-based VAD with draggable threshold marker on the VU meter
- **Streaming transcription** — faster-whisper segments appear progressively as you speak
- **Conversational AI** — llama3.1:8b via Ollama responds naturally in real time (streaming tokens)
- **Grammar feedback** — a separate LLM node detects errors and gives a one-sentence friendly correction
- **Sentence-level TTS** — Kokoro-82M starts speaking each sentence as soon as it's generated, overlapping with the LLM still writing the next one
- **Smart conversation memory** — stores up to 50 turns in Redis; retrieves the last 4 turns + up to 3 topically relevant older turns on each request
- **Mobile access** — ngrok + nginx WebSocket proxy lets your phone connect over HTTPS
- **LangGraph pipeline** — full observability via LangSmith tracing

---

## Architecture

```
Phone / Browser
      │  wss://
      ▼
  nginx :3000
      │  /ws  proxy
      ▼
FastAPI WebSocket /ws
      │
      ├─ 1. faster-whisper (STT, streaming segments)
      │
      └─ 2. LangGraph pipeline
               transcribe_node
                    │
               grammar_check_node  ── Ollama (YES/NO)
                    │
               llm_response_node   ── Ollama (streaming tokens)
                    │
          ┌─── grammar error? ───┐
         YES                    NO
          │                      │
     feedback_node          (skip)
          │
       tts_node  (skipped — ws.py handles TTS per-sentence)
                    │
      sentence-level TTS  ── Speaches / Kokoro-82M
      (fires for each sentence while LLM is still writing)
```

**Services (all Docker):**

| Service | Image | Role |
|---|---|---|
| `ollama` | `ollama/ollama` | LLM inference (llama3.1:8b) |
| `speaches` | `ghcr.io/speaches-ai/speaches:latest-cpu` | TTS (Kokoro-82M) |
| `redis` | `redis:7-alpine` | Conversation memory |
| `backend` | local build | FastAPI + LangGraph + faster-whisper |
| `frontend` | `nginx:alpine` | Static files + WebSocket proxy |

---

## Prerequisites

- **Docker Desktop** (Windows / Mac / Linux)
- **8 GB RAM** minimum (16 GB recommended for smooth LLM inference)
- *(optional)* Nvidia GPU + `nvidia-container-toolkit` for faster inference

---

## Quick Start

### 1. Clone & configure

```bash
git clone https://github.com/YOUR_USERNAME/local-voice-agent.git
cd local-voice-agent
cp .env.example .env
# Edit .env — only LANGCHAIN_API_KEY needs changing (optional)
```

### 2. Start all services

```bash
# Windows
dev.bat up

# Mac / Linux
docker compose up -d
```

### 3. Pull the Ollama model (first time only, ~4.7 GB)

```bash
docker exec voice-agent-ollama ollama pull llama3.1:8b
```

### 4. Open the app

```
http://localhost:3000
```

Click the microphone button and start speaking. The Whisper model downloads automatically on the first request (~240 MB for `small`).

---

## Phone Access (ngrok)

Mobile browsers require HTTPS for microphone access. ngrok provides a free HTTPS tunnel.

```bash
# Install ngrok (Windows)
winget install ngrok
ngrok config add-authtoken YOUR_NGROK_TOKEN   # free at ngrok.com

# Start everything + tunnel in one command
dev.bat phone
```

Copy the `https://` URL from the ngrok window and open it on your phone.

```bash
dev.bat phone-stop   # stop tunnel + containers
```

---

## Dev Scripts (`dev.bat` / `dev.ps1`)

| Command | Description |
|---|---|
| `dev.bat up` | Start all containers |
| `dev.bat down` | Stop all containers |
| `dev.bat phone` | Start containers + ngrok tunnel |
| `dev.bat phone-stop` | Stop ngrok + containers |
| `dev.bat ngrok` | Start ngrok only (containers already running) |
| `dev.bat ngrok-stop` | Kill ngrok |
| `dev.bat rebuild` | Rebuild backend image (after `requirements.txt` changes) |
| `dev.bat restart` | Restart backend (after `.env` changes) |
| `dev.bat frontend` | Recreate frontend container (after `nginx.conf` changes) |
| `dev.bat logs` | Follow backend logs |
| `dev.bat studio` | Start LangGraph Studio |
| `dev.bat ps` | Show container status |

---

## Configuration (`.env`)

| Variable | Default | Description |
|---|---|---|
| `OLLAMA_MODEL` | `llama3.1:8b` | Any model available in Ollama |
| `WHISPER_MODEL` | `small` | `tiny` / `base` / `small` / `medium` / `large-v3-turbo` |
| `WHISPER_LANGUAGE` | `en` | Skip auto-detection; set to your language code |
| `TTS_MODEL` | `speaches-ai/Kokoro-82M-v1.0-ONNX-fp16` | Kokoro voice model |
| `TTS_VOICE` | `af_heart` | Voice ID (see Speaches docs) |
| `LANGCHAIN_API_KEY` | *(empty)* | Optional — enables LangSmith tracing |

---

## UI Controls

| Control | Description |
|---|---|
| 🎙 button | Start / stop listening |
| VU meter | Live microphone level; **drag the red marker** to set VAD threshold |
| Silence slider | How long silence must last before the utterance is sent (300–1000 ms) |

**Bubble colours:**
- Blue (right) — your speech
- Dark (left) — agent reply
- Green border (left) — grammar correction

---

## Conversation Memory

- Up to **50 turns** stored per session in Redis (TTL: 7 days)
- Each request uses: last 4 turns (always) + up to 3 older turns selected by **keyword relevance** to the current utterance
- Session is tied to the browser tab — **refresh = new session**

---

## LangSmith Tracing (optional)

1. Sign up at [smith.langchain.com](https://smith.langchain.com/) (free)
2. Add your API key to `.env`
3. Restart: `dev.bat restart`
4. Every conversation turn appears as a traced run with per-node timings

---

## LangGraph Studio (optional)

Visualise and debug the pipeline interactively:

```bash
dev.bat studio
# Then open: https://smith.langchain.com/studio/?baseUrl=http://localhost:2024
```

---

## GPU Support

Switch Speaches to the CUDA image in `docker-compose.yml`:

```yaml
speaches:
  image: ghcr.io/speaches-ai/speaches:latest-cuda
```

For Ollama GPU passthrough, add under the `ollama` service:

```yaml
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
```

---

## Project Structure

```
local-voice-agent/
├── backend/
│   ├── graph/
│   │   ├── nodes.py          # LangGraph nodes (transcribe, grammar, LLM, feedback, TTS)
│   │   ├── pipeline.py       # Graph definition & routing
│   │   └── state.py          # AgentState TypedDict
│   ├── routers/
│   │   └── ws.py             # WebSocket endpoint + sentence-level TTS streaming
│   ├── services/
│   │   ├── redis_memory.py   # Conversation history (recency + keyword retrieval)
│   │   ├── speaches.py       # TTS client
│   │   └── whisper_stt.py    # faster-whisper STT service
│   ├── config.py             # Pydantic settings
│   ├── main.py               # FastAPI app
│   ├── langgraph.json        # LangGraph Studio config
│   └── requirements.txt
├── frontend/
│   ├── app.js                # WebSocket + AudioWorklet + VAD + audio queue
│   ├── vad-processor.js      # AudioWorklet processor (runs in audio thread)
│   ├── audio-buffer.js       # WAV encoder utilities
│   ├── index.html
│   ├── style.css
│   └── nginx.conf            # Static files + /ws proxy
├── docker-compose.yml
├── .env.example
├── dev.ps1                   # PowerShell dev scripts
└── dev.bat                   # Wrapper for dev.ps1
```

---

## License

MIT
