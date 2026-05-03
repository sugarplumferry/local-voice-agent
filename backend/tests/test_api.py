"""
API endpoint tests — all external calls are mocked so no live services are needed.
"""
import json
from unittest.mock import AsyncMock, MagicMock, PropertyMock

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_sse(text: str) -> list[dict]:
    """Parse an SSE response body into a list of {'event': ..., 'data': ...} dicts."""
    text = text.replace("\r\n", "\n")  # sse_starlette emits \r\n; normalise first
    events = []
    for block in text.strip().split("\n\n"):
        ev: dict = {}
        for line in block.split("\n"):
            if line.startswith("event:"):
                ev["event"] = line[6:].strip()
            elif line.startswith("data:"):
                raw = line[5:].strip()
                try:
                    ev["data"] = json.loads(raw)
                except json.JSONDecodeError:
                    ev["data"] = raw
        if ev:
            events.append(ev)
    return events


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------

class TestHealth:
    def _mock_all_healthy(self, mocker, whisper_loaded: bool = True):
        mocker.patch("routers.health.OllamaService.health", new_callable=AsyncMock, return_value=True)
        mocker.patch("routers.health.OllamaService.model_loaded", new_callable=AsyncMock, return_value=True)
        mocker.patch("routers.health.SpeachesService.health", new_callable=AsyncMock, return_value=True)
        mocker.patch("routers.health.RedisMemory.health", new_callable=AsyncMock, return_value=True)
        from services.whisper_stt import WhisperService
        mocker.patch.object(WhisperService, "is_loaded", new_callable=PropertyMock, return_value=whisper_loaded)

    async def test_all_services_ok(self, client, mocker):
        self._mock_all_healthy(mocker)

        r = await client.get("/health")

        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ok"
        assert all(body["services"].values())

    async def test_degraded_when_ollama_model_not_loaded(self, client, mocker):
        mocker.patch("routers.health.OllamaService.health", new_callable=AsyncMock, return_value=True)
        mocker.patch("routers.health.OllamaService.model_loaded", new_callable=AsyncMock, return_value=False)
        mocker.patch("routers.health.SpeachesService.health", new_callable=AsyncMock, return_value=True)
        mocker.patch("routers.health.RedisMemory.health", new_callable=AsyncMock, return_value=True)
        from services.whisper_stt import WhisperService
        mocker.patch.object(WhisperService, "is_loaded", new_callable=PropertyMock, return_value=True)

        r = await client.get("/health")

        assert r.json()["status"] == "degraded"

    async def test_response_has_expected_service_keys(self, client, mocker):
        self._mock_all_healthy(mocker)

        r = await client.get("/health")

        assert set(r.json()["services"].keys()) == {
            "ollama", "model_loaded", "speaches_tts", "whisper_loaded", "redis"
        }


# ---------------------------------------------------------------------------
# POST /transcribe
# ---------------------------------------------------------------------------

class TestTranscribe:
    async def test_returns_transcribed_text(self, client, mocker):
        mocker.patch(
            "routers.transcribe.whisper_service.transcribe",
            new_callable=AsyncMock,
            return_value="I went to school yesterday.",
        )

        r = await client.post(
            "/transcribe",
            files={"audio": ("test.wav", b"fake-wav-bytes", "audio/wav")},
        )

        assert r.status_code == 200
        assert r.json()["text"] == "I went to school yesterday."

    async def test_returns_500_on_stt_failure(self, client, mocker):
        mocker.patch(
            "routers.transcribe.whisper_service.transcribe",
            new_callable=AsyncMock,
            side_effect=Exception("Whisper model error"),
        )

        r = await client.post(
            "/transcribe",
            files={"audio": ("test.wav", b"fake-wav-bytes", "audio/wav")},
        )

        assert r.status_code == 500


# ---------------------------------------------------------------------------
# POST /speak
# ---------------------------------------------------------------------------

class TestSpeak:
    async def test_returns_wav_bytes(self, client, mocker):
        mocker.patch(
            "routers.speak._speaches.text_to_speech",
            new_callable=AsyncMock,
            return_value=b"RIFF\x00fake-wav",
        )

        r = await client.post("/speak", json={"text": "Hello world"})

        assert r.status_code == 200
        assert r.headers["content-type"] == "audio/wav"
        assert r.content.startswith(b"RIFF")

    async def test_returns_base64_when_requested(self, client, mocker):
        mocker.patch(
            "routers.speak._speaches.text_to_speech",
            new_callable=AsyncMock,
            return_value=b"audio-bytes",
        )

        r = await client.post("/speak", json={"text": "Hello", "encode_base64": True})

        assert r.status_code == 200
        assert "audio" in r.json()

    async def test_returns_502_on_tts_failure(self, client, mocker):
        mocker.patch(
            "routers.speak._speaches.text_to_speech",
            new_callable=AsyncMock,
            side_effect=Exception("TTS failed"),
        )

        r = await client.post("/speak", json={"text": "Hello"})

        assert r.status_code == 502


# ---------------------------------------------------------------------------
# POST /chat  (SSE streaming)
# ---------------------------------------------------------------------------

class TestChat:
    def _setup(self, mocker, events: list):
        """
        Patch the LangGraph pipeline and Redis memory.
        Returns the memory mock so individual tests can assert on it.
        """
        async def fake_astream(*args, **kwargs):
            for e in events:
                yield e

        from graph.pipeline import pipeline
        from services.redis_memory import RedisMemory

        mocker.patch.object(pipeline, "astream_events", new=fake_astream)

        mock_memory = AsyncMock(spec=RedisMemory)
        mock_memory.get_messages.return_value = []
        mocker.patch("routers.chat._memory", mock_memory)

        return mock_memory

    async def test_streams_llm_tokens(self, client, mocker):
        self._setup(mocker, [
            {"event": "on_chat_model_stream", "metadata": {"langgraph_node": "llm_response_node"}, "data": {"chunk": MagicMock(content="Hello ")}},
            {"event": "on_chat_model_stream", "metadata": {"langgraph_node": "llm_response_node"}, "data": {"chunk": MagicMock(content="there!")}},
            {"event": "on_chain_end", "metadata": {"langgraph_node": "tts_node"}, "data": {"output": {"response_text": "Hello there!", "audio_output": None, "node_timings": {}}}},
        ])

        r = await client.post("/chat", json={"text": "Hi!", "session_id": "t1"})
        events = parse_sse(r.text)

        tokens = [e for e in events if e.get("event") == "token"]
        assert len(tokens) == 2
        assert tokens[0]["data"]["content"] == "Hello "
        assert tokens[1]["data"]["content"] == "there!"

    async def test_feedback_event_emitted_on_grammar_error(self, client, mocker):
        self._setup(mocker, [
            {"event": "on_chain_end", "metadata": {"langgraph_node": "llm_response_node"}, "data": {"output": {"response_text": "That's fine!"}}},
            {"event": "on_chain_end", "metadata": {"langgraph_node": "feedback_node"}, "data": {"output": {"feedback_text": 'By the way, you could say: "I went to school."'}}},
            {"event": "on_chain_end", "metadata": {"langgraph_node": "tts_node"}, "data": {"output": {"audio_output": None, "node_timings": {}}}},
        ])

        r = await client.post("/chat", json={"text": "I goes to school.", "session_id": "t2"})
        events = parse_sse(r.text)

        feedback = [e for e in events if e.get("event") == "feedback"]
        assert len(feedback) == 1
        assert "you could say" in feedback[0]["data"]["content"].lower()

    async def test_no_feedback_event_for_correct_sentence(self, client, mocker):
        self._setup(mocker, [
            {"event": "on_chain_end", "metadata": {"langgraph_node": "tts_node"}, "data": {"output": {"response_text": "Good job!", "audio_output": None, "node_timings": {}}}},
        ])

        r = await client.post("/chat", json={"text": "I went to school.", "session_id": "t3"})
        events = parse_sse(r.text)

        assert not any(e.get("event") == "feedback" for e in events)

    async def test_done_event_contains_node_timings(self, client, mocker):
        self._setup(mocker, [
            {"event": "on_chain_end", "metadata": {"langgraph_node": "tts_node"}, "data": {"output": {"response_text": "Hello!", "audio_output": None, "node_timings": {"llm_response_node": 1.2, "tts_node": 0.4}}}},
        ])

        r = await client.post("/chat", json={"text": "Hi", "session_id": "t4"})
        events = parse_sse(r.text)

        done = [e for e in events if e.get("event") == "done"]
        assert len(done) == 1
        assert "node_timings" in done[0]["data"]

    async def test_turn_saved_to_redis_memory(self, client, mocker):
        mock_memory = self._setup(mocker, [
            {"event": "on_chain_end", "metadata": {"langgraph_node": "tts_node"}, "data": {"output": {"response_text": "Great!", "audio_output": None, "node_timings": {}}}},
        ])

        await client.post("/chat", json={"text": "Practice sentence.", "session_id": "t5"})

        mock_memory.add_turn.assert_called_once_with("t5", "Practice sentence.", "Great!")

    async def test_grammar_check_tokens_not_streamed(self, client, mocker):
        """Tokens from nodes other than llm_response_node must not leak to the client."""
        self._setup(mocker, [
            # From grammar_check_node — should be suppressed
            {"event": "on_chat_model_stream", "metadata": {"langgraph_node": "grammar_check_node"}, "data": {"chunk": MagicMock(content="YES")}},
            # From llm_response_node — should be streamed
            {"event": "on_chat_model_stream", "metadata": {"langgraph_node": "llm_response_node"}, "data": {"chunk": MagicMock(content="Sure!")}},
            {"event": "on_chain_end", "metadata": {"langgraph_node": "tts_node"}, "data": {"output": {"response_text": "Sure!", "audio_output": None, "node_timings": {}}}},
        ])

        r = await client.post("/chat", json={"text": "Is this correct?", "session_id": "t6"})
        events = parse_sse(r.text)

        tokens = [e for e in events if e.get("event") == "token"]
        assert len(tokens) == 1
        assert tokens[0]["data"]["content"] == "Sure!"

    async def test_audio_event_emitted_when_tts_succeeds(self, client, mocker):
        import base64
        fake_audio = base64.b64encode(b"fake-wav").decode()

        self._setup(mocker, [
            {"event": "on_chain_end", "metadata": {"langgraph_node": "tts_node"}, "data": {"output": {"response_text": "Hi!", "audio_output": fake_audio, "node_timings": {}}}},
        ])

        r = await client.post("/chat", json={"text": "Hello", "session_id": "t7"})
        events = parse_sse(r.text)

        audio_events = [e for e in events if e.get("event") == "audio"]
        assert len(audio_events) == 1
        assert audio_events[0]["data"]["content"] == fake_audio
