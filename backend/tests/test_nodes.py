"""
Unit tests for each LangGraph node — no live Ollama or Speaches required.
"""
import base64
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# transcribe_node
# ---------------------------------------------------------------------------

class TestTranscribeNode:
    async def test_records_timing(self, base_state):
        from graph.nodes import transcribe_node

        result = await transcribe_node(base_state)

        assert "transcribe_node" in result["node_timings"]
        assert result["node_timings"]["transcribe_node"] >= 0

    async def test_does_not_mutate_other_state_fields(self, base_state):
        from graph.nodes import transcribe_node

        result = await transcribe_node(base_state)

        # Only node_timings should be in the returned partial update
        assert list(result.keys()) == ["node_timings"]


# ---------------------------------------------------------------------------
# grammar_check_node
# ---------------------------------------------------------------------------

class TestGrammarCheckNode:
    async def test_detects_grammar_error(self, base_state):
        from graph.nodes import grammar_check_node

        with patch("graph.nodes.ChatOllama") as MockLLM:
            mock = AsyncMock()
            mock.ainvoke.return_value = MagicMock(content="YES, there is an error.")
            MockLLM.return_value = mock

            result = await grammar_check_node(base_state)

        assert result["grammar_error"] is True
        assert "grammar_check_node" in result["node_timings"]

    async def test_no_error_for_correct_sentence(self, correct_state):
        from graph.nodes import grammar_check_node

        with patch("graph.nodes.ChatOllama") as MockLLM:
            mock = AsyncMock()
            mock.ainvoke.return_value = MagicMock(content="NO")
            MockLLM.return_value = mock

            result = await grammar_check_node(correct_state)

        assert result["grammar_error"] is False

    async def test_defaults_to_false_on_llm_exception(self, base_state):
        from graph.nodes import grammar_check_node

        with patch("graph.nodes.ChatOllama") as MockLLM:
            MockLLM.side_effect = Exception("Connection refused")

            result = await grammar_check_node(base_state)

        assert result["grammar_error"] is False  # safe default
        assert "grammar_check_node" in result["node_timings"]

    async def test_timing_is_recorded(self, base_state):
        from graph.nodes import grammar_check_node

        with patch("graph.nodes.ChatOllama") as MockLLM:
            mock = AsyncMock()
            mock.ainvoke.return_value = MagicMock(content="NO")
            MockLLM.return_value = mock

            result = await grammar_check_node(base_state)

        assert result["node_timings"]["grammar_check_node"] >= 0


# ---------------------------------------------------------------------------
# llm_response_node
# ---------------------------------------------------------------------------

class TestLLMResponseNode:
    async def test_generates_response_text(self, base_state):
        from graph.nodes import llm_response_node

        with patch("graph.nodes.ChatOllama") as MockLLM:
            mock = AsyncMock()
            mock.ainvoke.return_value = MagicMock(content="That sounds great!")
            MockLLM.return_value = mock

            result = await llm_response_node(base_state)

        assert result["response_text"] == "That sounds great!"
        assert "llm_response_node" in result["node_timings"]

    async def test_system_prompt_is_included(self, base_state):
        from graph.nodes import llm_response_node
        from langchain_core.messages import SystemMessage

        captured = []

        with patch("graph.nodes.ChatOllama") as MockLLM:
            mock = AsyncMock()

            async def capture_and_respond(msgs):
                captured.extend(msgs)
                return MagicMock(content="Good!")

            mock.ainvoke.side_effect = capture_and_respond
            MockLLM.return_value = mock

            await llm_response_node(base_state)

        assert isinstance(captured[0], SystemMessage)

    async def test_conversation_history_is_included(self, base_state):
        from graph.nodes import llm_response_node
        from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

        base_state["messages"] = [
            {"role": "user", "content": "Hi there"},
            {"role": "assistant", "content": "Hello!"},
        ]
        captured = []

        with patch("graph.nodes.ChatOllama") as MockLLM:
            mock = AsyncMock()

            async def capture(msgs):
                captured.extend(msgs)
                return MagicMock(content="Sure!")

            mock.ainvoke.side_effect = capture
            MockLLM.return_value = mock

            await llm_response_node(base_state)

        # system + 2 history msgs + current input = 4
        assert len(captured) == 4
        assert isinstance(captured[0], SystemMessage)
        assert isinstance(captured[1], HumanMessage)
        assert isinstance(captured[2], AIMessage)
        assert isinstance(captured[3], HumanMessage)

    async def test_history_capped_at_20_messages(self, base_state):
        from graph.nodes import llm_response_node

        # 15 turns = 30 messages (over the 20-message cap)
        base_state["messages"] = [
            {"role": "user" if i % 2 == 0 else "assistant", "content": f"msg {i}"}
            for i in range(30)
        ]
        captured = []

        with patch("graph.nodes.ChatOllama") as MockLLM:
            mock = AsyncMock()

            async def capture(msgs):
                captured.extend(msgs)
                return MagicMock(content="OK")

            mock.ainvoke.side_effect = capture
            MockLLM.return_value = mock

            await llm_response_node(base_state)

        # system (1) + last 20 history msgs + current input (1) = 22
        assert len(captured) == 22


# ---------------------------------------------------------------------------
# feedback_node
# ---------------------------------------------------------------------------

class TestFeedbackNode:
    async def test_generates_feedback_text(self, base_state):
        from graph.nodes import feedback_node

        with patch("graph.nodes.ChatOllama") as MockLLM:
            mock = AsyncMock()
            mock.ainvoke.return_value = MagicMock(
                content='By the way, you could say: "I went to school yesterday."'
            )
            MockLLM.return_value = mock

            result = await feedback_node(base_state)

        assert "you could say" in result["feedback_text"].lower()
        assert "feedback_node" in result["node_timings"]

    async def test_feedback_text_is_stripped(self, base_state):
        from graph.nodes import feedback_node

        with patch("graph.nodes.ChatOllama") as MockLLM:
            mock = AsyncMock()
            mock.ainvoke.return_value = MagicMock(content="  By the way, you could say: X.  ")
            MockLLM.return_value = mock

            result = await feedback_node(base_state)

        assert result["feedback_text"] == 'By the way, you could say: X.'


# ---------------------------------------------------------------------------
# tts_node
# ---------------------------------------------------------------------------

class TestTTSNode:
    async def test_produces_base64_audio(self, base_state):
        from graph.nodes import tts_node

        base_state["response_text"] = "Hello there!"

        with patch("graph.nodes._speaches") as mock_speaches:
            mock_speaches.text_to_speech = AsyncMock(return_value=b"fake-audio-bytes")

            result = await tts_node(base_state)

        assert result["audio_output"] is not None
        decoded = base64.b64decode(result["audio_output"])
        assert decoded == b"fake-audio-bytes"
        assert "tts_node" in result["node_timings"]

    async def test_combines_response_and_feedback(self, base_state):
        from graph.nodes import tts_node

        base_state["response_text"] = "Great effort!"
        base_state["feedback_text"] = 'By the way, you could say: "I went."'

        sent_text = []

        with patch("graph.nodes._speaches") as mock_speaches:
            async def capture(text):
                sent_text.append(text)
                return b"audio"

            mock_speaches.text_to_speech = capture

            await tts_node(base_state)

        assert "Great effort!" in sent_text[0]
        assert "By the way" in sent_text[0]

    async def test_gracefully_handles_tts_failure(self, base_state):
        from graph.nodes import tts_node

        base_state["response_text"] = "Hello!"

        with patch("graph.nodes._speaches") as mock_speaches:
            mock_speaches.text_to_speech = AsyncMock(side_effect=Exception("TTS down"))

            result = await tts_node(base_state)

        assert result["audio_output"] is None          # degraded gracefully
        assert "tts_node" in result["node_timings"]    # timing still recorded

    async def test_timing_accumulates_previous_nodes(self, base_state):
        from graph.nodes import tts_node

        base_state["response_text"] = "Hi!"
        base_state["node_timings"] = {"transcribe_node": 0.001, "llm_response_node": 1.2}

        with patch("graph.nodes._speaches") as mock_speaches:
            mock_speaches.text_to_speech = AsyncMock(return_value=b"audio")

            result = await tts_node(base_state)

        timings = result["node_timings"]
        assert "transcribe_node" in timings
        assert "llm_response_node" in timings
        assert "tts_node" in timings
