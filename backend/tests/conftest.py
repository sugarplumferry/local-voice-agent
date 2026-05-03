import os

# Must be set before any app module is imported so pydantic-settings picks them up.
os.environ.setdefault("OLLAMA_BASE_URL", "http://localhost:11434")
os.environ.setdefault("SPEACHES_BASE_URL", "http://localhost:8000")
os.environ.setdefault("REDIS_URL", "redis://localhost:6379")
os.environ.setdefault("LANGCHAIN_TRACING_V2", "false")
os.environ.setdefault("LANGCHAIN_API_KEY", "dummy-key-for-tests")

import pytest
from httpx import ASGITransport, AsyncClient

from main import app


@pytest.fixture
async def client():
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as c:
        yield c


@pytest.fixture
def base_state() -> dict:
    """AgentState with a sentence that has a grammar error."""
    return {
        "messages": [],
        "current_input": "I goes to school yesterday.",
        "grammar_error": False,
        "feedback_text": None,
        "response_text": None,
        "node_timings": {},
        "session_id": "pytest-session",
        "audio_output": None,
    }


@pytest.fixture
def correct_state(base_state) -> dict:
    """AgentState with a grammatically correct sentence."""
    return {**base_state, "current_input": "I went to school yesterday."}
