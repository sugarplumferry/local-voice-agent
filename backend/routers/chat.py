import json

from fastapi import APIRouter
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from graph.pipeline import pipeline
from graph.state import AgentState
from services.redis_memory import RedisMemory

router = APIRouter()
_memory = RedisMemory()


class ChatRequest(BaseModel):
    text: str
    session_id: str = "default"


@router.post("/chat")
async def chat(request: ChatRequest):
    messages = await _memory.get_messages(request.session_id)

    initial_state: AgentState = {
        "messages": messages,
        "current_input": request.text,
        "grammar_error": False,
        "feedback_text": None,
        "response_text": None,
        "node_timings": {},
        "session_id": request.session_id,
        "audio_output": None,
    }

    async def event_generator():
        # Accumulate partial node outputs as the graph runs
        accumulated: dict = {}

        async for event in pipeline.astream_events(initial_state, version="v2"):
            kind = event["event"]
            node = event.get("metadata", {}).get("langgraph_node", "")

            # Stream LLM tokens only from the main response node
            if kind == "on_chat_model_stream" and node == "llm_response_node":
                chunk = event["data"]["chunk"]
                content = getattr(chunk, "content", None)
                if content:
                    yield {
                        "event": "token",
                        "data": json.dumps({"content": content}),
                    }

            # Collect each node's partial state update
            elif kind == "on_chain_end" and node:
                output = event.get("data", {}).get("output", {})
                if isinstance(output, dict):
                    accumulated.update(output)

        # Persist conversation turn to Redis
        response_text = accumulated.get("response_text", "")
        if response_text:
            await _memory.add_turn(request.session_id, request.text, response_text)

        # Send grammar feedback if present
        feedback = accumulated.get("feedback_text")
        if feedback:
            yield {
                "event": "feedback",
                "data": json.dumps({"content": feedback}),
            }

        # Send base64 audio if TTS succeeded
        audio = accumulated.get("audio_output")
        if audio:
            yield {
                "event": "audio",
                "data": json.dumps({"content": audio}),
            }

        # Final done event carries node timings
        yield {
            "event": "done",
            "data": json.dumps({"node_timings": accumulated.get("node_timings", {})}),
        }

    return EventSourceResponse(event_generator())
