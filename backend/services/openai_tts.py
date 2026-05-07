import logging

from langsmith import traceable
from openai import AsyncOpenAI

logger = logging.getLogger(__name__)

_VALID_VOICES = ("alloy", "echo", "fable", "onyx", "nova", "shimmer")


class OpenAITTSService:
    def __init__(self, api_key: str, voice: str = "alloy"):
        self._client = AsyncOpenAI(api_key=api_key)
        self.voice = voice if voice in _VALID_VOICES else "alloy"

    @traceable(name="tts_openai")
    async def text_to_speech(self, text: str) -> bytes:
        response = await self._client.audio.speech.create(
            model="tts-1",
            input=text,
            voice=self.voice,
            response_format="wav",
        )
        return response.content
