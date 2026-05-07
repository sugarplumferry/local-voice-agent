import logging

from openai import AsyncOpenAI

logger = logging.getLogger(__name__)


class OpenAISTTService:
    def __init__(self, api_key: str):
        self._client = AsyncOpenAI(api_key=api_key)

    async def transcribe(self, audio_bytes: bytes, language: str | None = None) -> str:
        transcript = await self._client.audio.transcriptions.create(
            model="whisper-1",
            file=("audio.wav", audio_bytes, "audio/wav"),
            language=language,
        )
        return transcript.text

    async def transcribe_streaming(self, audio_bytes: bytes, language: str | None = None):
        """OpenAI Whisper API has no streaming — yield the full result once."""
        text = await self.transcribe(audio_bytes, language)
        if text:
            yield text
