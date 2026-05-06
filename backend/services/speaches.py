import logging

import httpx
from langsmith import traceable

from config import settings

logger = logging.getLogger(__name__)


class SpeachesService:
    """TTS-only client for the Speaches service (STT is handled by WhisperService)."""

    def __init__(self):
        self.base_url = settings.speaches_base_url
        self.tts_model = settings.tts_model
        self.tts_voice = settings.tts_voice

    @traceable(name="tts_speaches")
    async def text_to_speech(self, text: str) -> bytes:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{self.base_url}/v1/audio/speech",
                json={
                    "model": self.tts_model,
                    "input": text,
                    "voice": self.tts_voice,
                },
            )
            response.raise_for_status()
            return response.content

    async def preload_tts(self) -> None:
        """Load the TTS model into Speaches via its model-management API.

        Speaches uses {model_id:path} routing so raw slashes must be kept
        in the URL (no percent-encoding).
        """
        await self._ensure_model(self.tts_model)

    async def _ensure_model(self, model_id: str) -> None:
        async with httpx.AsyncClient(timeout=600.0) as client:
            r = await client.get(f"{self.base_url}/v1/models/{model_id}")
            if r.status_code == 200:
                logger.info("TTS model already loaded: %s", model_id)
                return

            logger.info("Downloading TTS model %s (this may take a few minutes) ...", model_id)
            try:
                r = await client.post(f"{self.base_url}/v1/models/{model_id}")
                r.raise_for_status()
                logger.info("TTS model ready: %s", model_id)
            except Exception as e:
                logger.warning("TTS model download failed for %s: %s", model_id, e)

    async def health(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                r = await client.get(f"{self.base_url}/health")
                return r.status_code == 200
        except Exception:
            return False
