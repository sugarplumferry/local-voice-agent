from fastapi import APIRouter
from pydantic import BaseModel

from config import settings
from services.ollama import OllamaService
from services.redis_memory import RedisMemory
from services.speaches import SpeachesService
from services.whisper_stt import whisper_service

router = APIRouter()


class HealthResponse(BaseModel):
    status: str
    services: dict


@router.get("/health", response_model=HealthResponse)
async def health():
    speaches = SpeachesService()
    ollama = OllamaService()
    redis = RedisMemory()

    checks = {
        "ollama": await ollama.health(),
        "model_loaded": await ollama.model_loaded(settings.ollama_model),
        "speaches_tts": await speaches.health(),
        "whisper_loaded": whisper_service.is_loaded,
        "redis": await redis.health(),
    }

    return HealthResponse(
        status="ok" if all(checks.values()) else "degraded",
        services=checks,
    )
