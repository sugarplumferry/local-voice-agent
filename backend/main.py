import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from routers import chat, health, speak, transcribe, ws
from services.speaches import SpeachesService
from services.whisper_stt import whisper_service

logging.basicConfig(level=logging.INFO)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load Whisper into memory at startup (downloads ~1.5 GB on first run,
    # then cached in the whisper_models Docker volume).
    asyncio.create_task(whisper_service.preload())
    # Load Kokoro TTS model into Speaches (background, non-blocking).
    asyncio.create_task(SpeachesService().preload_tts())
    yield


app = FastAPI(title="Local Voice Agent", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router,    tags=["health"])
app.include_router(transcribe.router, tags=["transcribe"])
app.include_router(chat.router,       tags=["chat"])
app.include_router(speak.router,      tags=["speak"])
app.include_router(ws.router,         tags=["ws"])
