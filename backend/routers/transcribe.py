from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel

from services.whisper_stt import whisper_service

router = APIRouter()


class TranscribeResponse(BaseModel):
    text: str


@router.post("/transcribe", response_model=TranscribeResponse)
async def transcribe(audio: UploadFile = File(...), language: str | None = None):
    audio_bytes = await audio.read()
    try:
        text = await whisper_service.transcribe(audio_bytes, language=language)
        return TranscribeResponse(text=text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transcription failed: {e}")
