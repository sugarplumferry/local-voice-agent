import base64

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel

from services.speaches import SpeachesService

router = APIRouter()
_speaches = SpeachesService()


class SpeakRequest(BaseModel):
    text: str
    encode_base64: bool = False


@router.post("/speak")
async def speak(request: SpeakRequest):
    try:
        audio_bytes = await _speaches.text_to_speech(request.text)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"TTS failed: {e}")

    if request.encode_base64:
        return {"audio": base64.b64encode(audio_bytes).decode()}
    return Response(content=audio_bytes, media_type="audio/wav")
