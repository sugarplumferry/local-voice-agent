import asyncio
import io
import logging
from concurrent.futures import ThreadPoolExecutor

from faster_whisper import WhisperModel

from config import settings

logger = logging.getLogger(__name__)

# Single-threaded executor keeps GPU/CPU memory contention low
_executor = ThreadPoolExecutor(max_workers=1)


class WhisperService:
    def __init__(self):
        self._model: WhisperModel | None = None
        self._lock = asyncio.Lock()

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    async def preload(self) -> None:
        """Download and load the model into memory (idempotent)."""
        async with self._lock:
            if self._model is None:
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(None, self._load)

    def _load(self) -> None:
        logger.info(
            "Loading Whisper model: %s  device=%s  compute=%s",
            settings.whisper_model,
            settings.whisper_device,
            settings.whisper_compute_type,
        )
        self._model = WhisperModel(
            settings.whisper_model,
            device=settings.whisper_device,
            compute_type=settings.whisper_compute_type,
        )
        logger.info("Whisper model ready.")

    async def transcribe(self, audio_bytes: bytes, language: str | None = None) -> str:
        if self._model is None:
            await self.preload()
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            _executor, self._do_transcribe, audio_bytes, language
        )

    def _do_transcribe(self, audio_bytes: bytes, language: str | None) -> str:
        segments, _ = self._model.transcribe(
            io.BytesIO(audio_bytes),
            beam_size=1,
            language=language,
            vad_filter=True,
        )
        return " ".join(s.text for s in segments).strip()

    async def transcribe_streaming(self, audio_bytes: bytes, language: str | None = None):
        """Async-generator that yields segment texts as faster-whisper produces them.

        Filters Whisper hallucinations (silence / noise → spurious text):
          - drops segments whose avg_logprob is too low (model unsure)
          - drops segments where no_speech_prob is too high (silence)

        These thresholds are deliberately loose — production agents tune per
        audio quality. See `notes/whisper-hallucination.md` for typical values.
        """
        if self._model is None:
            await self.preload()

        loop: asyncio.AbstractEventLoop = asyncio.get_running_loop()
        q: asyncio.Queue = asyncio.Queue()

        def _run() -> None:
            try:
                segments, _ = self._model.transcribe(
                    io.BytesIO(audio_bytes), beam_size=1, language=language, vad_filter=True
                )
                for seg in segments:
                    avg_logprob = getattr(seg, "avg_logprob", 0.0)
                    no_speech_prob = getattr(seg, "no_speech_prob", 0.0)
                    text = (seg.text or "").strip()
                    if _looks_like_whisper_garbage(text, avg_logprob, no_speech_prob):
                        logger.debug(
                            "Dropping whisper segment as likely hallucination: "
                            "text=%r avg_logprob=%.3f no_speech_prob=%.3f",
                            text, avg_logprob, no_speech_prob,
                        )
                        continue
                    loop.call_soon_threadsafe(q.put_nowait, ("seg", seg.text))
                loop.call_soon_threadsafe(q.put_nowait, ("done", None))
            except Exception as exc:  # noqa: BLE001
                loop.call_soon_threadsafe(q.put_nowait, ("err", exc))

        _executor.submit(_run)

        while True:
            kind, value = await q.get()
            if kind == "done":
                return
            if kind == "err":
                raise value  # type: ignore[misc]
            yield value


# ─── Whisper hallucination filter ──────────────────────────────────────────────
# Whisper is well-known for inventing "Thank you", "Bye", song lyrics, etc.
# when fed silence or non-speech audio. These thresholds catch the common cases
# while staying lenient enough not to drop quiet real speech.

# avg_logprob:    -1.0 ≈ near-random tokens; > -0.5 ≈ confident
# no_speech_prob: 0..1; >0.6 means whisper itself thinks this is silence
_MIN_AVG_LOGPROB        = -1.0
_MAX_NO_SPEECH_PROB     = 0.6
_MIN_TEXT_LEN_CHARS     = 2

# Frequent hallucination patterns (case-insensitive substring match on trimmed text).
# Add to this list as you observe new ones in production logs.
_HALLUCINATION_PHRASES = {
    "thank you",
    "thanks for watching",
    "please subscribe",
    "see you next time",
    "bye bye",
    "you",  # very common single-word hallucination on silence
    ".",
    "...",
}


def _looks_like_whisper_garbage(text: str, avg_logprob: float, no_speech_prob: float) -> bool:
    if not text or len(text) < _MIN_TEXT_LEN_CHARS:
        return True
    if avg_logprob < _MIN_AVG_LOGPROB:
        return True
    if no_speech_prob > _MAX_NO_SPEECH_PROB:
        return True
    lowered = text.lower().strip(" .!?,\"'")
    if lowered in _HALLUCINATION_PHRASES:
        return True
    return False


whisper_service = WhisperService()
