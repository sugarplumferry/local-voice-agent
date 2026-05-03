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
            beam_size=5,
            language=language,
        )
        return " ".join(s.text for s in segments).strip()

    async def transcribe_streaming(self, audio_bytes: bytes, language: str | None = None):
        """Async-generator that yields segment texts as faster-whisper produces them."""
        if self._model is None:
            await self.preload()

        loop: asyncio.AbstractEventLoop = asyncio.get_running_loop()
        q: asyncio.Queue = asyncio.Queue()

        def _run() -> None:
            try:
                segments, _ = self._model.transcribe(
                    io.BytesIO(audio_bytes), beam_size=5, language=language
                )
                for seg in segments:
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


whisper_service = WhisperService()
