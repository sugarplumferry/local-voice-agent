from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    ollama_base_url: str = "http://ollama:11434"
    ollama_model: str = "qwen3:8b"

    # Speaches — TTS only
    speaches_base_url: str = "http://speaches:8000"
    tts_model: str = "speaches-ai/Kokoro-82M-v1.0-ONNX-fp16"
    tts_voice: str = "af_heart"

    # faster-whisper — runs directly inside the backend container
    whisper_model: str = "small"
    whisper_device: str = "cpu"
    whisper_compute_type: str = "int8"
    whisper_language: str = "en"

    redis_url: str = "redis://redis:6379"

    langchain_api_key: str = ""
    langchain_project: str = "local-voice-agent"
    langchain_tracing_v2: str = "true"


settings = Settings()
