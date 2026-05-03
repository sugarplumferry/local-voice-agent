import httpx

from config import settings


class OllamaService:
    def __init__(self):
        self.base_url = settings.ollama_base_url

    async def health(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                r = await client.get(f"{self.base_url}/api/tags")
                return r.status_code == 200
        except Exception:
            return False

    async def model_loaded(self, model: str) -> bool:
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                r = await client.get(f"{self.base_url}/api/tags")
                loaded = [m["name"] for m in r.json().get("models", [])]
                return any(model in name for name in loaded)
        except Exception:
            return False
