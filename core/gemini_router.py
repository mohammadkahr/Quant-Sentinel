"""Fixed-model routing, credential cycling, bounded transient retries."""
import itertools
import threading
import time
from google import genai
from google.genai import types
from google.genai.errors import APIError
from core.config import settings

class GeminiExclusiveRouter:
    def __init__(self):
        self.keys = itertools.cycle(settings.api_keys_list)
        self.lock = threading.Lock()
    def generate(self, prompt, system_instruction="", schema=None):
        if not settings.api_keys_list:
            raise RuntimeError("Set GEMINI_API_KEY or GEMINI_API_KEYS in .env")
        for attempt in range(4):
            with self.lock:
                key = next(self.keys)
            try:
                with genai.Client(api_key=key, http_options=types.HttpOptions(timeout=60000)) as client:
                    response = client.models.generate_content(
                        model=settings.GLOBAL_LLM_MODEL, contents=prompt,
                        config=types.GenerateContentConfig(
                            system_instruction=system_instruction,
                            temperature=.1 if schema else .3,
                            response_mime_type="application/json" if schema else "text/plain",
                            response_schema=schema))
                if not response.text:
                    raise RuntimeError("Empty or blocked model response")
                usage = response.usage_metadata
                return {"text": response.text, "model_used": settings.GLOBAL_LLM_MODEL,
                        "tokens_used": getattr(usage, "total_token_count", None),
                        "api_calls": attempt + 1,
                        "usage": usage.model_dump(mode="json") if usage else None}
            except APIError as error:
                if error.code not in {429, 500, 502, 503, 504} or attempt == 3:
                    raise RuntimeError(f"Gemini request failed (HTTP {error.code})") from None
                time.sleep(min(3 * 1.5 ** attempt, 10))
        raise RuntimeError("Inference exhausted retries")

gemini_router = GeminiExclusiveRouter()
