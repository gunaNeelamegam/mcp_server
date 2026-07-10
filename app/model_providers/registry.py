"""Maps the active provider name (from config) to a ModelProvider instance.

Adding a new model = one new provider file + one line in _BUILDERS below.
The orchestrator only ever calls `get_active()` — it never imports a
specific provider.
"""
from functools import lru_cache
from typing import Callable

from app.config import Settings, get_settings
from app.model_providers.base import ModelProvider
from app.model_providers.claude_provider import ClaudeProvider
from app.model_providers.ollama_provider import OllamaProvider

_BUILDERS: dict[str, Callable[[Settings], ModelProvider]] = {
    "ollama": lambda s: OllamaProvider(
        base_url=s.ollama_base_url,
        chat_model=s.ollama_chat_model,
        timeout=s.request_timeout_seconds,
    ),
    "claude": lambda s: ClaudeProvider(
        api_key=s.anthropic_api_key,
        model=s.anthropic_model,
    ),
}


@lru_cache
def get_active() -> ModelProvider:
    settings = get_settings()
    builder = _BUILDERS.get(settings.chat_model_provider)
    if builder is None:
        raise ValueError(f"Unknown chat model provider: {settings.chat_model_provider!r}")
    return builder(settings)
