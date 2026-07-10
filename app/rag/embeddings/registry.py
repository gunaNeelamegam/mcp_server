"""Maps config.embedding_provider to an EmbeddingProvider instance.

Single source of truth read by BOTH ingestion and query-time search, so
they can never silently drift onto different embedding models.
"""
from functools import lru_cache

from app.config import get_settings
from app.rag.embeddings.base import EmbeddingProvider
from app.rag.embeddings.ollama_embedding import OllamaEmbeddingProvider


@lru_cache
def get_active() -> EmbeddingProvider:
    settings = get_settings()
    if settings.embedding_provider == "ollama":
        return OllamaEmbeddingProvider(
            base_url=settings.ollama_base_url,
            model=settings.ollama_embedding_model,
        )
    raise ValueError(f"Unknown embedding provider: {settings.embedding_provider!r}")
