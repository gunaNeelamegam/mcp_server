"""EmbeddingProvider implementation backed by a local Ollama server
(default model: nomic-embed-text).

nomic-embed-text is trained with task prefixes and is poorly calibrated
without them — unrelated text can still score deceptively high on cosine
similarity, silently defeating any downstream relevance threshold. Always
prefix: "search_query: " for the query side, "search_document: " for
indexed chunks. See https://huggingface.co/nomic-ai/nomic-embed-text-v1.5
"""
import httpx

from app.rag.embeddings.base import EmbeddingProvider

_QUERY_PREFIX = "search_query: "
_DOCUMENT_PREFIX = "search_document: "


class OllamaEmbeddingProvider(EmbeddingProvider):
    def __init__(self, base_url: str, model: str, dimensions: int = 768, timeout: float = 60.0) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._dimensions = dimensions
        self._timeout = timeout

    @property
    def dimensions(self) -> int:
        return self._dimensions

    async def embed(self, texts: list[str], is_query: bool = False) -> list[list[float]]:
        if not texts:
            return []
        prefix = _QUERY_PREFIX if is_query else _DOCUMENT_PREFIX
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            response = await client.post(
                f"{self._base_url}/api/embed",
                json={"model": self._model, "input": [prefix + text for text in texts]},
            )
            response.raise_for_status()
            return response.json()["embeddings"]
