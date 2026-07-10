"""Pure vector/semantic similarity search."""
from typing import Any

from app.rag.embeddings.base import EmbeddingProvider
from app.rag.search.base import SearchStrategy
from app.rag.vector_store import VectorStore


class DenseSearch(SearchStrategy):
    def __init__(self, collection: str, vector_store: VectorStore, embedding_provider: EmbeddingProvider) -> None:
        self._collection = collection
        self._vector_store = vector_store
        self._embedding_provider = embedding_provider

    async def search(self, query: str, filters: dict[str, Any], top_k: int) -> list[dict[str, Any]]:
        [query_vector] = await self._embedding_provider.embed([query])
        return await self._vector_store.search(self._collection, query_vector, filters, top_k)
