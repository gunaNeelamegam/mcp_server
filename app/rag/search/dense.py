"""Pure vector/semantic similarity search."""
from typing import Any

from app.rag.embeddings.base import EmbeddingProvider
from app.rag.search.base import SearchStrategy
from app.rag.vector_store import VectorStore

# Floor on raw cosine similarity. Must be applied here, on the raw score,
# not after HybridSearch's min-max normalization — normalization rescales
# whatever the best candidate is to the top of the range regardless of its
# absolute similarity, so a single weak match would otherwise look as
# "relevant" as a strong one once merged. This matters most for small/
# single-chunk collections, where normalization has nothing else to scale
# against and always maps the sole hit to 1.0 regardless of magnitude.
#
# Calibrated empirically against nomic-embed-text (with the search_query/
# search_document prefixes applied in OllamaEmbeddingProvider) using the
# tests/rag_eval fixtures: true matches scored 0.82-0.85 raw cosine,
# unrelated cross-domain/tool-only questions scored 0.41-0.62. 0.65 sits
# in the gap between them. Re-tune against tests/rag_eval if the corpus
# or embedding model changes.
_MIN_SIMILARITY = 0.65


class DenseSearch(SearchStrategy):
    def __init__(self, collection: str, vector_store: VectorStore, embedding_provider: EmbeddingProvider) -> None:
        self._collection = collection
        self._vector_store = vector_store
        self._embedding_provider = embedding_provider

    async def search(self, query: str, filters: dict[str, Any], top_k: int) -> list[dict[str, Any]]:
        [query_vector] = await self._embedding_provider.embed([query], is_query=True)
        results = await self._vector_store.search(self._collection, query_vector, filters, top_k)
        return [r for r in results if r["score"] >= _MIN_SIMILARITY]
