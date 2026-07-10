"""Pure vector/semantic similarity search."""
from typing import Any

from app.rag.embeddings.base import EmbeddingProvider
from app.rag.search.base import SearchStrategy
from app.rag.vector_store import VectorStore

# Floor on raw cosine similarity. Must be applied here, on the raw score,
# not after HybridSearch's min-max normalization — normalization rescales
# whatever the best candidate is to the top of the range regardless of its
# absolute similarity, so a single weak match would otherwise look as
# "relevant" as a strong one once merged.
#
# This only catches clearly-unrelated pairs; short same-domain sentences
# (e.g. "shipment status" vs. "customs clearance", both logistics jargon)
# can still land close together in embedding space, especially with a
# small embedding model and a small per-country corpus. Real precision
# for compliance queries comes from the country/tag filter (exact match),
# not this threshold — treat this as a coarse floor, and re-tune it
# against tests/rag_eval once a production-sized corpus exists.
_MIN_SIMILARITY = 0.35


class DenseSearch(SearchStrategy):
    def __init__(self, collection: str, vector_store: VectorStore, embedding_provider: EmbeddingProvider) -> None:
        self._collection = collection
        self._vector_store = vector_store
        self._embedding_provider = embedding_provider

    async def search(self, query: str, filters: dict[str, Any], top_k: int) -> list[dict[str, Any]]:
        [query_vector] = await self._embedding_provider.embed([query])
        results = await self._vector_store.search(self._collection, query_vector, filters, top_k)
        return [r for r in results if r["score"] >= _MIN_SIMILARITY]
