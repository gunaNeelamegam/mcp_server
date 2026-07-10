"""BM25/exact-term matching — handles article numbers, HS codes,
regulation IDs that vector search alone tends to miss.

Builds the BM25 index on demand from the collection's stored chunks.
Fine for the document volumes this system targets (hundreds-thousands of
chunks per domain); if a domain grows large enough for this to matter,
cache/persist the index instead of rebuilding per query."""
import re
from typing import Any

from rank_bm25 import BM25Okapi

from app.rag.search.base import SearchStrategy
from app.rag.vector_store import VectorStore

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class KeywordSearch(SearchStrategy):
    def __init__(self, collection: str, vector_store: VectorStore) -> None:
        self._collection = collection
        self._vector_store = vector_store

    async def search(self, query: str, filters: dict[str, Any], top_k: int) -> list[dict[str, Any]]:
        points = await self._vector_store.scroll_all(self._collection, filters)
        if not points:
            return []

        corpus = [_tokenize(point["payload"].get("text", "")) for point in points]
        bm25 = BM25Okapi(corpus)
        scores = bm25.get_scores(_tokenize(query))

        ranked = sorted(zip(points, scores), key=lambda pair: pair[1], reverse=True)
        return [
            {"id": point["id"], "score": float(score), "payload": point["payload"]}
            for point, score in ranked[:top_k]
            if score > 0
        ]
