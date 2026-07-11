"""Generic RagSource: one Qdrant collection + one SearchStrategy. Every
domain in domain_sources.py gets one of these — domains never need a
bespoke RagSource implementation."""
from typing import Any

from app.rag.base import RagSource
from app.rag.search.base import SearchStrategy


class CollectionRagSource(RagSource):
    def __init__(self, domain: str, search_strategy: SearchStrategy) -> None:
        self._domain = domain
        self._search_strategy = search_strategy

    async def search(self, query: str, filters: dict[str, Any], top_k: int = 3) -> list[dict[str, Any]]:
        results = await self._search_strategy.search(query, filters, top_k)
        return [
            {
                "domain": self._domain,
                "doc_id": result["payload"].get("doc_id", ""),
                "text": result["payload"].get("text", ""),
                "score": result["score"],
                "metadata": {k: v for k, v in result["payload"].items() if k not in {"doc_id", "text"}},
            }
            for result in results
        ]
