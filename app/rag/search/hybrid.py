"""Runs dense + keyword search concurrently, normalizes each score list to
[0, 1], and merges by weighted sum — the default strategy. Compliance
documents mix natural-language meaning with exact codes/citations, so
neither pure dense nor pure keyword alone is reliable."""
import asyncio
from typing import Any

from app.rag.search.base import SearchStrategy
from app.rag.search.dense import DenseSearch
from app.rag.search.keyword import KeywordSearch

_DENSE_WEIGHT = 0.6
_KEYWORD_WEIGHT = 0.4


def _normalize(results: list[dict[str, Any]]) -> dict[str, float]:
    if not results:
        return {}
    scores = [r["score"] for r in results]
    lo, hi = min(scores), max(scores)
    if hi == lo:
        # all tied (including the single-result case) -> treat as top score,
        # not zero, so a sole match isn't silently dropped by the merge
        return {r["id"]: 1.0 for r in results}
    spread = hi - lo
    return {r["id"]: (r["score"] - lo) / spread for r in results}


class HybridSearch(SearchStrategy):
    def __init__(self, dense: DenseSearch, keyword: KeywordSearch) -> None:
        self._dense = dense
        self._keyword = keyword

    async def search(self, query: str, filters: dict[str, Any], top_k: int) -> list[dict[str, Any]]:
        # over-fetch each side so merging/re-ranking has enough candidates
        dense_results, keyword_results = await asyncio.gather(
            self._dense.search(query, filters, top_k * 2),
            self._keyword.search(query, filters, top_k * 2),
        )

        payload_by_id = {r["id"]: r["payload"] for r in dense_results + keyword_results}
        dense_norm = _normalize(dense_results)
        keyword_norm = _normalize(keyword_results)

        combined_ids = set(dense_norm) | set(keyword_norm)
        merged = [
            {
                "id": chunk_id,
                "score": _DENSE_WEIGHT * dense_norm.get(chunk_id, 0.0)
                + _KEYWORD_WEIGHT * keyword_norm.get(chunk_id, 0.0),
                "payload": payload_by_id[chunk_id],
            }
            for chunk_id in combined_ids
        ]
        merged.sort(key=lambda r: r["score"], reverse=True)
        return merged[:top_k]
