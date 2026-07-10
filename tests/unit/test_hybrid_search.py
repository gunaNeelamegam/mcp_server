from typing import Any

from app.rag.search.base import SearchStrategy
from app.rag.search.hybrid import HybridSearch


class FakeStrategy(SearchStrategy):
    def __init__(self, results: list[dict[str, Any]]) -> None:
        self._results = results

    async def search(self, query: str, filters: dict[str, Any], top_k: int) -> list[dict[str, Any]]:
        return self._results[:top_k]


async def test_sole_match_is_not_scored_zero():
    # regression test: min-max normalization used to map a single result's
    # score to 0 (lo == hi), silently dropping it in the weighted merge.
    dense = FakeStrategy([{"id": "a", "score": 0.8, "payload": {"text": "a"}}])
    keyword = FakeStrategy([])
    hybrid = HybridSearch(dense, keyword)

    results = await hybrid.search("query", {}, top_k=5)

    assert len(results) == 1
    assert results[0]["id"] == "a"
    assert results[0]["score"] > 0


async def test_keyword_support_can_lift_a_weaker_dense_candidate():
    # c is the weakest dense match but the only keyword match; it should
    # still outrank b, which has better dense support but no keyword hit.
    dense = FakeStrategy(
        [
            {"id": "a", "score": 0.9, "payload": {"text": "a"}},
            {"id": "b", "score": 0.7, "payload": {"text": "b"}},
            {"id": "c", "score": 0.5, "payload": {"text": "c"}},
        ]
    )
    keyword = FakeStrategy([{"id": "c", "score": 5.0, "payload": {"text": "c"}}])
    hybrid = HybridSearch(dense, keyword)

    results = await hybrid.search("query", {}, top_k=5)
    ids = [r["id"] for r in results]

    assert set(ids) == {"a", "b", "c"}
    assert ids[0] == "a"  # undisputed top dense match, no keyword needed
    assert ids.index("c") < ids.index("b")  # keyword boost lifts c above b


async def test_top_k_is_respected_after_merge():
    dense = FakeStrategy([{"id": str(i), "score": float(i), "payload": {"text": str(i)}} for i in range(10)])
    keyword = FakeStrategy([])
    hybrid = HybridSearch(dense, keyword)

    results = await hybrid.search("query", {}, top_k=3)
    assert len(results) == 3
