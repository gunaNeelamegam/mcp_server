from typing import Any

from app.rag.search.keyword import KeywordSearch, _tokenize


class FakeVectorStore:
    def __init__(self, points: list[dict[str, Any]]) -> None:
        self._points = points

    async def scroll_all(self, collection: str, filters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        return self._points


def test_tokenize_drops_stopwords():
    assert _tokenize("How are you today?") == ["today"]


def test_tokenize_keeps_content_words():
    assert _tokenize("How do I reset a stuck shift") == ["reset", "stuck", "shift"]


async def test_search_returns_nothing_for_a_stopword_only_query():
    # Regression test: confirmed live that "how are you" against a
    # support-docs corpus full of "How to..."/"How do I..." articles
    # scored positive BM25 matches purely on "how"/"are"/"you" — a plain
    # greeting got 5 fabricated "relevant" documents attached to it,
    # bloating the prompt enough to cause a request timeout. A query that
    # tokenizes to nothing (all stopwords) must not match anything.
    points = [
        {"id": "1", "payload": {"text": "How to reset a stuck shift"}},
        {"id": "2", "payload": {"text": "How do I reprint a delivery note"}},
        {"id": "3", "payload": {"text": "How do I merge duplicate customer profiles"}},
    ]
    search = KeywordSearch("docs", FakeVectorStore(points))

    results = await search.search("how are you", {}, top_k=5)

    assert results == []


async def test_search_still_matches_real_content_words():
    # BM25 IDF is degenerate at very small corpus sizes (e.g. a term in
    # exactly 1 of 2 docs can score 0), so this needs >=3 docs to be a
    # meaningful check rather than an artifact of the fixture's size.
    points = [
        {"id": "1", "payload": {"text": "How to reset a stuck shift"}},
        {"id": "2", "payload": {"text": "How do I reprint a delivery note"}},
        {"id": "3", "payload": {"text": "How do I merge duplicate customer profiles"}},
    ]
    search = KeywordSearch("docs", FakeVectorStore(points))

    results = await search.search("how do I reset a stuck shift", {}, top_k=5)

    assert len(results) == 1
    assert results[0]["id"] == "1"
