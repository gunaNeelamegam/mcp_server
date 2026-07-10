"""Builds the active SearchStrategy (config.search_strategy) for a given
collection. One place to swap dense/keyword/hybrid — domains never
branch on this themselves."""
from app.config import get_settings
from app.rag.embeddings.base import EmbeddingProvider
from app.rag.search.base import SearchStrategy
from app.rag.search.dense import DenseSearch
from app.rag.search.hybrid import HybridSearch
from app.rag.search.keyword import KeywordSearch
from app.rag.vector_store import VectorStore


def build_strategy(collection: str, vector_store: VectorStore, embedding_provider: EmbeddingProvider) -> SearchStrategy:
    strategy = get_settings().search_strategy
    dense = DenseSearch(collection, vector_store, embedding_provider)
    keyword = KeywordSearch(collection, vector_store)

    if strategy == "dense":
        return dense
    if strategy == "keyword":
        return keyword
    if strategy == "hybrid":
        return HybridSearch(dense, keyword)
    raise ValueError(f"Unknown search strategy: {strategy!r}")
