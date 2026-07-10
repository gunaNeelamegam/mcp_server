"""Maps domain name -> RagSource instance. Adding a new domain = one
entry in domain_sources.py; it appears here automatically."""
from functools import lru_cache

from app.rag.base import RagSource
from app.rag.collection_source import CollectionRagSource
from app.rag.domain_sources import DOMAINS
from app.rag.embeddings.registry import get_active as get_embedding_provider
from app.rag.search.registry import build_strategy
from app.rag.vector_store import get_store


@lru_cache
def get(domain: str) -> RagSource:
    config = DOMAINS[domain]
    strategy = build_strategy(config.collection, get_store(), get_embedding_provider())
    return CollectionRagSource(domain, strategy)


def all_domains() -> list[str]:
    return list(DOMAINS)
