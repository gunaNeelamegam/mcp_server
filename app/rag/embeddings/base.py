"""The EmbeddingProvider contract. Swappable the same way as ModelProvider.

Both ingestion and query-time search must use the SAME embedding
provider/model — that is enforced by reading it from app.config in one
place (app/rag/embeddings/registry.py), never configured separately for
ingest vs. query.
"""
from abc import ABC, abstractmethod


class EmbeddingProvider(ABC):
    @abstractmethod
    async def embed(self, texts: list[str], is_query: bool = False) -> list[list[float]]:
        """Returns one vector per input text. Must batch, not loop one-by-one.

        is_query distinguishes a search query from an indexed document chunk.
        Some models (e.g. nomic-embed-text) are trained with different task
        prefixes for each and produce poorly-calibrated similarity scores
        without them — implementations that don't need the distinction can
        ignore the flag."""
        ...

    @property
    @abstractmethod
    def dimensions(self) -> int:
        """Vector size this provider produces — needed to create the Qdrant collection."""
        ...
