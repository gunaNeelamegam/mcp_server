"""The RagSource contract. Each knowledge domain (compliance, support-docs,
known-issues, future ones) is an independent, pluggable RagSource — its
own collection, own ingestion path, own lifecycle."""
from abc import ABC, abstractmethod
from typing import Any


class RagSource(ABC):
    @abstractmethod
    async def search(self, query: str, filters: dict[str, Any], top_k: int = 5) -> list[dict[str, Any]]:
        """Returns top_k relevant chunks with source metadata:
        [{"domain": str, "doc_id": str, "text": str, "score": float, "metadata": dict}, ...]
        """
        ...
