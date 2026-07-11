"""The RagSource contract. Each knowledge domain (compliance, support-docs,
known-issues, future ones) is an independent, pluggable RagSource — its
own collection, own ingestion path, own lifecycle."""
from abc import ABC, abstractmethod
from typing import Any


class RagSource(ABC):
    @abstractmethod
    async def search(self, query: str, filters: dict[str, Any], top_k: int = 3) -> list[dict[str, Any]]:
        """Returns top_k relevant chunks with source metadata:
        [{"domain": str, "doc_id": str, "text": str, "score": float, "metadata": dict}, ...]

        Every returned chunk gets injected verbatim into the system
        prompt (see orchestrator.engine._build_system_prompt) with no
        further capping, and a question can match multiple domains at
        once — top_k=5 (the old default) was confirmed live to produce
        prompts large enough to exceed the local model's request timeout
        on CPU inference. Keep this conservative.
        """
        ...
