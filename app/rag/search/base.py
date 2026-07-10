"""The SearchStrategy contract — how a query is matched against stored
chunks within a single collection. Concrete strategies are constructed
per-collection (see registry.py) and swapped via config.search_strategy,
never by branching inside a domain's RagSource."""
from abc import ABC, abstractmethod
from typing import Any


class SearchStrategy(ABC):
    @abstractmethod
    async def search(self, query: str, filters: dict[str, Any], top_k: int) -> list[dict[str, Any]]:
        """Returns [{"id": str, "score": float, "payload": dict}, ...], best first."""
        ...
