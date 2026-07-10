"""Thin Qdrant wrapper — the only file in the codebase that imports
qdrant_client directly. Swapping to Chroma/pgvector/Milvus later means
rewriting this file only; RagSource/SearchStrategy implementations never
see the underlying client.

If QDRANT_URL looks like an http(s) URL, connects to a real Qdrant
server. Otherwise treats it as an on-disk path and uses qdrant-client's
embedded local mode — zero extra services needed for local dev.
"""
from functools import lru_cache
from typing import Any

from qdrant_client import AsyncQdrantClient, models

from app.config import get_settings


class VectorStore:
    def __init__(self, client: AsyncQdrantClient) -> None:
        self._client = client

    async def ensure_collection(self, collection: str, vector_size: int) -> None:
        if not await self._client.collection_exists(collection):
            await self._client.create_collection(
                collection_name=collection,
                vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE),
            )

    async def upsert_chunks(self, collection: str, points: list[dict[str, Any]]) -> None:
        """points: [{"id": str, "vector": list[float], "payload": dict}, ...]"""
        await self._client.upsert(
            collection_name=collection,
            points=[
                models.PointStruct(id=point["id"], vector=point["vector"], payload=point["payload"])
                for point in points
            ],
        )

    async def search(
        self, collection: str, query_vector: list[float], filters: dict[str, Any] | None, top_k: int
    ) -> list[dict[str, Any]]:
        if not await self._client.collection_exists(collection):
            return []
        result = await self._client.query_points(
            collection_name=collection,
            query=query_vector,
            query_filter=_build_filter(filters),
            limit=top_k,
        )
        return [{"id": str(p.id), "score": p.score, "payload": p.payload} for p in result.points]

    async def scroll_all(self, collection: str, filters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        if not await self._client.collection_exists(collection):
            return []
        points: list[dict[str, Any]] = []
        offset = None
        while True:
            batch, offset = await self._client.scroll(
                collection_name=collection,
                scroll_filter=_build_filter(filters),
                limit=256,
                offset=offset,
                with_payload=True,
            )
            points.extend({"id": str(p.id), "payload": p.payload} for p in batch)
            if offset is None:
                break
        return points

    async def delete_by_doc_id(self, collection: str, doc_id: str) -> None:
        if not await self._client.collection_exists(collection):
            return
        await self._client.delete(
            collection_name=collection,
            points_selector=models.FilterSelector(
                filter=models.Filter(must=[models.FieldCondition(key="doc_id", match=models.MatchValue(value=doc_id))])
            ),
        )


def _build_filter(filters: dict[str, Any] | None) -> models.Filter | None:
    if not filters:
        return None
    conditions = [models.FieldCondition(key=key, match=models.MatchValue(value=value)) for key, value in filters.items()]
    return models.Filter(must=conditions)


@lru_cache
def get_store() -> VectorStore:
    settings = get_settings()
    if settings.qdrant_url.startswith("http"):
        client = AsyncQdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key or None)
    else:
        client = AsyncQdrantClient(path=settings.qdrant_url)
    return VectorStore(client)
