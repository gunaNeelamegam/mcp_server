"""Conversation history + pending-confirmation state live outside the
orchestrator (Redis) — the orchestrator itself is stateless per call."""
import json
from functools import lru_cache
from typing import Any

from redis.asyncio import Redis

from app.config import get_settings

_HISTORY_TTL_SECONDS = 60 * 60 * 24  # 1 day
_PENDING_TTL_SECONDS = 60 * 15  # confirmation prompts don't stay valid forever


class SessionStore:
    def __init__(self, redis: Redis) -> None:
        self._redis = redis

    def _history_key(self, user_id: str) -> str:
        return f"session:{user_id}:messages"

    def _pending_key(self, request_id: str) -> str:
        return f"pending:{request_id}"

    async def get_history(self, user_id: str) -> list[dict[str, Any]]:
        raw = await self._redis.get(self._history_key(user_id))
        return json.loads(raw) if raw else []

    async def append(self, user_id: str, message: dict[str, Any]) -> None:
        history = await self.get_history(user_id)
        history.append(message)
        await self._redis.set(self._history_key(user_id), json.dumps(history), ex=_HISTORY_TTL_SECONDS)

    async def save_pending(self, request_id: str, state: dict[str, Any]) -> None:
        await self._redis.set(self._pending_key(request_id), json.dumps(state), ex=_PENDING_TTL_SECONDS)

    async def get_pending(self, request_id: str) -> dict[str, Any] | None:
        raw = await self._redis.get(self._pending_key(request_id))
        return json.loads(raw) if raw else None

    async def clear_pending(self, request_id: str) -> None:
        await self._redis.delete(self._pending_key(request_id))


@lru_cache
def get_store() -> SessionStore:
    settings = get_settings()
    return SessionStore(Redis.from_url(settings.redis_url, decode_responses=True))
