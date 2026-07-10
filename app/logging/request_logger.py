"""Every model call, tool call, and RAG query is logged with a shared
request_id so a full conversation can be reconstructed and debugged —
this is what makes "why did it answer this way" answerable later.

Structured JSON to stdout by default (log aggregators pick this up
without extra parsing). Swap the handler in configure() if a different
sink (file, log shipper) is needed later — callers never change.
"""
import json
import logging
import sys
import time
from typing import Any

from app.config import get_settings

_logger = logging.getLogger("logistics_ai.requests")


def configure() -> None:
    if _logger.handlers:
        return
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(message)s"))
    _logger.addHandler(handler)
    _logger.setLevel(get_settings().log_level)
    _logger.propagate = False


def _emit(event: str, request_id: str | None, **fields: Any) -> None:
    configure()
    record = {"event": event, "request_id": request_id, "ts": time.time(), **fields}
    _logger.info(json.dumps(record, default=str))


def log_model_call(request_id: str | None, messages: list[dict[str, Any]], result: dict[str, Any]) -> None:
    _emit("model_call", request_id, message_count=len(messages), result=result)


def log_tool_call(request_id: str | None, tool: str, args: dict[str, Any], result: Any) -> None:
    _emit("tool_call", request_id, tool=tool, args=args, result=result)


def log_rag_query(request_id: str | None, domain: str, question: str, results: list[dict[str, Any]]) -> None:
    _emit(
        "rag_query",
        request_id,
        domain=domain,
        question=question,
        chunk_ids=[r.get("doc_id") for r in results],
        top_score=max((r.get("score", 0.0) for r in results), default=None),
    )
