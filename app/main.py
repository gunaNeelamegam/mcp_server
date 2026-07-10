"""FastAPI entrypoint. Cross-cutting concerns (request timing/logging)
live here as middleware — never scattered inside orchestrator or
plug-in business logic."""
import logging
import time

from fastapi import FastAPI, Request

from app.api.routes.chat import router as chat_router
from app.logging.request_logger import configure as configure_logging

configure_logging()
_access_log = logging.getLogger("logistics_ai.access")

app = FastAPI(title="Logistics AI System")
app.include_router(chat_router)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.monotonic()
    response = await call_next(request)
    duration_ms = round((time.monotonic() - start) * 1000, 1)
    _access_log.info(
        '"%s %s" %d %sms', request.method, request.url.path, response.status_code, duration_ms
    )
    return response


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}
