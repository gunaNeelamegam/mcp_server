"""POST /chat (supports streaming) and POST /chat/confirm. Routes only
translate HTTP <-> orchestrator calls — no business logic lives here."""
import json
from typing import Any

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from app.orchestrator import engine
from app.schemas import ChatRequest, ToolConfirmRequest

router = APIRouter()


def _sse(event: dict[str, Any]) -> str:
    return f"data: {json.dumps(event)}\n\n"


async def _stream_events(request: ChatRequest):
    async for event in engine.handle_question_stream(
        request.user_id, request.question, tags=request.tags, request_id=request.request_id
    ):
        yield _sse(event)


@router.post("/chat")
async def chat(request: ChatRequest) -> Any:
    if request.stream:
        return StreamingResponse(_stream_events(request), media_type="text/event-stream")

    return await engine.handle_question(
        request.user_id, request.question, tags=request.tags, request_id=request.request_id
    )


@router.post("/chat/confirm")
async def confirm(request: ToolConfirmRequest) -> Any:
    return await engine.resume_after_confirmation(request.request_id, request.approved)
