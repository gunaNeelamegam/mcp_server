"""Pydantic request/response models for the FastAPI layer."""
from typing import Any, Literal

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    user_id: str
    question: str
    tags: dict[str, str] = Field(default_factory=dict)
    request_id: str | None = None
    stream: bool = False


class SourceChunk(BaseModel):
    domain: str
    doc_id: str
    text: str
    score: float
    metadata: dict[str, Any] = Field(default_factory=dict)


class ConfirmationRequired(BaseModel):
    type: Literal["confirmation_required"] = "confirmation_required"
    tool: str
    args: dict[str, Any]
    request_id: str


class ChatResponse(BaseModel):
    type: Literal["final_answer"] = "final_answer"
    text: str
    sources: list[SourceChunk] = Field(default_factory=list)
    request_id: str


class ToolConfirmRequest(BaseModel):
    user_id: str
    request_id: str
    tool: str
    args: dict[str, Any]
    approved: bool
