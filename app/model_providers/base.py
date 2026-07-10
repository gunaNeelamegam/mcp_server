"""The ModelProvider contract. Every chat model plugs in through this and
nothing else — the orchestrator never imports a provider SDK directly.

Tool schema convention passed into `generate()` (provider-agnostic, MCP-shaped):
    {
        "name": str,
        "description": str,
        "input_schema": dict,       # JSON schema for the tool's arguments
        "safety": "read_only" | "needs_approval",
    }
Each provider implementation is responsible for translating this into
whatever shape its own SDK/API expects.
"""
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import Any, TypedDict


class GenerateResult(TypedDict, total=False):
    type: str  # "final_answer" | "tool_call" | "text_delta"
    text: str
    tool: str
    args: dict[str, Any]


class ModelProvider(ABC):
    @abstractmethod
    async def generate(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        stream: bool = False,
    ) -> GenerateResult | AsyncIterator[GenerateResult]:
        """
        messages: full conversation history (system/user/assistant/tool turns)
        tools: available tool schemas (see module docstring), already
               filtered by auth for the caller's role
        stream: when False (default), returns a single GenerateResult:
                    {"type": "tool_call", "tool": str, "args": dict}
                 or {"type": "final_answer", "text": str}
                when True, returns an async iterator of GenerateResult:
                    zero or more {"type": "text_delta", "text": str}
                    followed by exactly one terminal result of the same
                    shape as the non-streaming case.
        """
        ...
