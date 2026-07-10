"""ModelProvider implementation backed by a local Ollama server."""
import json
from collections.abc import AsyncIterator
from typing import Any

import httpx

from app.model_providers.base import GenerateResult, ModelProvider


def _to_ollama_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool.get("description", ""),
                "parameters": tool.get("input_schema", {"type": "object", "properties": {}}),
            },
        }
        for tool in tools
    ]


def _message_to_result(message: dict[str, Any]) -> GenerateResult:
    tool_calls = message.get("tool_calls") or []
    if tool_calls:
        call = tool_calls[0]["function"]
        return {"type": "tool_call", "tool": call["name"], "args": call.get("arguments", {})}
    return {"type": "final_answer", "text": message.get("content", "")}


class OllamaProvider(ModelProvider):
    def __init__(self, base_url: str, chat_model: str, timeout: float = 60.0) -> None:
        self._base_url = base_url.rstrip("/")
        self._chat_model = chat_model
        self._timeout = timeout

    async def generate(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        stream: bool = False,
    ) -> GenerateResult | AsyncIterator[GenerateResult]:
        payload = {
            "model": self._chat_model,
            "messages": messages,
            "tools": _to_ollama_tools(tools),
            "stream": stream,
        }
        if not stream:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.post(f"{self._base_url}/api/chat", json=payload)
                response.raise_for_status()
                return _message_to_result(response.json()["message"])

        return self._stream(payload)

    async def _stream(self, payload: dict[str, Any]) -> AsyncIterator[GenerateResult]:
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            async with client.stream("POST", f"{self._base_url}/api/chat", json=payload) as response:
                response.raise_for_status()
                accumulated = ""
                async for line in response.aiter_lines():
                    if not line:
                        continue
                    chunk = json.loads(line)
                    message = chunk.get("message", {})
                    delta = message.get("content", "")
                    if delta:
                        accumulated += delta
                        yield {"type": "text_delta", "text": delta}
                    if chunk.get("done"):
                        tool_calls = message.get("tool_calls") or []
                        if tool_calls:
                            call = tool_calls[0]["function"]
                            yield {"type": "tool_call", "tool": call["name"], "args": call.get("arguments", {})}
                        else:
                            yield {"type": "final_answer", "text": accumulated}
                        return
