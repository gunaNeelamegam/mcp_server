"""ModelProvider implementation backed by the Claude API.

Message convention this provider expects (see app/orchestrator/engine.py):
    {"role": "system", "content": str}
    {"role": "user", "content": str}
    {"role": "assistant", "content": str}
    {"role": "assistant", "content": "", "tool_call": {"id": str, "name": str, "args": dict}}
    {"role": "tool", "tool_call_id": str, "content": str}
"""
from collections.abc import AsyncIterator
from typing import Any

import anthropic

from app.model_providers.base import GenerateResult, ModelProvider


def _to_claude_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "name": tool["name"],
            "description": tool.get("description", ""),
            "input_schema": tool.get("input_schema", {"type": "object", "properties": {}}),
        }
        for tool in tools
    ]


def _to_claude_messages(messages: list[dict[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
    system_parts: list[str] = []
    claude_messages: list[dict[str, Any]] = []

    for message in messages:
        role = message["role"]
        if role == "system":
            system_parts.append(message["content"])
        elif role == "user":
            claude_messages.append({"role": "user", "content": message["content"]})
        elif role == "assistant" and message.get("tool_call"):
            call = message["tool_call"]
            claude_messages.append(
                {
                    "role": "assistant",
                    "content": [
                        {"type": "tool_use", "id": call["id"], "name": call["name"], "input": call["args"]}
                    ],
                }
            )
        elif role == "assistant":
            claude_messages.append({"role": "assistant", "content": message["content"]})
        elif role == "tool":
            claude_messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": message["tool_call_id"],
                            "content": str(message["content"]),
                        }
                    ],
                }
            )
        else:
            raise ValueError(f"Unknown message role: {role!r}")

    return "\n".join(system_parts), claude_messages


def _response_to_result(response: "anthropic.types.Message") -> GenerateResult:
    for block in response.content:
        if block.type == "tool_use":
            return {"type": "tool_call", "tool": block.name, "args": block.input, "id": block.id}  # type: ignore[typeddict-item]
    text = "".join(block.text for block in response.content if block.type == "text")
    return {"type": "final_answer", "text": text}


class ClaudeProvider(ModelProvider):
    def __init__(self, api_key: str, model: str, max_tokens: int = 4096) -> None:
        self._client = anthropic.AsyncAnthropic(api_key=api_key)
        self._model = model
        self._max_tokens = max_tokens

    async def generate(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        stream: bool = False,
    ) -> GenerateResult | AsyncIterator[GenerateResult]:
        system, claude_messages = _to_claude_messages(messages)
        claude_tools = _to_claude_tools(tools)

        if not stream:
            response = await self._client.messages.create(
                model=self._model,
                max_tokens=self._max_tokens,
                system=system,
                messages=claude_messages,
                tools=claude_tools,
            )
            return _response_to_result(response)

        return self._stream(system, claude_messages, claude_tools)

    async def _stream(
        self, system: str, claude_messages: list[dict[str, Any]], claude_tools: list[dict[str, Any]]
    ) -> AsyncIterator[GenerateResult]:
        async with self._client.messages.stream(
            model=self._model,
            max_tokens=self._max_tokens,
            system=system,
            messages=claude_messages,
            tools=claude_tools,
        ) as stream:
            async for text in stream.text_stream:
                yield {"type": "text_delta", "text": text}
            final = await stream.get_final_message()
            yield _response_to_result(final)
