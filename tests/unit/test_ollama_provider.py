import json

import httpx

from app.model_providers.ollama_provider import OllamaProvider, _message_to_result, _to_ollama_messages, _to_ollama_tools


def _ndjson_transport(chunks: list[dict]) -> httpx.MockTransport:
    body = "".join(json.dumps(chunk) + "\n" for chunk in chunks).encode()

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=body)

    return httpx.MockTransport(handler)


async def _collect_stream(monkeypatch, chunks: list[dict]) -> list[dict]:
    transport = _ndjson_transport(chunks)
    real_client = httpx.AsyncClient

    def fake_client(*args, **kwargs):
        kwargs["transport"] = transport
        return real_client(*args, **kwargs)

    monkeypatch.setattr("app.model_providers.ollama_provider.httpx.AsyncClient", fake_client)

    provider = OllamaProvider(base_url="http://fake-ollama", chat_model="llama3.2")
    stream = await provider.generate([{"role": "user", "content": "hi"}], [], stream=True)
    return [event async for event in stream]


def test_tool_schema_conversion_drops_safety_tag():
    tools = [{"name": "getShipmentStatus", "description": "look up a shipment", "input_schema": {"type": "object"}, "safety": "read_only"}]
    converted = _to_ollama_tools(tools)
    assert converted == [
        {
            "type": "function",
            "function": {"name": "getShipmentStatus", "description": "look up a shipment", "parameters": {"type": "object"}},
        }
    ]


def test_assistant_tool_call_message_becomes_ollama_tool_calls_shape():
    messages = [
        {"role": "user", "content": "where is TRK1001?"},
        {"role": "assistant", "content": "", "tool_call": {"id": "call-1", "name": "getShipmentStatus", "args": {"tracking_id": "TRK1001"}}},
        {"role": "tool", "tool_call_id": "call-1", "content": {"status": "in_transit"}},
    ]
    converted = _to_ollama_messages(messages)

    assert converted[0] == {"role": "user", "content": "where is TRK1001?"}
    assert converted[1]["role"] == "assistant"
    assert converted[1]["tool_calls"] == [{"function": {"name": "getShipmentStatus", "arguments": {"tracking_id": "TRK1001"}}}]
    # dict tool results must be serialized to a string — Ollama's API expects plain string content
    assert converted[2] == {"role": "tool", "content": '{"status": "in_transit"}'}


def test_message_to_result_detects_tool_call():
    message = {"content": "", "tool_calls": [{"function": {"name": "getShipmentStatus", "arguments": {"tracking_id": "TRK1001"}}}]}
    assert _message_to_result(message) == {"type": "tool_call", "tool": "getShipmentStatus", "args": {"tracking_id": "TRK1001"}}


def test_message_to_result_final_answer_when_no_tool_calls():
    message = {"content": "here is your answer"}
    assert _message_to_result(message) == {"type": "final_answer", "text": "here is your answer"}


async def test_stream_captures_tool_call_delivered_on_a_non_terminal_chunk(monkeypatch):
    # Regression test: Ollama sends the full tool_calls array on its own
    # chunk with done: false, then a separate terminal done: true chunk
    # that has no tool_calls at all. An earlier version of _stream() only
    # inspected the done: true chunk and always missed the tool call,
    # silently falling through to an empty final_answer.
    chunks = [
        {
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"function": {"name": "getReturnStatus", "arguments": {"rma_id": "RMA4001"}}}],
            },
            "done": False,
        },
        {"message": {"role": "assistant", "content": ""}, "done": True},
    ]
    events = await _collect_stream(monkeypatch, chunks)
    assert events == [{"type": "tool_call", "tool": "getReturnStatus", "args": {"rma_id": "RMA4001"}}]


async def test_stream_yields_text_deltas_then_final_answer_when_no_tool_call(monkeypatch):
    chunks = [
        {"message": {"role": "assistant", "content": "Hello"}, "done": False},
        {"message": {"role": "assistant", "content": " there"}, "done": False},
        {"message": {"role": "assistant", "content": ""}, "done": True},
    ]
    events = await _collect_stream(monkeypatch, chunks)
    assert events == [
        {"type": "text_delta", "text": "Hello"},
        {"type": "text_delta", "text": " there"},
        {"type": "final_answer", "text": "Hello there"},
    ]
