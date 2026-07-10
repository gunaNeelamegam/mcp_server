from app.model_providers.ollama_provider import _message_to_result, _to_ollama_messages, _to_ollama_tools


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
