from app.model_providers.claude_provider import _to_claude_messages, _to_claude_tools


def test_tool_schema_conversion_drops_safety_tag():
    tools = [{"name": "getShipmentStatus", "description": "look up a shipment", "input_schema": {"type": "object"}, "safety": "needs_approval"}]
    converted = _to_claude_tools(tools)
    assert converted == [{"name": "getShipmentStatus", "description": "look up a shipment", "input_schema": {"type": "object"}}]


def test_system_messages_are_extracted_and_joined():
    messages = [
        {"role": "system", "content": "You are a logistics assistant."},
        {"role": "system", "content": "Be concise."},
        {"role": "user", "content": "hi"},
    ]
    system, claude_messages = _to_claude_messages(messages)
    assert system == "You are a logistics assistant.\nBe concise."
    assert claude_messages == [{"role": "user", "content": "hi"}]


def test_tool_call_and_result_round_trip():
    messages = [
        {"role": "user", "content": "where is TRK1001?"},
        {"role": "assistant", "content": "", "tool_call": {"id": "call-1", "name": "getShipmentStatus", "args": {"tracking_id": "TRK1001"}}},
        {"role": "tool", "tool_call_id": "call-1", "content": {"status": "in_transit"}},
    ]
    _, claude_messages = _to_claude_messages(messages)

    assert claude_messages[1] == {
        "role": "assistant",
        "content": [{"type": "tool_use", "id": "call-1", "name": "getShipmentStatus", "input": {"tracking_id": "TRK1001"}}],
    }
    assert claude_messages[2] == {
        "role": "user",
        "content": [{"type": "tool_result", "tool_use_id": "call-1", "content": "{'status': 'in_transit'}"}],
    }


def test_unknown_role_raises():
    import pytest

    with pytest.raises(ValueError):
        _to_claude_messages([{"role": "bogus", "content": "x"}])
