from typing import Any

from app.mcp_clients.base import McpClient
from app.mcp_clients.registry import McpRegistry


class FakeMcpClient(McpClient):
    """Minimal in-memory McpClient — proves the registry only depends on
    the interface, never a concrete transport."""

    def __init__(self, tools: list[dict[str, Any]]) -> None:
        self._tools = tools
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def list_tools(self) -> list[dict[str, Any]]:
        return self._tools

    async def call_tool(self, tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
        self.calls.append((tool_name, args))
        return {"ok": True, "tool": tool_name}


def _registry() -> tuple[McpRegistry, FakeMcpClient]:
    client = FakeMcpClient(
        [
            {"name": "getShipmentStatus", "description": "", "input_schema": {}, "safety": "read_only"},
            {"name": "updateShipmentEta", "description": "", "input_schema": {}, "safety": "needs_approval"},
        ]
    )
    return McpRegistry([client]), client


def test_all_tool_schemas_lists_every_client_tool():
    registry, _ = _registry()
    names = {schema["name"] for schema in registry.all_tool_schemas()}
    assert names == {"getShipmentStatus", "updateShipmentEta"}


def test_get_owner_resolves_to_the_right_client():
    registry, client = _registry()
    assert registry.get_owner("getShipmentStatus") is client


def test_safety_tag_round_trips_through_schema_lookup():
    registry, _ = _registry()
    assert registry.get_schema("getShipmentStatus")["safety"] == "read_only"
    assert registry.get_schema("updateShipmentEta")["safety"] == "needs_approval"


async def test_call_tool_delegates_to_owner():
    registry, client = _registry()
    result = await registry.get_owner("getShipmentStatus").call_tool("getShipmentStatus", {"tracking_id": "TRK1001"})
    assert result == {"ok": True, "tool": "getShipmentStatus"}
    assert client.calls == [("getShipmentStatus", {"tracking_id": "TRK1001"})]
