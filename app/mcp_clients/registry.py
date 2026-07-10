"""Maps each tool name to the McpClient instance that owns it.

Adding a new MCP server (e.g. warehouse-mcp) = one new client file +
one line in `_build_clients()` below. No orchestrator changes — the
orchestrator only ever calls `get_registry()` and then `get_owner()` /
`get_schema()` / `all_tool_schemas()`.
"""
from typing import Any

from app.config import Settings, get_settings
from app.mcp_clients.base import McpClient
from app.mcp_clients.shipments_client import ShipmentsMcpClient


class McpRegistry:
    def __init__(self, clients: list[McpClient]) -> None:
        self._owner_by_tool: dict[str, McpClient] = {}
        self._schema_by_tool: dict[str, dict[str, Any]] = {}
        for client in clients:
            for schema in client.list_tools():
                self._owner_by_tool[schema["name"]] = client
                self._schema_by_tool[schema["name"]] = schema

    def all_tool_schemas(self) -> list[dict[str, Any]]:
        return list(self._schema_by_tool.values())

    def get_schema(self, tool_name: str) -> dict[str, Any]:
        return self._schema_by_tool[tool_name]

    def get_owner(self, tool_name: str) -> McpClient:
        return self._owner_by_tool[tool_name]


async def _build_clients(settings: Settings) -> list[McpClient]:
    shipments = await ShipmentsMcpClient.create(settings.shipments_mcp_url)
    return [shipments]
    # Adding warehouse-mcp later: append
    #   await WarehouseMcpClient.create(settings.warehouse_mcp_url)


_registry: McpRegistry | None = None


async def get_registry() -> McpRegistry:
    global _registry
    if _registry is None:
        settings = get_settings()
        _registry = McpRegistry(await _build_clients(settings))
    return _registry
