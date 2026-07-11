"""McpClient implementation that talks to the returns/RMA MCP server over
plain HTTP JSON (see mcp_servers/returns/server.py)."""
from typing import Any

import httpx

from app.mcp_clients.base import McpClient


class ReturnsMcpClient(McpClient):
    def __init__(self, base_url: str, tools: list[dict[str, Any]], timeout: float = 30.0) -> None:
        self._base_url = base_url.rstrip("/")
        self._tools = tools
        self._timeout = timeout

    @classmethod
    async def create(cls, base_url: str, timeout: float = 30.0) -> "ReturnsMcpClient":
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.get(f"{base_url.rstrip('/')}/tools")
            response.raise_for_status()
            tools = response.json()
        return cls(base_url, tools, timeout)

    def list_tools(self) -> list[dict[str, Any]]:
        return self._tools

    async def call_tool(self, tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            response = await client.post(
                f"{self._base_url}/call_tool", json={"tool": tool_name, "args": args}
            )
            response.raise_for_status()
            return response.json()
