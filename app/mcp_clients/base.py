"""The McpClient contract. Every MCP server connection plugs in through
this — the orchestrator asks the registry "who owns tool X" and calls
`call_tool` on whichever client owns it. It never imports a client directly.

Tool schema returned by `list_tools()` (provider-agnostic, feeds straight
into ModelProvider.generate()):
    {
        "name": str,
        "description": str,
        "input_schema": dict,
        "safety": "read_only" | "needs_approval",
    }

Write-action safety: any tool tagged "needs_approval" must never be
auto-executed by the orchestrator — it returns a confirmation prompt to
the user/frontend first. This tag lives on the tool definition, not as
if/else logic scattered in the orchestrator.
"""
from abc import ABC, abstractmethod
from typing import Any


class McpClient(ABC):
    @abstractmethod
    def list_tools(self) -> list[dict[str, Any]]:
        """Returns tool schemas this server exposes (see module docstring)."""
        ...

    @abstractmethod
    async def call_tool(self, tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
        """Executes the tool, returns the raw result."""
        ...
