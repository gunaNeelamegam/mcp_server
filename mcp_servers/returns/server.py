"""Standalone returns/RMA MCP server — same plain-HTTP-JSON pattern as
mcp_servers/shipments/server.py (GET /tools, POST /call_tool).

Run: uv run uvicorn mcp_servers.returns.server:app --port 7005
"""
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="returns-mcp")

_RETURNS: dict[str, dict[str, Any]] = {
    "RMA4001": {
        "rma_id": "RMA4001",
        "order_id": "SO3003",
        "reason": "Damaged item on arrival",
        "status": "requested",
    },
    "RMA4002": {
        "rma_id": "RMA4002",
        "order_id": "SO3001",
        "reason": "Wrong item shipped",
        "status": "approved",
    },
}

_TOOLS = [
    {
        "name": "getReturnStatus",
        "description": "Look up the current status and reason for a return (RMA) by RMA ID.",
        "input_schema": {
            "type": "object",
            "properties": {"rma_id": {"type": "string", "description": "e.g. RMA4001"}},
            "required": ["rma_id"],
        },
        "safety": "read_only",
    },
    {
        "name": "listReturns",
        "description": "List returns (RMAs), optionally filtered by status (requested, approved, rejected, received).",
        "input_schema": {
            "type": "object",
            "properties": {"status": {"type": "string"}},
        },
        "safety": "read_only",
    },
    {
        "name": "approveReturn",
        "description": "Approve a requested return (RMA), moving it to 'approved'. This writes data and requires user confirmation.",
        "input_schema": {
            "type": "object",
            "properties": {"rma_id": {"type": "string"}},
            "required": ["rma_id"],
        },
        "safety": "needs_approval",
    },
]


class CallToolRequest(BaseModel):
    tool: str
    args: dict[str, Any]


@app.get("/tools")
async def list_tools() -> list[dict[str, Any]]:
    return _TOOLS


@app.post("/call_tool")
async def call_tool(request: CallToolRequest) -> dict[str, Any]:
    if request.tool == "getReturnStatus":
        rma_id = request.args.get("rma_id", "")
        rma = _RETURNS.get(rma_id)
        if not rma:
            raise HTTPException(404, f"No return found for {rma_id!r}")
        return rma

    if request.tool == "listReturns":
        status = request.args.get("status")
        returns = list(_RETURNS.values())
        if status:
            returns = [r for r in returns if r["status"] == status]
        return {"returns": returns}

    if request.tool == "approveReturn":
        rma_id = request.args.get("rma_id", "")
        rma = _RETURNS.get(rma_id)
        if not rma:
            raise HTTPException(404, f"No return found for {rma_id!r}")
        rma["status"] = "approved"
        return rma

    raise HTTPException(404, f"Unknown tool {request.tool!r}")
