"""Standalone inventory/stock MCP server — same plain-HTTP-JSON pattern as
mcp_servers/shipments/server.py (GET /tools, POST /call_tool).

Run: uv run uvicorn mcp_servers.inventory.server:app --port 7004
"""
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="inventory-mcp")

_STOCK: dict[str, dict[str, Any]] = {
    "SKU1001": {"sku": "SKU1001", "location": "DXB-A1", "quantity_on_hand": 180, "quantity_allocated": 20},
    "SKU1002": {"sku": "SKU1002", "location": "DXB-B3", "quantity_on_hand": 45, "quantity_allocated": 5},
    "SKU1003": {"sku": "SKU1003", "location": "DXB-A2", "quantity_on_hand": 8, "quantity_allocated": 0},
}

_TOOLS = [
    {
        "name": "getStockLevel",
        "description": "Look up on-hand and allocated quantity for a SKU by its identifier.",
        "input_schema": {
            "type": "object",
            "properties": {"sku": {"type": "string", "description": "e.g. SKU1001"}},
            "required": ["sku"],
        },
        "safety": "read_only",
    },
    {
        "name": "listLowStock",
        "description": "List SKUs whose available quantity (on_hand - allocated) is at or below a threshold (default 10).",
        "input_schema": {
            "type": "object",
            "properties": {"threshold": {"type": "integer"}},
        },
        "safety": "read_only",
    },
    {
        "name": "adjustStock",
        "description": "Apply a manual quantity adjustment (positive or negative) to a SKU's on-hand stock, e.g. after a cycle count. This writes data and requires user confirmation.",
        "input_schema": {
            "type": "object",
            "properties": {
                "sku": {"type": "string"},
                "delta": {"type": "integer", "description": "Signed adjustment, e.g. -3 or 12"},
                "reason": {"type": "string"},
            },
            "required": ["sku", "delta", "reason"],
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
    if request.tool == "getStockLevel":
        sku = request.args.get("sku", "")
        item = _STOCK.get(sku)
        if not item:
            raise HTTPException(404, f"No stock record found for {sku!r}")
        return item

    if request.tool == "listLowStock":
        threshold = int(request.args.get("threshold", 10))
        low = [
            item for item in _STOCK.values()
            if (item["quantity_on_hand"] - item["quantity_allocated"]) <= threshold
        ]
        return {"low_stock": low, "threshold": threshold}

    if request.tool == "adjustStock":
        sku = request.args.get("sku", "")
        item = _STOCK.get(sku)
        if not item:
            raise HTTPException(404, f"No stock record found for {sku!r}")
        item["quantity_on_hand"] += int(request.args.get("delta", 0))
        return item

    raise HTTPException(404, f"Unknown tool {request.tool!r}")
