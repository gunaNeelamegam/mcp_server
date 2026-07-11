"""Standalone outbound/sales-order MCP server — same plain-HTTP-JSON
pattern as mcp_servers/shipments/server.py (GET /tools, POST /call_tool).

Run: uv run uvicorn mcp_servers.outbound.server:app --port 7003
"""
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="outbound-mcp")

_SALES_ORDERS: dict[str, dict[str, Any]] = {
    "SO3001": {
        "order_id": "SO3001",
        "customer_id": "CUST1001",
        "status": "pending",
        "items": [{"sku": "SKU1001", "qty": 20}],
    },
    "SO3002": {
        "order_id": "SO3002",
        "customer_id": "CUST1002",
        "status": "allocated",
        "items": [{"sku": "SKU1002", "qty": 5}],
    },
    "SO3003": {
        "order_id": "SO3003",
        "customer_id": "CUST1001",
        "status": "shipped",
        "items": [{"sku": "SKU1001", "qty": 10}],
    },
}

_TOOLS = [
    {
        "name": "getSalesOrderStatus",
        "description": "Look up the current status, customer and line items for a sales order by order ID.",
        "input_schema": {
            "type": "object",
            "properties": {"order_id": {"type": "string", "description": "e.g. SO3001"}},
            "required": ["order_id"],
        },
        "safety": "read_only",
    },
    {
        "name": "listSalesOrders",
        "description": "List sales orders, optionally filtered by status (pending, allocated, picked, packed, shipped, cancelled).",
        "input_schema": {
            "type": "object",
            "properties": {"status": {"type": "string"}},
        },
        "safety": "read_only",
    },
    {
        "name": "allocateInventory",
        "description": "Allocate on-hand inventory to a pending sales order, moving it to 'allocated'. This writes data and requires user confirmation.",
        "input_schema": {
            "type": "object",
            "properties": {"order_id": {"type": "string"}},
            "required": ["order_id"],
        },
        "safety": "needs_approval",
    },
    {
        "name": "createPickList",
        "description": "Generate a warehouse pick list for an allocated sales order, moving it to 'picked'. This writes data and requires user confirmation.",
        "input_schema": {
            "type": "object",
            "properties": {"order_id": {"type": "string"}},
            "required": ["order_id"],
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
    if request.tool == "getSalesOrderStatus":
        order_id = request.args.get("order_id", "")
        order = _SALES_ORDERS.get(order_id)
        if not order:
            raise HTTPException(404, f"No sales order found for {order_id!r}")
        return order

    if request.tool == "listSalesOrders":
        status = request.args.get("status")
        orders = list(_SALES_ORDERS.values())
        if status:
            orders = [o for o in orders if o["status"] == status]
        return {"sales_orders": orders}

    if request.tool == "allocateInventory":
        order_id = request.args.get("order_id", "")
        order = _SALES_ORDERS.get(order_id)
        if not order:
            raise HTTPException(404, f"No sales order found for {order_id!r}")
        order["status"] = "allocated"
        return order

    if request.tool == "createPickList":
        order_id = request.args.get("order_id", "")
        order = _SALES_ORDERS.get(order_id)
        if not order:
            raise HTTPException(404, f"No sales order found for {order_id!r}")
        order["status"] = "picked"
        return {"order_id": order_id, "pick_list": order["items"]}

    raise HTTPException(404, f"Unknown tool {request.tool!r}")
