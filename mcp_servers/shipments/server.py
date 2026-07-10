"""Standalone shipments MCP server.

Exposes tools over plain HTTP JSON (GET /tools, POST /call_tool) so
app/mcp_clients/shipments_client.py can connect without any MCP-protocol
transport machinery. Swap the transport later without touching the
McpClient interface it implements against.

Run: uv run uvicorn mcp_servers.shipments.server:app --port 7001
"""
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="shipments-mcp")

_SHIPMENTS: dict[str, dict[str, Any]] = {
    "TRK1001": {"tracking_id": "TRK1001", "status": "in_transit", "eta": "2026-07-14", "origin": "Dubai", "destination": "Riyadh"},
    "TRK1002": {"tracking_id": "TRK1002", "status": "delivered", "eta": "2026-07-08", "origin": "Doha", "destination": "Dubai"},
    "TRK1003": {"tracking_id": "TRK1003", "status": "delayed", "eta": "2026-07-16", "origin": "Riyadh", "destination": "Doha"},
}

_TOOLS = [
    {
        "name": "getShipmentStatus",
        "description": "Look up the current status, ETA, origin and destination for a shipment by tracking ID.",
        "input_schema": {
            "type": "object",
            "properties": {"tracking_id": {"type": "string", "description": "e.g. TRK1001"}},
            "required": ["tracking_id"],
        },
        "safety": "read_only",
    },
    {
        "name": "listShipments",
        "description": "List shipments, optionally filtered by status (in_transit, delivered, delayed).",
        "input_schema": {
            "type": "object",
            "properties": {"status": {"type": "string"}},
        },
        "safety": "read_only",
    },
    {
        "name": "updateShipmentEta",
        "description": "Update the ETA for a shipment. This writes data and requires user confirmation.",
        "input_schema": {
            "type": "object",
            "properties": {
                "tracking_id": {"type": "string"},
                "new_eta": {"type": "string", "description": "ISO date, e.g. 2026-07-20"},
            },
            "required": ["tracking_id", "new_eta"],
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
    if request.tool == "getShipmentStatus":
        tracking_id = request.args.get("tracking_id", "")
        shipment = _SHIPMENTS.get(tracking_id)
        if not shipment:
            raise HTTPException(404, f"No shipment found for {tracking_id!r}")
        return shipment

    if request.tool == "listShipments":
        status = request.args.get("status")
        shipments = list(_SHIPMENTS.values())
        if status:
            shipments = [s for s in shipments if s["status"] == status]
        return {"shipments": shipments}

    if request.tool == "updateShipmentEta":
        tracking_id = request.args.get("tracking_id", "")
        shipment = _SHIPMENTS.get(tracking_id)
        if not shipment:
            raise HTTPException(404, f"No shipment found for {tracking_id!r}")
        shipment["eta"] = request.args.get("new_eta", shipment["eta"])
        return shipment

    raise HTTPException(404, f"Unknown tool {request.tool!r}")
