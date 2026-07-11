"""Standalone inbound/ASN MCP server — same plain-HTTP-JSON pattern as
mcp_servers/shipments/server.py (GET /tools, POST /call_tool).

Run: uv run uvicorn mcp_servers.inbound.server:app --port 7002
"""
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="inbound-mcp")

_ASNS: dict[str, dict[str, Any]] = {
    "ASN2001": {
        "asn_id": "ASN2001",
        "po_number": "PO5001",
        "supplier": "Acme Distribution",
        "expected_date": "2026-07-15",
        "status": "in_transit",
        "items": [{"sku": "SKU1001", "expected_qty": 200, "received_qty": 0}],
    },
    "ASN2002": {
        "asn_id": "ASN2002",
        "po_number": "PO5002",
        "supplier": "Gulf Freight Co",
        "expected_date": "2026-07-10",
        "status": "received",
        "items": [{"sku": "SKU1002", "expected_qty": 50, "received_qty": 50}],
    },
    "ASN2003": {
        "asn_id": "ASN2003",
        "po_number": "PO5003",
        "supplier": "Acme Distribution",
        "expected_date": "2026-07-20",
        "status": "expected",
        "items": [{"sku": "SKU1001", "expected_qty": 100, "received_qty": 0}],
    },
}

_TOOLS = [
    {
        "name": "getAsnStatus",
        "description": "Look up the current status, supplier, expected date and line items for an inbound ASN (Advance Shipping Notice) by ASN ID.",
        "input_schema": {
            "type": "object",
            "properties": {"asn_id": {"type": "string", "description": "e.g. ASN2001"}},
            "required": ["asn_id"],
        },
        "safety": "read_only",
    },
    {
        "name": "listAsns",
        "description": "List inbound ASNs, optionally filtered by status (expected, in_transit, received, partially_received).",
        "input_schema": {
            "type": "object",
            "properties": {"status": {"type": "string"}},
        },
        "safety": "read_only",
    },
    {
        "name": "receiveAsn",
        "description": "Record receipt of goods against an ASN, updating received quantities and status. This writes data and requires user confirmation.",
        "input_schema": {
            "type": "object",
            "properties": {
                "asn_id": {"type": "string"},
                "received_items": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {"sku": {"type": "string"}, "qty": {"type": "integer"}},
                        "required": ["sku", "qty"],
                    },
                },
            },
            "required": ["asn_id", "received_items"],
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
    if request.tool == "getAsnStatus":
        asn_id = request.args.get("asn_id", "")
        asn = _ASNS.get(asn_id)
        if not asn:
            raise HTTPException(404, f"No ASN found for {asn_id!r}")
        return asn

    if request.tool == "listAsns":
        status = request.args.get("status")
        asns = list(_ASNS.values())
        if status:
            asns = [a for a in asns if a["status"] == status]
        return {"asns": asns}

    if request.tool == "receiveAsn":
        asn_id = request.args.get("asn_id", "")
        asn = _ASNS.get(asn_id)
        if not asn:
            raise HTTPException(404, f"No ASN found for {asn_id!r}")
        received_by_sku = {item["sku"]: item["qty"] for item in request.args.get("received_items", [])}
        fully_received = True
        for line in asn["items"]:
            line["received_qty"] += received_by_sku.get(line["sku"], 0)
            if line["received_qty"] < line["expected_qty"]:
                fully_received = False
        asn["status"] = "received" if fully_received else "partially_received"
        return asn

    raise HTTPException(404, f"Unknown tool {request.tool!r}")
