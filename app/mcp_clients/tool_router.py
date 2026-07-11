"""Decides which MCP tools are relevant to a question — the same
keyword/intent-rule pattern as app/rag/router.py, applied to tools
instead of RAG domains.

Necessary because small local models become unreliable once too many
tool schemas are attached to a single call: confirmed live that
llama3.2:3b calls the correct tool reliably when only the relevant one
is offered, but with all tools attached it sometimes hallucinates an
unrelated tool call (e.g. for a plain "hi") and sometimes returns an
empty answer instead of calling the obviously-relevant tool. Narrowing
the tool list before it reaches the model fixes the root cause instead
of retrying or switching providers.
"""
import re
from dataclasses import dataclass
from typing import Any

_TOOL_GROUPS: dict[str, frozenset[str]] = {
    "shipments": frozenset({"getShipmentStatus", "listShipments", "updateShipmentEta"}),
    "inbound": frozenset({"getAsnStatus", "listAsns", "receiveAsn"}),
    "outbound": frozenset({"getSalesOrderStatus", "listSalesOrders", "allocateInventory", "createPickList"}),
    "inventory": frozenset({"getStockLevel", "listLowStock", "adjustStock"}),
    "returns": frozenset({"getReturnStatus", "listReturns", "approveReturn"}),
}

# A glossary ID prefix appearing in the question (e.g. "TRK1001") is a
# much stronger signal than keywords — checked first, before falling
# back to looser keyword matching.
_PREFIX_TO_GROUP: dict[str, str] = {
    "TRK": "shipments",
    "ASN": "inbound",
    "SO": "outbound",
    "SKU": "inventory",
    "RMA": "returns",
}

_KEYWORDS: dict[str, tuple[str, ...]] = {
    "shipments": ("shipment", "tracking", "eta", "delivery", "delivered", "in transit", "carrier"),
    "inbound": ("asn", "inbound", "receiving", "receive goods", "purchase order", "supplier"),
    "outbound": ("sales order", "outbound", "allocate", "pick list", "picking", "fulfillment", "fulfill"),
    "inventory": ("stock", "inventory", "on hand", "on-hand", "warehouse", "cycle count"),
    "returns": ("return", "rma", "refund"),
}


@dataclass(frozen=True)
class ToolRoutingDecision:
    groups: frozenset[str]
    # False when nothing matched and we fell back to offering every
    # group speculatively (e.g. genuinely ambiguous small talk) — mirrors
    # app.rag.router.RoutingDecision.confident.
    confident: bool


def route(question: str) -> ToolRoutingDecision:
    groups: set[str] = set()

    for prefix, group in _PREFIX_TO_GROUP.items():
        if re.search(rf"\b{re.escape(prefix)}\d+\b", question, re.IGNORECASE):
            groups.add(group)

    q = question.lower()
    for group, keywords in _KEYWORDS.items():
        if any(keyword in q for keyword in keywords):
            groups.add(group)

    if groups:
        return ToolRoutingDecision(groups=frozenset(groups), confident=True)
    return ToolRoutingDecision(groups=frozenset(_TOOL_GROUPS), confident=False)


def filter_tools(decision: ToolRoutingDecision, tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    allowed_names: set[str] = set()
    for group in decision.groups:
        allowed_names |= _TOOL_GROUPS.get(group, frozenset())

    # Tools that don't belong to any known group (e.g. a newly added
    # domain not yet classified here) are safer to always include than
    # to silently hide — falling back to "unclassified" rather than
    # "invisible".
    known_names = {name for names in _TOOL_GROUPS.values() for name in names}
    return [t for t in tools if t["name"] in allowed_names or t["name"] not in known_names]
