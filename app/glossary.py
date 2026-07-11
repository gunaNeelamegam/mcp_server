"""Small, stable, structured terminology — NOT part of RAG. Injected
directly into the system prompt (cheap, always available) rather than
retrieved via search. If this grows past a handful of entries, graduate
it into its own RAG domain instead of a giant dict.

NOTE: these prefixes are a single global placeholder set. Different
organizations use different prefix conventions for the same entity
types (e.g. one org's "SO" might be another's "SORD"), so this dict is
expected to be replaced by a per-organization prefix registry, resolved
the same way `app.auth.permissions.get_role` resolves a user's role.
Until then, every org sees the same prefixes."""

ID_FORMATS: dict[str, dict[str, str]] = {
    "TRK": {"meaning": "Shipment Tracking ID", "example": "TRK1001", "regex": r"^TRK\d{4}$"},
    "ORD": {"meaning": "Order ID", "example": "ORD1001", "regex": r"^ORD\d{4}$"},
    "CUST": {"meaning": "Customer ID", "example": "CUST1001", "regex": r"^CUST\d{4}$"},
    "ASN": {"meaning": "Advance Shipping Notice (inbound)", "example": "ASN2001", "regex": r"^ASN\d{4}$"},
    "PO": {"meaning": "Purchase Order", "example": "PO5001", "regex": r"^PO\d{4}$"},
    "SO": {"meaning": "Sales Order (outbound)", "example": "SO3001", "regex": r"^SO\d{4}$"},
    "SKU": {"meaning": "Stock Keeping Unit (inventory item)", "example": "SKU1001", "regex": r"^SKU\d{4}$"},
    "RMA": {"meaning": "Return Merchandise Authorization", "example": "RMA4001", "regex": r"^RMA\d{4}$"},
}


def as_system_prompt_block() -> str:
    lines = ["Internal ID formats:"]
    for prefix, info in ID_FORMATS.items():
        lines.append(f"- {prefix}xxxx = {info['meaning']} (e.g. {info['example']})")
    return "\n".join(lines)
