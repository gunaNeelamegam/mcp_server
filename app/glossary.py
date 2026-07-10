"""Small, stable, structured terminology — NOT part of RAG. Injected
directly into the system prompt (cheap, always available) rather than
retrieved via search. If this grows past a handful of entries, graduate
it into its own RAG domain instead of a giant dict."""

ID_FORMATS: dict[str, dict[str, str]] = {
    "TRK": {"meaning": "Tracking ID", "example": "TRK1001", "regex": r"^TRK\d{4}$"},
    "ORD": {"meaning": "Order ID", "example": "ORD1001", "regex": r"^ORD\d{4}$"},
    "CUST": {"meaning": "Customer ID", "example": "CUST1001", "regex": r"^CUST\d{4}$"},
}


def as_system_prompt_block() -> str:
    lines = ["Internal ID formats:"]
    for prefix, info in ID_FORMATS.items():
        lines.append(f"- {prefix}xxxx = {info['meaning']} (e.g. {info['example']})")
    return "\n".join(lines)
