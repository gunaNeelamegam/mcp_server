"""Decides which knowledge domain(s) a question should search. Keeps this
simple (keyword/intent rules) — only reach for a model-based classifier
if these rules prove insufficient in practice."""
from typing import Any

from app.rag.domain_sources import DOMAINS

_COMPLIANCE_KEYWORDS = (
    "customs", "import", "export", "duty", "tariff", "hs code", "regulation",
    "compliance", "border", "shipping document", "declaration", "tax",
)
_SUPPORT_KEYWORDS = (
    "how do i", "how to", "bug", "error", "not working", "issue", "broken",
    "fails", "failing", "crash", "stuck", "reset",
)


def route(question: str, tags: dict[str, Any] | None = None) -> list[str]:
    q = question.lower()
    domains: set[str] = set()

    if tags and "country" in tags:
        domains.add("compliance")
    if any(keyword in q for keyword in _COMPLIANCE_KEYWORDS):
        domains.add("compliance")
    if any(keyword in q for keyword in _SUPPORT_KEYWORDS):
        domains.add("support-docs")
        domains.add("known-issues")

    if not domains:
        # ambiguous -> search everything, let scoring/re-ranking sort it out
        domains = set(DOMAINS)

    return list(domains)


def expected_domain_but_no_hits(domains: list[str], chunks: list[dict[str, Any]]) -> bool:
    return bool(domains) and not chunks
