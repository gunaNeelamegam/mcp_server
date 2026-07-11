"""Decides which knowledge domain(s) a question should search. Keeps this
simple (keyword/intent rules) — only reach for a model-based classifier
if these rules prove insufficient in practice."""
from dataclasses import dataclass
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

# These use deliberately specific, multi-word "policy/process" phrasing
# rather than bare entity nouns ("return", "rma", "stock") — a bare noun
# would also match a live-data tool question like "status of return
# RMA4001", and RAG confidently matching with zero hits short-circuits
# the request into a refusal before the tool-calling step ever runs
# (see expected_domain_but_no_hits). Keep new domain keywords narrow.
_INBOUND_KEYWORDS = (
    "asn receiving", "advance shipping notice", "receiving process", "cross-dock",
    "cross dock", "inbound dock", "supplier submission", "quality inspection hold",
    "short-shipped", "over-received", "over received", "putaway",
)
_OUTBOUND_KEYWORDS = (
    "pick list", "picking process", "packing guideline", "cartonization", "backorder",
    "fulfillment cutoff", "allocation rule", "allocation policy", "split shipment",
    "short-pick", "short pick",
)
_INVENTORY_KEYWORDS = (
    "cycle count", "stock adjustment", "negative stock", "bin location",
    "reconciliation", "safety stock", "reorder policy", "quarantine", "lost inventory",
)
_RETURNS_KEYWORDS = (
    "rma approval", "return policy", "restocking", "refund policy", "return fraud",
    "return receiving", "aged rma", "return shipping label", "exchange policy",
)


@dataclass(frozen=True)
class RoutingDecision:
    domains: list[str]
    # False when nothing matched and we fell back to searching every domain
    # speculatively — e.g. a pure tool question like "status of TRK1001" has
    # no relevant domain at all, so a miss there shouldn't trigger a refusal.
    confident: bool


def route(question: str, tags: dict[str, Any] | None = None) -> RoutingDecision:
    q = question.lower()
    domains: set[str] = set()

    if tags and "country" in tags:
        domains.add("compliance")
    if any(keyword in q for keyword in _COMPLIANCE_KEYWORDS):
        domains.add("compliance")
    if any(keyword in q for keyword in _SUPPORT_KEYWORDS):
        domains.add("support-docs")
        domains.add("known-issues")
    if any(keyword in q for keyword in _INBOUND_KEYWORDS):
        domains.add("inbound")
    if any(keyword in q for keyword in _OUTBOUND_KEYWORDS):
        domains.add("outbound")
    if any(keyword in q for keyword in _INVENTORY_KEYWORDS):
        domains.add("inventory")
    if any(keyword in q for keyword in _RETURNS_KEYWORDS):
        domains.add("returns")

    if domains:
        return RoutingDecision(domains=list(domains), confident=True)

    # ambiguous -> search everything, let scoring/re-ranking sort it out,
    # but don't treat a miss here as "documentation should have existed"
    return RoutingDecision(domains=list(DOMAINS), confident=False)


def expected_domain_but_no_hits(decision: RoutingDecision, chunks: list[dict[str, Any]]) -> bool:
    # `confident` alone is enough: if the question confidently matched a
    # domain but chunks ended up empty — whether from zero search hits or
    # auth filtering the domain out entirely — the honest answer is the
    # same "nothing to show" refusal either way.
    return decision.confident and not chunks
