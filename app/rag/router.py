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
