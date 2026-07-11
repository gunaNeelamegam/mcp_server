"""Config: knowledge domain -> Qdrant collection + chunking rules +
ingestion folder. Adding a new domain later is one entry here (plus a
routing rule in router.py) — no orchestrator or other-domain changes."""
from dataclasses import dataclass, field


@dataclass(frozen=True)
class DomainConfig:
    name: str
    collection: str
    data_dir: str
    file_types: tuple[str, ...]
    tag_keys: tuple[str, ...] = field(default_factory=tuple)  # extra payload/filter keys, e.g. "country"
    chunk_size: int = 800
    chunk_overlap: int = 120


DOMAINS: dict[str, DomainConfig] = {
    "compliance": DomainConfig(
        name="compliance",
        collection="compliance",
        data_dir="ingestion/data/compliance",
        file_types=(".pdf",),
        tag_keys=("country",),
    ),
    "support-docs": DomainConfig(
        name="support-docs",
        collection="support_docs",
        data_dir="ingestion/data/support-docs",
        file_types=(".md", ".txt"),
    ),
    "known-issues": DomainConfig(
        name="known-issues",
        collection="known_issues",
        data_dir="ingestion/data/known-issues",
        file_types=(".md", ".txt"),
    ),
    "inbound": DomainConfig(
        name="inbound",
        collection="inbound_docs",
        data_dir="ingestion/data/inbound",
        file_types=(".md", ".txt"),
    ),
    "outbound": DomainConfig(
        name="outbound",
        collection="outbound_docs",
        data_dir="ingestion/data/outbound",
        file_types=(".md", ".txt"),
    ),
    "inventory": DomainConfig(
        name="inventory",
        collection="inventory_docs",
        data_dir="ingestion/data/inventory",
        file_types=(".md", ".txt"),
    ),
    "returns": DomainConfig(
        name="returns",
        collection="returns_docs",
        data_dir="ingestion/data/returns",
        file_types=(".md", ".txt"),
    ),
}
