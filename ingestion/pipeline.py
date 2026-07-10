"""Shared embed+chunk+upsert logic used by every ingest subcommand
(add/sync/rebuild). doc_id is the file's path relative to the repo root —
stable across runs, human-readable in logs/manifest, and used both as the
Qdrant payload's doc_id and to derive per-chunk point IDs so re-ingesting
the same file overwrites its old points instead of duplicating them."""
import uuid
from pathlib import Path
from typing import Any

from ingestion.chunking import chunk_text
from ingestion.loaders import load_text
from app.rag.domain_sources import DomainConfig
from app.rag.embeddings.base import EmbeddingProvider
from app.rag.vector_store import VectorStore

_REPO_ROOT = Path(__file__).resolve().parent.parent


def doc_id_for(path: Path) -> str:
    try:
        return path.resolve().relative_to(_REPO_ROOT).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _point_id(doc_id: str, chunk_index: int) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{doc_id}:{chunk_index}"))


async def ingest_file(
    path: Path,
    domain: str,
    domain_config: DomainConfig,
    tags: dict[str, str],
    embedding_provider: EmbeddingProvider,
    vector_store: VectorStore,
) -> int:
    """Chunks + embeds + upserts one file. Returns the chunk count (0 if
    the file had no extractable text)."""
    doc_id = doc_id_for(path)
    text = load_text(path)
    chunks = chunk_text(text, domain_config.chunk_size, domain_config.chunk_overlap)

    await vector_store.ensure_collection(domain_config.collection, embedding_provider.dimensions)
    await vector_store.delete_by_doc_id(domain_config.collection, doc_id)

    if not chunks:
        return 0

    vectors = await embedding_provider.embed(chunks)
    points: list[dict[str, Any]] = [
        {
            "id": _point_id(doc_id, i),
            "vector": vector,
            "payload": {"doc_id": doc_id, "text": chunk, "domain": domain, "source_path": str(path), **tags},
        }
        for i, (chunk, vector) in enumerate(zip(chunks, vectors))
    ]
    await vector_store.upsert_chunks(domain_config.collection, points)
    return len(points)


async def remove_doc(doc_id: str, domain_config: DomainConfig, vector_store: VectorStore) -> None:
    await vector_store.delete_by_doc_id(domain_config.collection, doc_id)
