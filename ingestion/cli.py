"""Ingestion CLI: add/sync/rebuild/status/remove. Works identically
across every domain — domain is just a parameter, never special-cased.
Run with `python -m ingestion.cli <command>`.
"""
import asyncio
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import click

from app.config import get_settings
from app.rag.domain_sources import DOMAINS, DomainConfig
from app.rag.embeddings.registry import get_active as get_embedder
from app.rag.vector_store import get_store
from ingestion.manifest import hash_file, load, save
from ingestion.pipeline import doc_id_for, ingest_file, remove_doc


def _parse_tags(raw: tuple[str, ...]) -> dict[str, str]:
    tags = {}
    for item in raw:
        key, _, value = item.partition("=")
        if not value:
            raise click.UsageError(f"--tag must be key=value, got: {item!r}")
        tags[key] = value
    return tags


def _infer_tags(config: DomainConfig, path: Path) -> dict[str, str]:
    """Compliance-style domains tag files by their immediate subfolder
    under data_dir (ingestion/data/compliance/uae/*.pdf -> country=UAE) so
    sync/rebuild don't require re-specifying --tag by hand."""
    if not config.tag_keys:
        return {}
    rel_parts = path.resolve().relative_to(Path(config.data_dir).resolve()).parts
    if len(rel_parts) > 1:
        return {config.tag_keys[0]: rel_parts[0].upper()}
    return {}


def _collect_files(root: Path, config: DomainConfig) -> list[Path]:
    if not root.exists():
        return []
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in config.file_types)


def _manifest_entry(domain: str, tags: dict[str, str], file_hash: str, chunk_count: int) -> dict[str, Any]:
    return {
        "domain": domain,
        "tags": tags,
        "hash": file_hash,
        "embedding_model": get_settings().ollama_embedding_model,
        "chunk_count": chunk_count,
        "last_embedded": datetime.now(UTC).isoformat(),
    }


@click.group()
def cli() -> None:
    pass


@cli.command()
@click.option("--domain", required=True, type=click.Choice(list(DOMAINS)))
@click.option("--tag", "raw_tags", multiple=True, help="key=value, repeatable")
@click.option("--path", "path_str", required=True)
def add(domain: str, raw_tags: tuple[str, ...], path_str: str) -> None:
    """Ingest a file or directory into DOMAIN, skipping unchanged files."""
    asyncio.run(_add(domain, _parse_tags(raw_tags), Path(path_str)))


async def _add(domain: str, explicit_tags: dict[str, str], path: Path) -> None:
    config = DOMAINS[domain]
    embedder, store = get_embedder(), get_store()
    manifest = load()

    files = [path] if path.is_file() else _collect_files(path, config)
    for file_path in files:
        key = doc_id_for(file_path)
        file_hash = hash_file(file_path)
        prior = manifest["files"].get(key)
        if prior and prior["hash"] == file_hash:
            click.echo(f"skip (unchanged): {key}")
            continue

        tags = {**_infer_tags(config, file_path), **explicit_tags}
        count = await ingest_file(file_path, domain, config, tags, embedder, store)
        manifest["files"][key] = _manifest_entry(domain, tags, file_hash, count)
        click.echo(f"ingested {count} chunks: {key}")

    save(manifest)


@cli.command()
def sync() -> None:
    """Scan every domain, only process changed/new files. Safe to run often."""
    asyncio.run(_sync())


async def _sync() -> None:
    manifest = load()
    embedder, store = get_embedder(), get_store()
    seen: set[str] = set()

    for domain, config in DOMAINS.items():
        for file_path in _collect_files(Path(config.data_dir), config):
            key = doc_id_for(file_path)
            seen.add(key)
            file_hash = hash_file(file_path)
            prior = manifest["files"].get(key)
            if prior and prior["hash"] == file_hash:
                continue

            tags = _infer_tags(config, file_path)
            count = await ingest_file(file_path, domain, config, tags, embedder, store)
            manifest["files"][key] = _manifest_entry(domain, tags, file_hash, count)
            click.echo(f"[{domain}] ingested {count} chunks: {key}")

    missing = [key for key in manifest["files"] if key not in seen]
    if missing:
        click.echo("Files removed from disk (vectors kept — run `ingest remove` to drop them):")
        for key in missing:
            click.echo(f"  - {key}")

    save(manifest)


@cli.command()
@click.option("--domain", type=click.Choice(list(DOMAINS)))
@click.option("--tag", "raw_tags", multiple=True, help="key=value, filters which files get rebuilt")
@click.option("--all", "rebuild_all", is_flag=True, help="rebuild every domain")
def rebuild(domain: str | None, raw_tags: tuple[str, ...], rebuild_all: bool) -> None:
    """Force a full re-embed — the manual escape hatch for an embedding
    model change or a suspected out-of-sync vector store."""
    if not rebuild_all and not domain:
        raise click.UsageError("Specify --domain or --all")
    asyncio.run(_rebuild(domain, _parse_tags(raw_tags), rebuild_all))


async def _rebuild(domain: str | None, tag_filter: dict[str, str], rebuild_all: bool) -> None:
    manifest = load()
    embedder, store = get_embedder(), get_store()
    domains = list(DOMAINS) if rebuild_all else [domain]

    for d in domains:
        config = DOMAINS[d]
        for file_path in _collect_files(Path(config.data_dir), config):
            tags = _infer_tags(config, file_path)
            if tag_filter and not all(tags.get(k) == v for k, v in tag_filter.items()):
                continue

            key = doc_id_for(file_path)
            count = await ingest_file(file_path, d, config, tags, embedder, store)
            manifest["files"][key] = _manifest_entry(d, tags, hash_file(file_path), count)
            click.echo(f"[{d}] rebuilt {count} chunks: {key}")

    save(manifest)


@cli.command()
def status() -> None:
    """Show manifest: per-domain file counts, last embedded time, pending changes."""
    manifest = load()
    by_domain: dict[str, list[dict[str, Any]]] = {}
    for entry in manifest["files"].values():
        by_domain.setdefault(entry["domain"], []).append(entry)

    for domain in DOMAINS:
        entries = by_domain.get(domain, [])
        last = max((e["last_embedded"] for e in entries), default=None)
        click.echo(f"{domain}: {len(entries)} files ingested, last embedded: {last or 'never'}")

    changed: list[str] = []
    all_on_disk_keys: set[str] = set()
    for domain, config in DOMAINS.items():
        for file_path in _collect_files(Path(config.data_dir), config):
            key = doc_id_for(file_path)
            all_on_disk_keys.add(key)
            prior = manifest["files"].get(key)
            if not prior or prior["hash"] != hash_file(file_path):
                changed.append(key)

    missing = [key for key in manifest["files"] if key not in all_on_disk_keys]

    if changed:
        click.echo("\nPending (new/changed, run `ingest sync`):")
        for key in changed:
            click.echo(f"  - {key}")
    if missing:
        click.echo("\nMissing from disk (vectors kept, run `ingest remove` to drop):")
        for key in missing:
            click.echo(f"  - {key}")


@cli.command()
@click.option("--domain", required=True, type=click.Choice(list(DOMAINS)))
@click.option("--file", "file_str", required=True, help="path (as ingested) or manifest key")
def remove(domain: str, file_str: str) -> None:
    """Remove a specific document + its vectors."""
    asyncio.run(_remove(domain, file_str))


async def _remove(domain: str, file_str: str) -> None:
    config = DOMAINS[domain]
    store = get_store()
    path = Path(file_str)
    key = doc_id_for(path) if path.exists() else file_str

    await remove_doc(key, config, store)

    manifest = load()
    if key in manifest["files"]:
        del manifest["files"][key]
        save(manifest)

    click.echo(f"removed: {key}")


if __name__ == "__main__":
    cli()
