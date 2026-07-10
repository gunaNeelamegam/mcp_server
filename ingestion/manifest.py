"""The ingestion manifest: file path + content hash + embedding-model-used
per file, so `sync` can skip anything unchanged instead of re-processing
every file every run. This is the audit trail (SKILL.md "Document
Versioning & Audit Trail") — a natural extension, not a new system."""
import hashlib
import json
from pathlib import Path
from typing import Any

MANIFEST_PATH = Path(__file__).parent / "manifest.json"


def load() -> dict[str, Any]:
    if not MANIFEST_PATH.exists():
        return {"files": {}}
    return json.loads(MANIFEST_PATH.read_text())


def save(manifest: dict[str, Any]) -> None:
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, sort_keys=True))


def hash_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
