"""Extracts plain text from a source file, dispatched by extension."""
from pathlib import Path

from pypdf import PdfReader


def load_text(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        reader = PdfReader(str(path))
        return "\n\n".join(page.extract_text() or "" for page in reader.pages)
    return path.read_text(encoding="utf-8")
