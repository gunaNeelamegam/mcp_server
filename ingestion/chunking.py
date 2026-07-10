"""Character-based sliding-window chunker. Simple on purpose — swap for a
smarter (sentence/section-aware) splitter later if retrieval quality
demands it; nothing outside this file needs to change."""


def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    text = text.strip()
    if not text:
        return []

    step = chunk_size - chunk_overlap
    chunks = []
    for start in range(0, len(text), step):
        chunk = text[start : start + chunk_size].strip()
        if chunk:
            chunks.append(chunk)
        if start + chunk_size >= len(text):
            break
    return chunks
