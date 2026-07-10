"""Output validation: verify the model's answer actually relates to the
retrieved chunks before trusting it — a lightweight heuristic, not a
second full model call (see SKILL.md Non-Negotiable Constraints)."""
import logging
import re

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_OVERLAP_THRESHOLD = 0.15

# Without this, common words ("the", "in", "a"...) inflate overlap between
# any two unrelated pieces of English text enough to clear the threshold,
# defeating the whole point of the check.
_STOPWORDS = frozenset(
    "a an the this that these those is are was were be been being to of "
    "in on at for with and or but not no as it its i you he she they we "
    "do does did will would can could should may might have has had".split()
)

_logger = logging.getLogger("logistics_ai.grounding")


def _tokenize(text: str) -> set[str]:
    return {token for token in _TOKEN_RE.findall(text.lower()) if token not in _STOPWORDS}


def verify(answer_text: str, chunks: list[dict]) -> bool:
    """Returns True if the answer plausibly used the retrieved chunks (or
    there were none to use). Logs a warning rather than blocking the
    response — this catches drift in logs, it doesn't gate delivery."""
    if not chunks:
        return True

    answer_tokens = _tokenize(answer_text)
    if not answer_tokens:
        return True

    for chunk in chunks:
        chunk_tokens = _tokenize(chunk.get("text", ""))
        if not chunk_tokens:
            continue
        overlap = answer_tokens & chunk_tokens
        if len(overlap) / len(answer_tokens) >= _OVERLAP_THRESHOLD:
            return True

    _logger.warning(
        "grounding check failed: answer does not overlap with any of %d retrieved chunks",
        len(chunks),
    )
    return False
