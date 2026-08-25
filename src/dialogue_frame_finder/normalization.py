"""Text normalization for transcript words and target dialogue.

Rules: lowercase, strip leading/trailing punctuation, collapse whitespace.
Apostrophes are preserved so contractions ("don't") survive normalization
intact rather than being split or mangled.
"""

import re
from typing import List

_STRIP_PATTERN = re.compile(r"^[^\w']+|[^\w']+$")


def normalize_word(word: str) -> str:
    """Normalize a single token. Idempotent: normalizing an already-normalized
    word returns it unchanged."""
    if not word:
        return ""
    w = word.lower().strip()
    w = _STRIP_PATTERN.sub("", w)
    return w


def normalize_words(text: str) -> List[str]:
    """Split text on whitespace and normalize each token, dropping tokens that
    normalize to nothing (pure punctuation, empty)."""
    if not text:
        return []
    return [w for w in (normalize_word(tok) for tok in text.split()) if w]
