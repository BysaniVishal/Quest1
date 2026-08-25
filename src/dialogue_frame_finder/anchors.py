"""Anchor selection: pick the target words/n-grams most useful as index
search anchors, based on their rarity within this transcript.

The system does not depend on one distinctive word always existing (ASR may
corrupt it): candidates include every single word AND every short n-gram
(up to length 3) drawn from the target phrase, so a common-word-only target
can still anchor on a rare bigram/trigram even when no single word is rare.
"""

from dataclasses import dataclass
from typing import List, Tuple

from .index import TranscriptIndex


@dataclass(frozen=True)
class AnchorCandidate:
    text: Tuple[str, ...]
    target_start: int
    length: int
    frequency: int


def _ngram_frequency(index: TranscriptIndex, ngram_words: Tuple[str, ...]) -> int:
    if len(ngram_words) == 1:
        return index.frequency(ngram_words[0])
    return len(index.ngram_positions(ngram_words))


def generate_anchor_candidates(
    target_words: List[str], index: TranscriptIndex, max_ngram: int = 3
) -> List[AnchorCandidate]:
    """Generate every single-word and short n-gram (up to max_ngram) drawn
    from the (already normalized) target words, each tagged with its
    transcript-local frequency."""
    n = len(target_words)
    candidates: List[AnchorCandidate] = []
    for length in range(1, min(max_ngram, n) + 1):
        for start in range(0, n - length + 1):
            ngram = tuple(target_words[start : start + length])
            candidates.append(
                AnchorCandidate(
                    text=ngram,
                    target_start=start,
                    length=length,
                    frequency=_ngram_frequency(index, ngram),
                )
            )
    return candidates


def rank_anchors(candidates: List[AnchorCandidate]) -> List[AnchorCandidate]:
    """Rank anchors rarest-first. Only candidates that actually occur in the
    transcript (frequency > 0) are usable as retrieval anchors -- an anchor
    with zero postings can never retrieve anything.

    Ties are broken deterministically so anchor choice never depends on
    dict/set iteration order: prefer the more specific (longer) anchor, then
    the earlier position in the target phrase, then lexicographic order on
    the anchor text itself.
    """
    usable = [c for c in candidates if c.frequency > 0]
    return sorted(usable, key=lambda c: (c.frequency, -c.length, c.target_start, c.text))


def select_anchors(
    target_words: List[str], index: TranscriptIndex, max_ngram: int = 3
) -> List[AnchorCandidate]:
    """Rank every usable single-word/n-gram anchor for this target phrase,
    rarest (most distinctive) first."""
    return rank_anchors(generate_anchor_candidates(target_words, index, max_ngram=max_ngram))
