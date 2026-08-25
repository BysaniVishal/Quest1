"""Index-based candidate retrieval: the fast primary path.

Relying on a single top-ranked anchor is not enough for occurrence recall:
if the target dialogue occurs more than once and ASR corrupts a different
word in each occurrence, the single most SELECTIVE anchor (typically a rare
n-gram, since n-grams are naturally rarer than any one word) may happen to
exist in only one of those occurrences, silently hiding the others. Single
words are more robust across occurrences precisely because they are less
likely to all be corrupted the same way -- so retrieval tries several of the
rarest unigrams (not just the single rarest anchor overall) and unions their
hit positions, in addition to the single most selective overall anchor for
efficiency in the common-word-only case. These positions are candidate
locations, not verified matches -- local whole-phrase verification (Phase 2)
decides validity. Fuzzy/approximate anchor lookup and bounded-scan fallback
(for when every exact anchor is absent, e.g. due to heavy ASR corruption of
every candidate anchor) are also Phase 2 concerns.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

from .anchors import AnchorCandidate, generate_anchor_candidates, rank_anchors, select_anchors
from .index import TranscriptIndex


@dataclass(frozen=True)
class Candidate:
    position: int
    start: float
    end: float
    anchor: Optional[AnchorCandidate]


def lookup_anchor(anchor: AnchorCandidate, index: TranscriptIndex) -> List[Candidate]:
    if anchor.length == 1:
        positions = index.positions(anchor.text[0])
    else:
        positions = index.ngram_positions(anchor.text)
    candidates = []
    for pos in positions:
        entry = index.word_entry(pos)
        candidates.append(Candidate(position=pos, start=entry.start, end=entry.end, anchor=anchor))
    return candidates


def retrieve_candidates(
    target_words: List[str],
    index: TranscriptIndex,
    max_ngram: int = 3,
    max_unigram_anchors: int = 3,
) -> Tuple[List[Candidate], List[AnchorCandidate]]:
    """Union candidate positions from several robust anchors: the
    `max_unigram_anchors` rarest single words, plus the single most
    selective anchor overall (which may be a rarer n-gram, useful when every
    target word is individually too common to be selective -- see
    anchors.py). Returns (positions, anchors_actually_used) where the latter
    lists every anchor that contributed at least one hit, in the order
    tried, for diagnostics/explainability.

    Returns ([], []) if no single word or short n-gram from the target
    exists anywhere in the transcript's exact vocabulary -- exact retrieval
    is exhausted at that point and fuzzy/fallback retrieval (Phase 2) takes
    over.
    """
    unigram_anchors = rank_anchors(generate_anchor_candidates(target_words, index, max_ngram=1))
    tried: List[AnchorCandidate] = list(unigram_anchors[:max_unigram_anchors])

    overall_ranked = select_anchors(target_words, index, max_ngram=max_ngram)
    if overall_ranked and overall_ranked[0] not in tried:
        tried.append(overall_ranked[0])

    seen_positions = set()
    hits: List[Candidate] = []
    anchors_used: List[AnchorCandidate] = []
    for anchor in tried:
        anchor_hits = lookup_anchor(anchor, index)
        if not anchor_hits:
            continue
        anchors_used.append(anchor)
        for hit in anchor_hits:
            if hit.position not in seen_positions:
                seen_positions.add(hit.position)
                hits.append(hit)

    hits.sort(key=lambda h: h.position)
    return hits, anchors_used
