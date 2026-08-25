"""End-to-end dialogue search: Phase 1 (indexed retrieval) + Phase 2
(neighborhood verification, fallback escalation, earliest-valid selection).

Tier order, escalating only when the previous tier finds no candidate
positions at all:
  1. exact_anchor  -- Phase 1 inverted-index anchor retrieval (fast path)
  2. fuzzy_anchor   -- approximate vocabulary lookup per target word
  3. bounded_scan   -- tiled scan of the whole transcript (last resort)
  4. none           -- no candidate positions found by any tier

Which tier actually produced the result is always reported, never silent.
"""

from dataclasses import dataclass
from typing import List, Optional, Set

from .anchors import AnchorCandidate
from .config import DEFAULT_CONFIG, SearchConfig
from .fallback import bounded_scan_windows, fuzzy_anchor_positions
from .index import TranscriptIndex
from .neighborhood import extract_window, merge_windows
from .normalization import normalize_words
from .retrieval import Candidate, retrieve_candidates
from .selection import select_earliest_valid
from .transcript import Transcript
from .verification import VerificationResult, verify_window


@dataclass(frozen=True)
class SearchResult:
    chosen: Optional[VerificationResult]
    other_valid: List[VerificationResult]
    tier_used: str
    anchors_used: List[AnchorCandidate]
    windows_verified: int


def _fuzzy_retrieve(target_words: List[str], index: TranscriptIndex, config: SearchConfig) -> List[Candidate]:
    positions: Set[int] = set()
    for word in target_words:
        positions.update(fuzzy_anchor_positions(word, index, config.fuzzy_anchor_max_distance))
    candidates = []
    for pos in sorted(positions):
        entry = index.word_entry(pos)
        candidates.append(Candidate(position=pos, start=entry.start, end=entry.end, anchor=None))
    return candidates


def search_dialogue(
    target_text: str, transcript: Transcript, config: SearchConfig = DEFAULT_CONFIG
) -> SearchResult:
    target_words = normalize_words(target_text)
    if not target_words:
        return SearchResult(None, [], "none", [], 0)

    index = TranscriptIndex.build(transcript)

    hits, anchors = retrieve_candidates(
        target_words, index, max_ngram=config.max_ngram_anchor,
        max_unigram_anchors=config.max_unigram_anchors,
    )
    tier = "exact_anchor"

    if not hits:
        fuzzy_hits = _fuzzy_retrieve(target_words, index, config)
        if fuzzy_hits:
            hits, anchors, tier = fuzzy_hits, [], "fuzzy_anchor"

    if hits:
        raw_windows = [
            extract_window(
                hit.position,
                hit.anchor.target_start if hit.anchor is not None else 0,
                len(target_words),
                len(transcript),
                config.neighborhood_tolerance,
            )
            for hit in hits
        ]
        windows = merge_windows(raw_windows)
    else:
        windows = bounded_scan_windows(
            len(target_words), len(transcript), config.neighborhood_tolerance
        )
        tier = "bounded_scan" if windows else "none"

    results = [verify_window(target_words, index, window, config) for window in windows]
    chosen, others = select_earliest_valid(results)

    return SearchResult(
        chosen=chosen,
        other_valid=others,
        tier_used=tier,
        anchors_used=anchors,
        windows_verified=len(results),
    )
