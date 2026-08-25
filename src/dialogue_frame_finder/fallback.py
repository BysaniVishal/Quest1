"""Fallback retrieval tiers, used only when exact-index anchor retrieval
(Phase 1) finds nothing at all -- e.g. because ASR corrupted every candidate
anchor word/n-gram beyond exact matching.

Tier order (see retrieval escalation in search.py):
  1. exact index lookup (Phase 1, retrieval.py) -- fast primary path
  2. fuzzy vocabulary lookup here -- approximate retrieval, still index-based
  3. bounded scan here -- last resort, verifies tiled windows across the
     whole transcript

Every escalation past tier 1 is logged (returned explicitly as the
`tier` used) rather than happening silently, so the caller/APPROACH.md can
be honest about how a result was actually found.
"""

from typing import Iterable, List, Tuple

from rapidfuzz.distance import Levenshtein

from .index import TranscriptIndex


def fuzzy_word_matches(word: str, vocabulary: Iterable[str], max_distance: int) -> List[str]:
    """Vocabulary words within edit distance `max_distance` of `word`."""
    return [
        v for v in vocabulary
        if v != word and Levenshtein.distance(word, v) <= max_distance
    ]


def fuzzy_anchor_positions(word: str, index: TranscriptIndex, max_distance: int = 2) -> List[int]:
    """Union of postings for every vocabulary word within edit-distance
    tolerance of `word` -- recovers an anchor whose exact spelling was
    corrupted by ASR (e.g. "stagnation" -> "stagnashun")."""
    matches = fuzzy_word_matches(word, index.vocabulary, max_distance)
    positions = set()
    for match in matches:
        positions.update(index.positions(match))
    return sorted(positions)


def bounded_scan_windows(
    target_len: int, transcript_len: int, tolerance: int = 3
) -> List[Tuple[int, int]]:
    """Tile the entire transcript into overlap-tolerant windows sized for the
    target phrase. Last resort: used only when no anchor (exact or fuzzy)
    exists anywhere in the transcript. Windows advance by target_len per
    step (not by 1) to keep the scan bounded rather than exhaustively
    overlapping."""
    if transcript_len <= 0 or target_len <= 0:
        return []
    window_size = target_len + 2 * tolerance
    step = max(1, target_len)
    windows: List[Tuple[int, int]] = []
    start = 0
    while start < transcript_len:
        end = min(transcript_len - 1, start + window_size - 1)
        windows.append((start, end))
        if end >= transcript_len - 1:
            break
        start += step
    return windows
