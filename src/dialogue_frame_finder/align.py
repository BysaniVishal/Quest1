"""Word-level alignment between a target phrase and a local transcript window.

Produces, for each target word, which window position (if any) it aligns to
and how similar that pairing is. This alignment is the basis for: coverage
(how many target words are actually present), contiguity (how close together
the matched words are), lexical similarity, and -- critically -- locating the
transcript position of the target phrase's *first word* specifically, which
downstream onset extraction depends on.

Alignment uses a small dynamic-programming table (target words are short,
windows are small neighborhoods, not full transcripts) with free leading/
trailing skips on the window side (filler words around the target cost
nothing) but a real gap penalty for skips *inside* the aligned span, so a
scattered/shuffled word order is naturally scored worse than a contiguous
match -- word order is enforced by construction, not as a separate metric.
"""

from dataclasses import dataclass
from typing import List, Optional, Sequence

from rapidfuzz import fuzz

DEFAULT_GAP_PENALTY = -0.6


@dataclass(frozen=True)
class AlignmentEntry:
    target_index: int
    window_index: Optional[int]  # local index into window_words, or None (gap)
    similarity: float  # 0.0 for a gap


def _word_similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return fuzz.ratio(a, b) / 100.0


def align_words(
    target_words: Sequence[str],
    window_words: Sequence[str],
    gap_penalty: float = DEFAULT_GAP_PENALTY,
) -> List[AlignmentEntry]:
    n = len(target_words)
    m = len(window_words)
    if n == 0:
        return []
    if m == 0:
        return [AlignmentEntry(i, None, 0.0) for i in range(n)]

    dp = [[0.0] * (m + 1) for _ in range(n + 1)]
    # 'D' diagonal (aligned pair), 'T' skip target word (deletion), 'W' skip window word (insertion)
    ptr = [[None] * (m + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        dp[i][0] = dp[i - 1][0] + gap_penalty
        ptr[i][0] = "T"

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            sim = _word_similarity(target_words[i - 1], window_words[j - 1])
            # Signed so a poor match (sim near 0) actively costs almost as
            # much as a gap, instead of looking "free" -- otherwise the DP
            # would happily diagonal through two unrelated words rather than
            # taking a real gap, since sim in [0, 1] alone is never worse
            # than 0.
            diag = dp[i - 1][j - 1] + (2 * sim - 1)
            up = dp[i - 1][j] + gap_penalty
            left = dp[i][j - 1] + gap_penalty
            best, move = diag, "D"
            if up > best:
                best, move = up, "T"
            if left > best:
                best, move = left, "W"
            dp[i][j] = best
            ptr[i][j] = move

    best_j = max(range(m + 1), key=lambda j: dp[n][j])

    i, j = n, best_j
    reversed_pairs = []
    while i > 0:
        move = ptr[i][j]
        if move == "D":
            reversed_pairs.append((i - 1, j - 1))
            i -= 1
            j -= 1
        elif move == "T":
            reversed_pairs.append((i - 1, None))
            i -= 1
        else:  # "W" -- skip a window word, target index unchanged
            j -= 1
    reversed_pairs.reverse()

    entries: List[AlignmentEntry] = []
    for target_index, window_index in reversed_pairs:
        if window_index is None:
            entries.append(AlignmentEntry(target_index, None, 0.0))
        else:
            sim = _word_similarity(target_words[target_index], window_words[window_index])
            entries.append(AlignmentEntry(target_index, window_index, sim))
    return entries
