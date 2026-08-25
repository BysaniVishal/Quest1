"""Neighborhood extraction around an anchor hit.

Given an anchor hit at transcript position p, compute the small local window
of transcript positions to inspect during verification -- not the whole
transcript. Windows are clipped to valid transcript bounds and, when several
anchors produce overlapping windows, merged so a single true occurrence is
never verified (or reported) twice.
"""

from typing import List, Tuple


def extract_window(
    anchor_position: int,
    anchor_target_start: int,
    target_len: int,
    transcript_len: int,
    tolerance: int = 3,
) -> Tuple[int, int]:
    """Compute the inclusive [start, end] transcript-position window to
    inspect for an anchor hit.

    anchor_position: transcript position where the anchor was found.
    anchor_target_start: the anchor's starting index within the target
        phrase (0 if the anchor is the target's first word) -- used to
        estimate where the target phrase itself would begin/end relative to
        the anchor hit.
    target_len: number of words in the target phrase.
    transcript_len: total transcript length, for clipping.
    tolerance: extra words of padding on each side to absorb ASR
        insertions/deletions near the target boundary.
    """
    expected_target_start = anchor_position - anchor_target_start
    expected_target_end = expected_target_start + target_len - 1
    start = max(0, expected_target_start - tolerance)
    end = min(transcript_len - 1, expected_target_end + tolerance)
    return start, end


def merge_windows(windows: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """Merge overlapping/adjacent [start, end] windows (interval union),
    sorted ascending. Non-overlapping windows are left distinct."""
    if not windows:
        return []
    ordered = sorted(windows)
    merged = [ordered[0]]
    for start, end in ordered[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end + 1:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged
