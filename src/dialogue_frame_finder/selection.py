"""Earliest-valid-occurrence selection.

The target dialogue may appear more than once. Among all verified-valid
occurrences, the one selected is whichever comes FIRST in video time -- never
whichever scored highest. Other valid occurrences are retained as diagnostic
metadata (useful for confidence/ambiguity reporting) but never override the
earliest-time selection.
"""

from typing import List, Optional, Sequence, Tuple

from .verification import VerificationResult


def select_earliest_valid(
    results: Sequence[VerificationResult],
) -> Tuple[Optional[VerificationResult], List[VerificationResult]]:
    valid = [r for r in results if r.valid and r.first_word_start is not None]
    if not valid:
        return None, []
    ordered = sorted(valid, key=lambda r: r.first_word_start)
    return ordered[0], ordered[1:]
