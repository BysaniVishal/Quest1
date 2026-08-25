"""Local whole-phrase verification of a candidate neighborhood.

The index only proposes candidate locations; this module disposes. It
compares the ENTIRE target phrase against a local transcript window using
word alignment (lexical similarity + coverage + contiguity), producing a
threshold-based valid/invalid classification and -- critically -- the
transcript position of the target phrase's first matched word, which is the
primary onset signal for later stages.
"""

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from .align import AlignmentEntry, align_words
from .config import DEFAULT_CONFIG, SearchConfig
from .index import TranscriptIndex


@dataclass(frozen=True)
class VerificationResult:
    window_start: int
    window_end: int
    alignment: List[AlignmentEntry]
    lexical_similarity: float
    coverage: float
    contiguity: float
    score: float
    valid: bool
    first_word_position: Optional[int]  # global transcript position
    first_word_start: Optional[float]
    first_word_end: Optional[float]


def _matched_entries(entries: Sequence[AlignmentEntry], threshold: float) -> List[AlignmentEntry]:
    return [e for e in entries if e.window_index is not None and e.similarity >= threshold]


def _contiguity(matched: Sequence[AlignmentEntry]) -> float:
    if len(matched) >= 2:
        positions = sorted(e.window_index for e in matched)
        span = positions[-1] - positions[0] + 1
        return len(matched) / span
    if len(matched) == 1:
        return 1.0
    return 0.0


def verify_window(
    target_words: Sequence[str],
    index: TranscriptIndex,
    window: Tuple[int, int],
    config: SearchConfig = DEFAULT_CONFIG,
) -> VerificationResult:
    window_start, window_end = window
    window_words = [
        index.transcript.words[p].normalized for p in range(window_start, window_end + 1)
    ]
    entries = align_words(target_words, window_words, gap_penalty=config.gap_penalty)

    matched = _matched_entries(entries, config.per_word_match_threshold)
    coverage = len(matched) / len(target_words) if target_words else 0.0
    contiguity = _contiguity(matched)
    lexical_similarity = sum(e.similarity for e in entries) / len(entries) if entries else 0.0

    score = (
        config.weight_lexical * lexical_similarity
        + config.weight_coverage * coverage
        + config.weight_contiguity * contiguity
    )
    valid = score >= config.valid_score_threshold

    first_word_position: Optional[int] = None
    first_word_start: Optional[float] = None
    first_word_end: Optional[float] = None
    for entry in sorted(matched, key=lambda e: e.target_index):
        global_pos = window_start + entry.window_index
        word_entry = index.transcript.words[global_pos]
        first_word_position = global_pos
        first_word_start = word_entry.start
        first_word_end = word_entry.end
        break

    return VerificationResult(
        window_start=window_start,
        window_end=window_end,
        alignment=entries,
        lexical_similarity=lexical_similarity,
        coverage=coverage,
        contiguity=contiguity,
        score=score,
        valid=valid,
        first_word_position=first_word_position,
        first_word_start=first_word_start,
        first_word_end=first_word_end,
    )
