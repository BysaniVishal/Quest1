import pytest

from dialogue_frame_finder.index import TranscriptIndex
from dialogue_frame_finder.transcript import Transcript

pytestmark = pytest.mark.unit


def _index_from_words(words):
    """words: list of plain strings, spaced 0.1s apart, for concise fixtures."""
    tuples = [(w, i * 0.1, i * 0.1 + 0.1) for i, w in enumerate(words)]
    return TranscriptIndex.build(Transcript.from_word_tuples(tuples))


def test_build_creates_postings_for_each_word():
    idx = _index_from_words(["my", "mind", "rebels", "at", "stagnation"])
    assert idx.frequency("stagnation") == 1


def test_frequency_counts_repeated_occurrences():
    idx = _index_from_words(["my", "cat", "and", "my", "dog"])
    assert idx.frequency("my") == 2


def test_frequency_unknown_word_is_zero():
    idx = _index_from_words(["my", "mind"])
    assert idx.frequency("nonexistent") == 0


def test_frequency_normalizes_input():
    idx = _index_from_words(["stagnation"])
    assert idx.frequency("STAGNATION.") == 1


def test_positions_returns_correct_ascending_indices():
    idx = _index_from_words(["my", "cat", "and", "my", "dog"])
    assert idx.positions("my") == [0, 3]


def test_positions_unknown_word_returns_empty_list():
    idx = _index_from_words(["my", "mind"])
    assert idx.positions("nonexistent") == []


def test_word_entry_returns_correct_entry_at_position():
    idx = _index_from_words(["my", "mind", "rebels"])
    assert idx.word_entry(1).normalized == "mind"


def test_vocabulary_contains_all_distinct_normalized_words():
    idx = _index_from_words(["my", "mind", "my", "rebels"])
    assert set(idx.vocabulary) == {"my", "mind", "rebels"}


def test_ngram_positions_finds_contiguous_sequence():
    idx = _index_from_words(["well", "i", "think", "my", "mind", "rebels", "at", "stagnation"])
    assert idx.ngram_positions(["my", "mind"]) == [3]


def test_ngram_positions_no_match_returns_empty():
    idx = _index_from_words(["my", "mind", "rebels"])
    assert idx.ngram_positions(["mind", "my"]) == []


def test_ngram_positions_does_not_match_non_contiguous_words():
    idx = _index_from_words(["my", "cat", "mind"])
    assert idx.ngram_positions(["my", "mind"]) == []


def test_ngram_positions_boundary_at_end_of_transcript_no_error():
    idx = _index_from_words(["at", "stagnation"])
    # ngram would extend past the last position -- must not raise, must not match
    assert idx.ngram_positions(["stagnation", "extra"]) == []


def test_ngram_positions_empty_ngram_returns_empty():
    idx = _index_from_words(["my", "mind"])
    assert idx.ngram_positions([]) == []


def test_ngram_positions_multiple_occurrences_all_found():
    idx = _index_from_words(["at", "stagnation", "later", "at", "stagnation"])
    assert idx.ngram_positions(["at", "stagnation"]) == [0, 3]
