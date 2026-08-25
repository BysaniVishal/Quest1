import pytest

from dialogue_frame_finder.fallback import (
    bounded_scan_windows,
    fuzzy_anchor_positions,
    fuzzy_word_matches,
)
from dialogue_frame_finder.index import TranscriptIndex
from dialogue_frame_finder.transcript import Transcript

pytestmark = pytest.mark.unit


def _index_from_words(words):
    tuples = [(w, i * 0.1, i * 0.1 + 0.1) for i, w in enumerate(words)]
    return TranscriptIndex.build(Transcript.from_word_tuples(tuples))


def test_fuzzy_word_matches_finds_close_variant():
    vocabulary = ["stagnashun", "unrelated", "completely"]
    matches = fuzzy_word_matches("stagnation", vocabulary, max_distance=3)
    assert "stagnashun" in matches


def test_fuzzy_word_matches_excludes_distant_words():
    vocabulary = ["completely", "unrelated"]
    matches = fuzzy_word_matches("stagnation", vocabulary, max_distance=2)
    assert matches == []


def test_fuzzy_word_matches_excludes_exact_self_match():
    vocabulary = ["stagnation"]
    matches = fuzzy_word_matches("stagnation", vocabulary, max_distance=2)
    assert matches == []


def test_fuzzy_anchor_positions_unions_postings_of_all_near_matches():
    idx = _index_from_words(["stagnaton", "and", "later", "stagnashun"])
    positions = fuzzy_anchor_positions("stagnation", idx, max_distance=3)
    assert positions == [0, 3]


def test_fuzzy_anchor_positions_empty_when_nothing_close():
    idx = _index_from_words(["completely", "unrelated", "words"])
    assert fuzzy_anchor_positions("stagnation", idx, max_distance=2) == []


def test_bounded_scan_windows_covers_entire_transcript():
    windows = bounded_scan_windows(target_len=5, transcript_len=40, tolerance=3)
    assert windows[0][0] == 0
    assert windows[-1][1] == 39


def test_bounded_scan_windows_empty_transcript():
    assert bounded_scan_windows(target_len=5, transcript_len=0, tolerance=3) == []


def test_bounded_scan_windows_short_transcript_single_window():
    windows = bounded_scan_windows(target_len=5, transcript_len=8, tolerance=3)
    assert len(windows) == 1
    assert windows[0] == (0, 7)
