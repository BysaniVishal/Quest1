import pytest

from dialogue_frame_finder.align import align_words

pytestmark = pytest.mark.unit


def _positions(entries):
    return [e.window_index for e in entries]


def test_align_exact_match_aligns_every_word_in_order():
    target = ["my", "mind", "rebels", "at", "stagnation"]
    entries = align_words(target, target)
    assert _positions(entries) == [0, 1, 2, 3, 4]
    assert all(e.similarity == pytest.approx(1.0) for e in entries)


def test_align_tolerates_free_leading_filler():
    target = ["my", "mind", "rebels", "at", "stagnation"]
    window = ["well", "i", "think", "my", "mind", "rebels", "at", "stagnation"]
    entries = align_words(target, window)
    assert _positions(entries) == [3, 4, 5, 6, 7]
    assert all(e.similarity == pytest.approx(1.0) for e in entries)


def test_align_tolerates_free_trailing_filler():
    target = ["my", "mind", "rebels", "at", "stagnation"]
    window = ["my", "mind", "rebels", "at", "stagnation", "and", "more", "words"]
    entries = align_words(target, window)
    assert _positions(entries) == [0, 1, 2, 3, 4]


def test_align_handles_single_word_substitution():
    # TM-03: "my mind rebels in stagnation" -- ASR substituted "at" -> "in"
    target = ["my", "mind", "rebels", "at", "stagnation"]
    window = ["my", "mind", "rebels", "in", "stagnation"]
    entries = align_words(target, window)
    by_index = {e.target_index: e for e in entries}
    assert by_index[0].similarity == pytest.approx(1.0)
    assert by_index[4].similarity == pytest.approx(1.0)
    assert by_index[3].similarity < 0.5  # "at" vs "in" -- weak match


def test_align_near_miss_substitution_scores_high():
    # rebels -> revels: one-character ASR slip, should still align with high similarity
    target = ["my", "mind", "rebels", "at", "stagnation"]
    window = ["my", "mind", "revels", "at", "stagnation"]
    entries = align_words(target, window)
    by_index = {e.target_index: e for e in entries}
    assert by_index[2].window_index == 2
    assert by_index[2].similarity > 0.75


def test_align_handles_missing_target_word_as_true_gap():
    # "rebels" entirely absent from the window (ASR dropped it)
    target = ["my", "mind", "rebels", "at", "stagnation"]
    window = ["my", "mind", "at", "stagnation"]
    entries = align_words(target, window)
    by_index = {e.target_index: e for e in entries}
    assert by_index[2].window_index is None
    assert by_index[0].window_index == 0
    assert by_index[1].window_index == 1
    assert by_index[3].window_index == 2
    assert by_index[4].window_index == 3


def test_align_handles_inserted_word_between_target_words():
    # TM-05: "my mind ... at stagnation" -- extra word inserted mid-phrase
    target = ["my", "mind", "rebels", "at", "stagnation"]
    window = ["my", "mind", "well", "rebels", "at", "stagnation"]
    entries = align_words(target, window)
    by_index = {e.target_index: e for e in entries}
    assert by_index[2].window_index == 3  # rebels still found, past the inserted word
    assert by_index[2].similarity == pytest.approx(1.0)


def test_align_scrambled_word_order_leaves_words_unmatched_or_low_similarity():
    # TM-04: "stagnation at rebels mind my" -- correct words, wrong order
    target = ["my", "mind", "rebels", "at", "stagnation"]
    window = ["stagnation", "at", "rebels", "mind", "my"]
    entries = align_words(target, window)
    matched_high_sim = [e for e in entries if e.window_index is not None and e.similarity > 0.9]
    # order-preserving alignment cannot match every word well against a full reversal
    assert len(matched_high_sim) < len(target)


def test_align_empty_target_returns_empty_list():
    assert align_words([], ["my", "mind"]) == []


def test_align_empty_window_all_target_words_are_gaps():
    entries = align_words(["my", "mind"], [])
    assert all(e.window_index is None for e in entries)
    assert len(entries) == 2
