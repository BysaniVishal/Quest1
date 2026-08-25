import pytest

from dialogue_frame_finder.config import SearchConfig
from dialogue_frame_finder.index import TranscriptIndex
from dialogue_frame_finder.normalization import normalize_words
from dialogue_frame_finder.transcript import Transcript
from dialogue_frame_finder.verification import verify_window

pytestmark = pytest.mark.unit


def _index_from_words(words):
    tuples = [(w, i * 0.4, i * 0.4 + 0.35) for i, w in enumerate(words)]
    return TranscriptIndex.build(Transcript.from_word_tuples(tuples))


def test_verify_window_exact_match_is_valid_with_max_score():
    words = ["well", "i", "think", "my", "mind", "rebels", "at", "stagnation"]
    idx = _index_from_words(words)
    target = normalize_words("My mind rebels at stagnation")
    result = verify_window(target, idx, (3, 7))
    assert result.valid is True
    assert result.coverage == pytest.approx(1.0)
    assert result.contiguity == pytest.approx(1.0)
    assert result.score == pytest.approx(1.0, abs=1e-6)


def test_verify_window_exact_match_first_word_position_and_timestamp():
    words = ["well", "i", "think", "my", "mind", "rebels", "at", "stagnation"]
    idx = _index_from_words(words)
    target = normalize_words("My mind rebels at stagnation")
    result = verify_window(target, idx, (3, 7))
    assert result.first_word_position == 3
    assert result.first_word_start == idx.transcript.words[3].start
    assert result.first_word_end == idx.transcript.words[3].end


def test_verify_window_minor_substitution_still_valid_but_lower_score():
    # TM-03: "my mind rebels in stagnation" -- still eligible, reduced confidence
    exact_words = ["well", "i", "think", "my", "mind", "rebels", "at", "stagnation"]
    subst_words = ["well", "i", "think", "my", "mind", "rebels", "in", "stagnation"]
    idx_exact = _index_from_words(exact_words)
    idx_subst = _index_from_words(subst_words)
    target = normalize_words("My mind rebels at stagnation")

    exact_result = verify_window(target, idx_exact, (3, 7))
    subst_result = verify_window(target, idx_subst, (3, 7))

    assert subst_result.valid is True
    assert subst_result.score < exact_result.score
    assert subst_result.coverage == pytest.approx(0.8)  # 4 of 5 target words matched


def test_verify_window_scrambled_order_scores_lower_than_contiguous_match():
    # TM-04: correct words, wrong order -- must not beat a correct contiguous match
    exact_words = ["my", "mind", "rebels", "at", "stagnation"]
    scrambled_words = ["stagnation", "at", "rebels", "mind", "my"]
    idx_exact = _index_from_words(exact_words)
    idx_scrambled = _index_from_words(scrambled_words)
    target = normalize_words("My mind rebels at stagnation")

    exact_result = verify_window(target, idx_exact, (0, 4))
    scrambled_result = verify_window(target, idx_scrambled, (0, 4))

    assert scrambled_result.score < exact_result.score
    assert scrambled_result.valid is False


def test_verify_window_missing_word_reduces_coverage_but_can_remain_valid():
    words = ["my", "mind", "at", "stagnation"]  # "rebels" entirely dropped by ASR
    idx = _index_from_words(words)
    target = normalize_words("My mind rebels at stagnation")
    result = verify_window(target, idx, (0, 3))
    assert result.coverage == pytest.approx(0.8)
    assert result.valid is True  # coverage-tolerant, not exact-match-only


def test_verify_window_unrelated_text_is_not_valid():
    words = ["completely", "unrelated", "sentence", "about", "nothing"]
    idx = _index_from_words(words)
    target = normalize_words("My mind rebels at stagnation")
    result = verify_window(target, idx, (0, 4))
    assert result.valid is False
    assert result.first_word_position is None


def test_verify_window_respects_configurable_threshold():
    words = ["my", "mind", "at", "stagnation"]  # coverage 0.8, as above
    idx = _index_from_words(words)
    target = normalize_words("My mind rebels at stagnation")
    lenient = verify_window(target, idx, (0, 3), config=SearchConfig(valid_score_threshold=0.5))
    strict = verify_window(target, idx, (0, 3), config=SearchConfig(valid_score_threshold=0.99))
    assert lenient.valid is True
    assert strict.valid is False
