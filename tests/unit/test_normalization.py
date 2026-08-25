import pytest

from dialogue_frame_finder.normalization import normalize_word, normalize_words

pytestmark = pytest.mark.unit


def test_lowercases():
    assert normalize_word("STAGNATION") == "stagnation"


def test_strips_leading_and_trailing_punctuation():
    assert normalize_word("stagnation.") == "stagnation"
    assert normalize_word('"stagnation"') == "stagnation"
    assert normalize_word("(stagnation)") == "stagnation"


def test_preserves_internal_apostrophe():
    assert normalize_word("don't") == "don't"
    assert normalize_word("Don't.") == "don't"


def test_strips_whitespace():
    assert normalize_word("  stagnation  ") == "stagnation"


def test_empty_string_normalizes_to_empty():
    assert normalize_word("") == ""


def test_punctuation_only_token_normalizes_to_empty():
    assert normalize_word("...") == ""
    assert normalize_word("--") == ""


def test_normalize_word_is_idempotent():
    once = normalize_word("Stagnation.")
    twice = normalize_word(once)
    assert once == twice == "stagnation"


def test_normalize_words_splits_and_normalizes_sentence():
    result = normalize_words("My mind rebels at stagnation.")
    assert result == ["my", "mind", "rebels", "at", "stagnation"]


def test_normalize_words_collapses_repeated_whitespace():
    result = normalize_words("My   mind\trebels  at stagnation")
    assert result == ["my", "mind", "rebels", "at", "stagnation"]


def test_normalize_words_drops_punctuation_only_tokens():
    result = normalize_words("My mind rebels ... at stagnation")
    assert result == ["my", "mind", "rebels", "at", "stagnation"]


def test_normalize_words_empty_string_returns_empty_list():
    assert normalize_words("") == []


def test_normalize_words_case_and_capitalization_variants_match():
    a = normalize_words("MY MIND REBELS AT STAGNATION")
    b = normalize_words("my mind rebels at stagnation")
    assert a == b
