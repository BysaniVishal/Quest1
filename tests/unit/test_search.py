import pytest

from dialogue_frame_finder.search import search_dialogue
from dialogue_frame_finder.transcript import Transcript

pytestmark = pytest.mark.unit


def _transcript_from_words(words):
    tuples = [(w, i * 0.4, i * 0.4 + 0.35) for i, w in enumerate(words)]
    return Transcript.from_word_tuples(tuples)


TARGET = "My mind rebels at stagnation"


def test_search_dialogue_single_occurrence_high_confidence():
    words = ["well", "i", "think", "my", "mind", "rebels", "at", "stagnation", "today"]
    result = search_dialogue(TARGET, _transcript_from_words(words))
    assert result.chosen is not None
    assert result.chosen.valid is True
    assert result.tier_used == "exact_anchor"
    assert result.chosen.first_word_position == 3


def test_search_dialogue_punctuation_and_case_difference_still_matches():
    # TM-02: "my mind rebels at stagnation." vs target "My mind rebels at
    # stagnation" -- same high-confidence match despite case/punctuation.
    words = ["my", "mind", "rebels", "at", "stagnation."]
    result = search_dialogue(TARGET, _transcript_from_words(words))
    assert result.chosen is not None
    assert result.chosen.score == pytest.approx(1.0, abs=1e-6)


def test_search_dialogue_asr_substitution_still_found():
    # TM-03
    words = ["well", "i", "think", "my", "mind", "rebels", "in", "stagnation"]
    result = search_dialogue(TARGET, _transcript_from_words(words))
    assert result.chosen is not None
    assert result.chosen.valid is True


def test_search_dialogue_tolerates_inserted_word_gap():
    # TM-05: "my mind rebels ... at stagnation" -- candidate retained across
    # a small ASR-inserted gap, with reduced confidence rather than rejection.
    words = ["well", "i", "think", "my", "mind", "rebels", "extra", "at", "stagnation"]
    result = search_dialogue(TARGET, _transcript_from_words(words))
    assert result.chosen is not None
    assert result.chosen.valid is True
    assert result.chosen.score < 1.0  # eligible, but confidence reduced


def test_search_dialogue_scrambled_words_no_confident_match():
    # TM-04: lower sequence score must not produce a confident match
    words = ["stagnation", "at", "rebels", "mind", "my"]
    result = search_dialogue(TARGET, _transcript_from_words(words))
    assert result.chosen is None


def test_search_dialogue_target_entirely_absent():
    # TM-06 -- shares no vocabulary with the target at all
    words = ["completely", "unrelated", "sentence", "about", "nothing", "everywhere", "random"]
    result = search_dialogue(TARGET, _transcript_from_words(words))
    assert result.chosen is None
    assert result.tier_used in ("bounded_scan", "none")


def test_search_dialogue_returns_earliest_occurrence_not_highest_scoring():
    # first occurrence has a minor ASR substitution (still valid, slightly lower
    # score); second occurrence is a perfect match with a higher score. The
    # earlier, weaker-but-valid occurrence must win.
    words = (
        ["well", "i", "think", "my", "mind", "rebels", "in", "stagnation"]  # weaker, earlier
        + ["filler"] * 15
        + ["my", "mind", "rebels", "at", "stagnation"]  # stronger, later
    )
    result = search_dialogue(TARGET, _transcript_from_words(words))
    assert result.chosen is not None
    assert result.chosen.first_word_position == 3  # the earlier occurrence's "my"
    assert len(result.other_valid) >= 1
    assert result.other_valid[0].first_word_start > result.chosen.first_word_start


def test_search_dialogue_repeated_phrase_all_valid_occurrences_tracked():
    words = (
        ["my", "mind", "rebels", "at", "stagnation"]
        + ["filler"] * 10
        + ["my", "mind", "rebels", "at", "stagnation"]
        + ["filler"] * 10
        + ["my", "mind", "rebels", "at", "stagnation"]
    )
    result = search_dialogue(TARGET, _transcript_from_words(words))
    assert result.chosen is not None
    assert result.chosen.first_word_position == 0  # earliest of the three
    assert len(result.other_valid) == 2


def test_search_dialogue_target_near_transcript_start():
    # TM-08
    words = ["my", "mind", "rebels", "at", "stagnation", "and", "more", "words", "follow"]
    result = search_dialogue(TARGET, _transcript_from_words(words))
    assert result.chosen is not None
    assert result.chosen.first_word_position == 0


def test_search_dialogue_target_near_transcript_end():
    # TM-09
    words = ["words", "come", "before", "my", "mind", "rebels", "at", "stagnation"]
    result = search_dialogue(TARGET, _transcript_from_words(words))
    assert result.chosen is not None
    assert result.chosen.first_word_position == 3


def test_search_dialogue_empty_target_returns_no_match():
    result = search_dialogue("   ", _transcript_from_words(["my", "mind"]))
    assert result.chosen is None
    assert result.tier_used == "none"


def test_search_dialogue_empty_transcript_returns_no_match():
    result = search_dialogue(TARGET, Transcript(words=[]))
    assert result.chosen is None
    assert result.tier_used in ("bounded_scan", "none")


def test_search_dialogue_falls_back_to_fuzzy_anchor_when_exact_anchors_absent():
    # Every target word is corrupted by a single-character ASR-style edit, so
    # NO exact single word or n-gram from the target exists anywhere in the
    # transcript's vocabulary -- exact retrieval (tier 1) must find nothing,
    # forcing escalation to fuzzy vocabulary lookup (tier 2).
    words = ["well", "i", "think", "muy", "mnd", "rebbels", "att", "stagnaton"]
    result = search_dialogue(TARGET, _transcript_from_words(words))
    assert result.tier_used == "fuzzy_anchor"
    assert result.chosen is not None
