import pytest

from dialogue_frame_finder.anchors import generate_anchor_candidates, rank_anchors, select_anchors
from dialogue_frame_finder.index import TranscriptIndex
from dialogue_frame_finder.transcript import Transcript

pytestmark = pytest.mark.unit


def _index_from_words(words):
    tuples = [(w, i * 0.1, i * 0.1 + 0.1) for i, w in enumerate(words)]
    return TranscriptIndex.build(Transcript.from_word_tuples(tuples))


def test_generate_anchor_candidates_includes_unigrams_bigrams_trigrams():
    # design-doc example: "my" and "at" frequent, "mind" moderate,
    # "rebels"/"stagnation" rare -- see design.docx section 3
    idx = _index_from_words(
        [
            "well", "i", "think", "my", "mind", "rebels", "at", "stagnation",
            "and", "my", "at", "my", "mind", "at", "my", "at", "mind", "at", "my",
        ]
    )
    target = ["my", "mind", "rebels", "at", "stagnation"]
    candidates = generate_anchor_candidates(target, idx, max_ngram=3)
    # 5 unigrams + 4 bigrams + 3 trigrams = 12
    assert len(candidates) == 12
    lengths = sorted(c.length for c in candidates)
    assert lengths == [1, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3]


def test_rank_anchors_prefers_lower_frequency_word():
    idx = _index_from_words(
        ["well", "i", "think", "my", "mind", "rebels", "at", "stagnation",
         "and", "my", "at", "my", "mind", "at", "my"]
    )
    target = ["my", "mind", "rebels", "at", "stagnation"]
    # isolate single-word ranking from n-gram interference
    candidates = generate_anchor_candidates(target, idx, max_ngram=1)
    ranked = rank_anchors(candidates)
    assert ranked[0].frequency == 1
    assert ranked[0].text[0] in ("rebels", "stagnation")
    # frequency is non-decreasing down the ranked list
    freqs = [c.frequency for c in ranked]
    assert freqs == sorted(freqs)


def test_rank_anchors_zero_frequency_candidates_excluded():
    idx = _index_from_words(["my", "mind"])
    target = ["my", "mind", "rebels"]  # "rebels" never occurs
    candidates = generate_anchor_candidates(target, idx, max_ngram=1)
    ranked = rank_anchors(candidates)
    assert all(c.frequency > 0 for c in ranked)
    assert "rebels" not in [c.text[0] for c in ranked]


def test_rank_anchors_deterministic_tie_break_by_target_position():
    idx = _index_from_words(["dog", "cat"])  # both frequency 1
    target = ["dog", "cat"]
    candidates = generate_anchor_candidates(target, idx, max_ngram=1)
    ranked = rank_anchors(candidates)
    assert ranked[0].frequency == ranked[1].frequency == 1
    # earlier target position wins the tie, not iteration/dict order
    assert ranked[0].text == ("dog",)
    assert ranked[1].text == ("cat",)


def test_rank_anchors_is_stable_across_repeated_calls():
    idx = _index_from_words(["dog", "cat", "bird"])
    target = ["dog", "cat", "bird"]
    candidates = generate_anchor_candidates(target, idx, max_ngram=1)
    first = [c.text for c in rank_anchors(candidates)]
    second = [c.text for c in rank_anchors(candidates)]
    assert first == second


def test_select_anchors_common_word_only_target_falls_back_to_rare_bigram():
    # "and" and "then" are each individually frequent, but the contiguous
    # bigram "and then" occurs only once -- the bigram must outrank either
    # common unigram as an anchor.
    idx = _index_from_words(
        ["and", "q", "then", "and", "r", "then", "and", "then", "s", "t", "u"]
    )
    target = ["and", "then"]
    ranked = select_anchors(target, idx, max_ngram=2)
    assert ranked[0].text == ("and", "then")
    assert ranked[0].frequency == 1


def test_select_anchors_returns_empty_when_nothing_present():
    idx = _index_from_words(["completely", "unrelated", "words"])
    target = ["my", "mind", "rebels", "at", "stagnation"]
    assert select_anchors(target, idx) == []
