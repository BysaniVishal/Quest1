import pytest

from dialogue_frame_finder.index import TranscriptIndex
from dialogue_frame_finder.retrieval import retrieve_candidates
from dialogue_frame_finder.transcript import Transcript

pytestmark = pytest.mark.unit


def _index_from_words(words):
    tuples = [(w, i * 0.1, i * 0.1 + 0.1) for i, w in enumerate(words)]
    return TranscriptIndex.build(Transcript.from_word_tuples(tuples))


def test_retrieve_candidates_single_occurrence():
    idx = _index_from_words(
        ["well", "i", "think", "my", "mind", "rebels", "at", "stagnation"]
    )
    target = ["my", "mind", "rebels", "at", "stagnation"]
    hits, anchors = retrieve_candidates(target, idx)
    assert anchors  # at least one anchor produced hits
    assert hits  # at least one candidate position found
    assert all(3 <= h.position <= 7 for h in hits)


def test_retrieve_candidates_includes_rarest_unigram_anchor():
    idx = _index_from_words(
        ["well", "i", "think", "my", "mind", "rebels", "at", "stagnation"]
    )
    target = ["my", "mind", "rebels", "at", "stagnation"]
    _, anchors = retrieve_candidates(target, idx)
    unigram_anchors_used = [a for a in anchors if a.length == 1]
    assert unigram_anchors_used  # a single-word anchor was tried, not just an n-gram


def test_retrieve_candidates_multiple_occurrences_found_in_order():
    idx = _index_from_words(
        ["filler"] * 5
        + ["stagnation"]
        + ["filler"] * 10
        + ["stagnation"]
        + ["filler"] * 3
    )
    hits, anchors = retrieve_candidates(["stagnation"], idx)
    assert anchors[0].text == ("stagnation",)
    assert [h.position for h in hits] == [5, 16]
    # candidate timestamps come straight from the transcript, ascending
    assert hits[0].start < hits[1].start


def test_retrieve_candidates_anchor_at_transcript_start():
    words = ["stagnation", "and", "more", "filler", "words"]
    idx = _index_from_words(words)
    hits, anchors = retrieve_candidates(["stagnation"], idx)
    assert hits[0].position == 0


def test_retrieve_candidates_anchor_at_transcript_end():
    words = ["filler", "words", "before", "it", "stagnation"]
    idx = _index_from_words(words)
    hits, anchors = retrieve_candidates(["stagnation"], idx)
    assert hits[0].position == len(words) - 1


def test_retrieve_candidates_no_match_returns_empty_hits_and_anchors():
    idx = _index_from_words(["completely", "unrelated", "words"])
    hits, anchors = retrieve_candidates(["my", "mind", "rebels", "at", "stagnation"], idx)
    assert hits == []
    assert anchors == []


def test_retrieve_candidates_repeated_phrase_covers_every_occurrence():
    words = (
        ["well", "i", "think"]
        + ["my", "mind", "rebels", "at", "stagnation"]  # occurrence 1 starts at 3
        + ["filler"] * 8
        + ["my", "mind", "rebels", "at", "stagnation"]  # occurrence 2 starts at 16
        + ["filler"] * 8
        + ["my", "mind", "rebels", "at", "stagnation"]  # occurrence 3 starts at 29
    )
    idx = _index_from_words(words)
    target = ["my", "mind", "rebels", "at", "stagnation"]
    hits, anchors = retrieve_candidates(target, idx)
    positions = [h.position for h in hits]
    assert positions == sorted(positions)
    # every occurrence's span must be represented by at least one hit
    for occurrence_start in (3, 16, 29):
        assert any(occurrence_start <= p <= occurrence_start + 4 for p in positions)


def test_retrieve_candidates_recall_survives_different_word_corrupted_per_occurrence():
    # occurrence 1 has "at" -> "in" corrupted; occurrence 2 is clean. No single
    # anchor covers both unless retrieval tries more than the single most
    # selective (and therefore most occurrence-specific) anchor.
    words = (
        ["well", "i", "think", "my", "mind", "rebels", "in", "stagnation"]
        + ["filler"] * 10
        + ["my", "mind", "rebels", "at", "stagnation"]
    )
    idx = _index_from_words(words)
    target = ["my", "mind", "rebels", "at", "stagnation"]
    hits, anchors = retrieve_candidates(target, idx)
    positions = [h.position for h in hits]
    assert any(3 <= p <= 7 for p in positions)  # occurrence 1 (corrupted) still reachable
    assert any(18 <= p <= 22 for p in positions)  # occurrence 2 (clean)
