import pytest

from dialogue_frame_finder.config import SearchConfig
from dialogue_frame_finder.semantic import (
    SemanticCandidateInput,
    SemanticMatchResult,
    apply_semantic_fallback,
    build_semantic_prompt,
    gather_semantic_candidates,
    parse_semantic_response,
    should_use_semantic_fallback,
)
from dialogue_frame_finder.search import search_dialogue
from dialogue_frame_finder.transcript import Transcript
from dialogue_frame_finder.verification import VerificationResult

pytestmark = pytest.mark.unit

TARGET = "My mind rebels at stagnation"


def _transcript_from_words(words):
    tuples = [(w, i * 0.4, i * 0.4 + 0.35) for i, w in enumerate(words)]
    return Transcript.from_word_tuples(tuples)


def _verification(score, valid=True, first_word_start=1.0):
    return VerificationResult(
        window_start=0, window_end=4, alignment=[], lexical_similarity=score,
        coverage=score, contiguity=score, score=score, valid=valid,
        first_word_position=0, first_word_start=first_word_start, first_word_end=first_word_start + 0.3,
    )


class _FakeMatcher:
    def __init__(self, result):
        self._result = result
        self.calls = []

    def match(self, target_text, candidates):
        self.calls.append((target_text, candidates))
        return self._result


# --- prompt building / parsing -------------------------------------------

def test_build_semantic_prompt_includes_target_and_candidates():
    candidates = [SemanticCandidateInput("c0", "my mind rebels at stagnation", 1.3, (0, 4))]
    prompt = build_semantic_prompt(TARGET, candidates)
    assert TARGET in prompt
    assert "id=c0" in prompt
    assert "my mind rebels at stagnation" in prompt
    assert "Do not invent words or timestamps" in prompt


def test_parse_semantic_response_valid_json():
    raw = '{"candidate_id": "c1", "match_confidence": 0.82, "rationale": "close match"}'
    result = parse_semantic_response(raw, valid_candidate_ids=["c0", "c1"])
    assert result.candidate_id == "c1"
    assert result.match_confidence == pytest.approx(0.82)


def test_parse_semantic_response_tolerates_surrounding_text():
    raw = 'Here is my answer:\n{"candidate_id": "c0", "match_confidence": 0.9, "rationale": "ok"}\nThanks!'
    result = parse_semantic_response(raw, valid_candidate_ids=["c0"])
    assert result.candidate_id == "c0"


def test_parse_semantic_response_rejects_hallucinated_candidate_id():
    # the model named a candidate that was never actually offered to it
    raw = '{"candidate_id": "c99", "match_confidence": 0.95, "rationale": "made up"}'
    result = parse_semantic_response(raw, valid_candidate_ids=["c0", "c1"])
    assert result.candidate_id is None
    assert result.match_confidence == 0.0


def test_parse_semantic_response_null_candidate_id_is_no_match():
    raw = '{"candidate_id": null, "match_confidence": 0.0, "rationale": "nothing plausible"}'
    result = parse_semantic_response(raw, valid_candidate_ids=["c0"])
    assert result.candidate_id is None


def test_parse_semantic_response_malformed_json_is_no_match():
    result = parse_semantic_response("not json at all", valid_candidate_ids=["c0"])
    assert result.candidate_id is None
    assert result.match_confidence == 0.0


def test_parse_semantic_response_missing_json_entirely():
    result = parse_semantic_response("", valid_candidate_ids=["c0"])
    assert result.candidate_id is None


# --- candidate gathering ---------------------------------------------------

def test_gather_semantic_candidates_covers_transcript():
    words = ["well", "i", "think", "my", "mind", "rebels", "at", "stagnation"]
    transcript = _transcript_from_words(words)
    candidates = gather_semantic_candidates(["my", "mind", "rebels", "at", "stagnation"], transcript)
    assert len(candidates) >= 1
    assert all(c.candidate_id.startswith("c") for c in candidates)


def test_gather_semantic_candidates_respects_max_cap():
    words = ["word"] * 200
    transcript = _transcript_from_words(words)
    config = SearchConfig(semantic_max_candidates=3)
    candidates = gather_semantic_candidates(["a", "b"], transcript, config)
    assert len(candidates) <= 3


# --- trigger condition -------------------------------------------------

def test_should_use_semantic_fallback_when_no_match():
    result = search_dialogue(TARGET, _transcript_from_words(["unrelated", "words", "only"]))
    assert should_use_semantic_fallback(result) is True


def test_should_not_use_semantic_fallback_on_confident_lexical_match():
    result = search_dialogue(TARGET, _transcript_from_words(
        ["my", "mind", "rebels", "at", "stagnation"]
    ))
    assert should_use_semantic_fallback(result) is False


def test_should_use_semantic_fallback_when_ambiguous():
    from dialogue_frame_finder.search import SearchResult
    chosen = _verification(0.90, first_word_start=1.0)
    competitor = _verification(0.89, first_word_start=50.0)
    result = SearchResult(chosen=chosen, other_valid=[competitor], tier_used="exact_anchor", anchors_used=[], windows_verified=2)
    assert should_use_semantic_fallback(result) is True


# --- apply_semantic_fallback orchestration --------------------------------

def test_apply_semantic_fallback_replaces_no_match_with_semantic_pick():
    # a genuine paraphrase: same idea, no lexical overlap worth matching on
    words = ["stagnation", "is", "something", "my", "thoughts", "resist", "deeply"]
    transcript = _transcript_from_words(words)
    result = search_dialogue(TARGET, transcript)
    assert result.chosen is None  # weak lexical evidence -- paraphrase, not a lexical match

    candidates = gather_semantic_candidates(["my", "mind", "rebels", "at", "stagnation"], transcript)
    picked_id = candidates[0].candidate_id
    matcher = _FakeMatcher(SemanticMatchResult(candidate_id=picked_id, match_confidence=0.75, rationale="paraphrase"))

    updated = apply_semantic_fallback(result, TARGET, transcript, matcher)
    assert updated.tier_used == "semantic_fallback"
    assert updated.chosen is not None
    assert updated.chosen.score == pytest.approx(0.75)
    assert updated.chosen.valid is True


def test_apply_semantic_fallback_leaves_result_unchanged_when_matcher_finds_nothing():
    words = ["completely", "unrelated", "content"]
    transcript = _transcript_from_words(words)
    result = search_dialogue(TARGET, transcript)
    matcher = _FakeMatcher(SemanticMatchResult(candidate_id=None, match_confidence=0.0, rationale="nothing plausible"))

    updated = apply_semantic_fallback(result, TARGET, transcript, matcher)
    assert updated is result


def test_apply_semantic_fallback_rejects_low_confidence_match():
    words = ["completely", "unrelated", "content", "here"]
    transcript = _transcript_from_words(words)
    result = search_dialogue(TARGET, transcript)
    candidates = gather_semantic_candidates(["my", "mind", "rebels", "at", "stagnation"], transcript)
    matcher = _FakeMatcher(SemanticMatchResult(candidate_id=candidates[0].candidate_id, match_confidence=0.3, rationale="weak guess"))

    config = SearchConfig(semantic_min_confidence=0.6)
    updated = apply_semantic_fallback(result, TARGET, transcript, matcher, config)
    assert updated is result  # below semantic_min_confidence -- not applied


def test_apply_semantic_fallback_not_invoked_on_confident_lexical_match():
    transcript = _transcript_from_words(["my", "mind", "rebels", "at", "stagnation"])
    result = search_dialogue(TARGET, transcript)
    matcher = _FakeMatcher(SemanticMatchResult(candidate_id="should-not-be-used", match_confidence=0.99, rationale="n/a"))

    updated = apply_semantic_fallback(result, TARGET, transcript, matcher)
    assert updated is result
    assert matcher.calls == []  # never even consulted


def test_apply_semantic_fallback_empty_transcript_no_candidates():
    transcript = Transcript(words=[])
    result = search_dialogue(TARGET, transcript)
    matcher = _FakeMatcher(SemanticMatchResult(candidate_id=None, match_confidence=0.0, rationale=""))
    updated = apply_semantic_fallback(result, TARGET, transcript, matcher)
    assert updated is result
