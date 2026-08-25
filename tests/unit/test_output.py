import pytest

from dialogue_frame_finder.align import AlignmentEntry
from dialogue_frame_finder.config import SearchConfig
from dialogue_frame_finder.frame_mapping import FrameResult
from dialogue_frame_finder.onset import OnsetRefinementResult
from dialogue_frame_finder.output import MatchStatus, build_output_record, classify_confidence, extract_matched_text
from dialogue_frame_finder.search import SearchResult
from dialogue_frame_finder.transcript import Transcript
from dialogue_frame_finder.verification import VerificationResult

import numpy as np

pytestmark = pytest.mark.unit


def _verification(score, first_word_start=1.0, window_start=0, alignment=None):
    return VerificationResult(
        window_start=window_start, window_end=window_start + 4, alignment=alignment or [],
        lexical_similarity=score, coverage=score, contiguity=score, score=score, valid=True,
        first_word_position=window_start, first_word_start=first_word_start, first_word_end=first_word_start + 0.3,
    )


def _search_result(chosen, other_valid=None, tier_used="exact_anchor"):
    return SearchResult(chosen=chosen, other_valid=other_valid or [], tier_used=tier_used, anchors_used=[], windows_verified=1)


def test_classify_confidence_no_match():
    result = _search_result(None, tier_used="none")
    assert classify_confidence(result) == MatchStatus.NO_CONFIDENT_MATCH


def test_classify_confidence_high():
    result = _search_result(_verification(0.95))
    assert classify_confidence(result) == MatchStatus.HIGH_CONFIDENCE


def test_classify_confidence_medium():
    result = _search_result(_verification(0.70))
    assert classify_confidence(result) == MatchStatus.MEDIUM_CONFIDENCE


def test_classify_confidence_low():
    result = _search_result(_verification(0.61))
    assert classify_confidence(result) == MatchStatus.LOW_CONFIDENCE


def test_classify_confidence_ambiguous_when_close_competitor_exists():
    # TM-07: "same phrase occurs twice" -- close scores should flag ambiguity
    # even though the chosen (earliest) occurrence is otherwise high-scoring
    chosen = _verification(0.90, first_word_start=5.0)
    competitor = _verification(0.89, first_word_start=50.0)
    result = _search_result(chosen, other_valid=[competitor])
    assert classify_confidence(result) == MatchStatus.AMBIGUOUS_MATCH


def test_classify_confidence_not_ambiguous_when_competitor_far_weaker():
    chosen = _verification(0.90, first_word_start=5.0)
    weak_other = _verification(0.62, first_word_start=50.0)
    result = _search_result(chosen, other_valid=[weak_other])
    assert classify_confidence(result) == MatchStatus.HIGH_CONFIDENCE


def test_classify_confidence_ambiguity_margin_is_configurable():
    chosen = _verification(0.90, first_word_start=5.0)
    competitor = _verification(0.80, first_word_start=50.0)
    result = _search_result(chosen, other_valid=[competitor])
    assert classify_confidence(result, SearchConfig(ambiguity_score_margin=0.02)) == MatchStatus.HIGH_CONFIDENCE
    assert classify_confidence(result, SearchConfig(ambiguity_score_margin=0.15)) == MatchStatus.AMBIGUOUS_MATCH


def test_extract_matched_text_reconstructs_original_words():
    transcript = Transcript.from_word_tuples(
        [("Well", 0, 0.3), ("I", 0.3, 0.4), ("think", 0.4, 0.7),
         ("My", 0.7, 1.0), ("mind", 1.0, 1.3), ("rebels", 1.3, 1.6),
         ("at", 1.6, 1.8), ("stagnation.", 1.8, 2.3)]
    )
    alignment = [
        AlignmentEntry(target_index=0, window_index=3, similarity=1.0),
        AlignmentEntry(target_index=1, window_index=4, similarity=1.0),
        AlignmentEntry(target_index=2, window_index=5, similarity=1.0),
        AlignmentEntry(target_index=3, window_index=6, similarity=1.0),
        AlignmentEntry(target_index=4, window_index=7, similarity=1.0),
    ]
    chosen = _verification(1.0, window_start=0, alignment=alignment)
    text = extract_matched_text(chosen, transcript)
    assert text == "My mind rebels at stagnation."


def test_extract_matched_text_empty_when_nothing_matched():
    transcript = Transcript.from_word_tuples([("hello", 0, 0.3)])
    chosen = _verification(0.0, alignment=[AlignmentEntry(0, None, 0.0)])
    assert extract_matched_text(chosen, transcript) == ""


def test_build_output_record_no_match_has_none_fields():
    result = _search_result(None, tier_used="bounded_scan")
    transcript = Transcript.from_word_tuples([("hello", 0, 0.3)])
    record = build_output_record(result, transcript, onset_result=None, frame_result=None, image_path=None)
    assert record.status == MatchStatus.NO_CONFIDENT_MATCH
    assert record.timestamp is None
    assert record.frame is None
    assert record.text is None
    assert record.image_path is None
    assert record.confidence_score is None
    assert record.diagnostics["tier_used"] == "bounded_scan"


def test_build_output_record_full_success_case():
    transcript = Transcript.from_word_tuples(
        [("My", 0.7, 1.0), ("mind", 1.0, 1.3), ("rebels", 1.3, 1.6), ("at", 1.6, 1.8), ("stagnation", 1.8, 2.3)]
    )
    alignment = [AlignmentEntry(i, i, 1.0) for i in range(5)]
    chosen = _verification(0.95, first_word_start=0.7, window_start=0, alignment=alignment)
    search_result = _search_result(chosen)
    onset_result = OnsetRefinementResult(asr_onset=0.7, refined_onset=0.68, refined=True, reason="local_transition_detected")
    frame_result = FrameResult(pts_seconds=0.68, frame_number=17, image=np.zeros((4, 4, 3), dtype=np.uint8))

    record = build_output_record(search_result, transcript, onset_result, frame_result, "outputs/frame_0.68.png")

    assert record.status == MatchStatus.HIGH_CONFIDENCE
    assert record.timestamp == "00:00:00.680"
    assert record.frame == 17
    assert record.text == "My mind rebels at stagnation"
    assert record.image_path == "outputs/frame_0.68.png"
    assert record.confidence_score == pytest.approx(0.95)
    assert record.diagnostics["onset_refined"] is True
    assert record.diagnostics["frame_pts_seconds"] == pytest.approx(0.68)


def test_build_output_record_timestamp_matches_extracted_frame_not_raw_onset():
    # OUT-05: when the refined onset and the extracted frame's own PTS
    # differ (as they do in practice -- "first frame with PTS >= onset"
    # generally lands slightly after the onset, not exactly on it), the
    # reported Timestamp must match the frame that was actually extracted,
    # not the intermediate onset estimate used to locate it.
    transcript = Transcript.from_word_tuples(
        [("My", 325.30, 325.55), ("mind", 325.55, 325.80), ("rebels", 325.80, 326.05),
         ("at", 326.05, 326.25), ("stagnation", 326.25, 326.70)]
    )
    alignment = [AlignmentEntry(i, i, 1.0) for i in range(5)]
    chosen = _verification(0.9, first_word_start=325.30, window_start=0, alignment=alignment)
    search_result = _search_result(chosen)
    onset_result = OnsetRefinementResult(
        asr_onset=325.30, refined_onset=325.28, refined=True, reason="local_transition_detected"
    )
    frame_result = FrameResult(pts_seconds=325.305, frame_number=7812, image=np.zeros((4, 4, 3), dtype=np.uint8))

    record = build_output_record(search_result, transcript, onset_result, frame_result, "out/f.png")

    assert record.timestamp == "00:05:25.305"  # frame's own PTS, not 325.28
    assert record.frame == 7812


def test_build_output_record_falls_back_to_onset_when_no_frame_result():
    transcript = Transcript.from_word_tuples([("my", 1.1, 1.4)])
    alignment = [AlignmentEntry(0, 0, 1.0)]
    chosen = _verification(0.9, first_word_start=1.1, window_start=0, alignment=alignment)
    search_result = _search_result(chosen)
    onset_result = OnsetRefinementResult(asr_onset=1.1, refined_onset=1.08, refined=True, reason="local_transition_detected")

    record = build_output_record(search_result, transcript, onset_result, frame_result=None, image_path=None)
    assert record.timestamp == "00:00:01.080"
    assert record.frame is None


def test_build_output_record_falls_back_to_asr_onset_when_no_refinement():
    transcript = Transcript.from_word_tuples([("my", 1.1, 1.4)])
    alignment = [AlignmentEntry(0, 0, 1.0)]
    chosen = _verification(0.9, first_word_start=1.1, window_start=0, alignment=alignment)
    search_result = _search_result(chosen)
    record = build_output_record(search_result, transcript, onset_result=None, frame_result=None, image_path=None)
    assert record.timestamp == "00:00:01.100"
