import numpy as np
import pytest

from dialogue_frame_finder.config import SearchConfig
from dialogue_frame_finder.onset import AudioClip, frame_rms, refine_onset, resolve_dialogue_onset
from dialogue_frame_finder.search import SearchResult
from dialogue_frame_finder.verification import VerificationResult

pytestmark = pytest.mark.unit

SAMPLE_RATE = 16000


def _tone(duration: float, amplitude: float = 0.5, seed: int = 0) -> np.ndarray:
    # band-limited noise, not a pure sine: a pure tone's RMS fluctuates with
    # frame-boundary phase alignment, which can spuriously look like an
    # energy dip. Noise has stable short-term RMS, closer to real speech.
    rng = np.random.default_rng(seed)
    return (amplitude * rng.standard_normal(int(duration * SAMPLE_RATE))).astype(np.float64)


def _silence(duration: float) -> np.ndarray:
    return np.zeros(int(duration * SAMPLE_RATE), dtype=np.float64)


def _clip(*segments: np.ndarray) -> AudioClip:
    return AudioClip(samples=np.concatenate(segments), sample_rate=SAMPLE_RATE)


def test_frame_rms_silence_is_near_zero():
    clip = _clip(_silence(0.5))
    frames = frame_rms(clip, frame_ms=20, hop_ms=10)
    assert all(rms < 1e-9 for _, rms in frames)


def test_frame_rms_tone_has_positive_energy():
    clip = _clip(_tone(0.5))
    frames = frame_rms(clip, frame_ms=20, hop_ms=10)
    assert all(rms > 0.1 for _, rms in frames)


def test_refine_onset_detects_clean_transition_and_corrects_late_asr_timestamp():
    # true acoustic onset is exactly at 1.0s; ASR (as usual) reports it a
    # little late/rounded at 1.08s
    clip = _clip(_silence(1.0), _tone(1.0))
    result = refine_onset(clip, asr_onset=1.08)
    assert result.refined is True
    assert result.reason == "local_transition_detected"
    assert result.refined_onset == pytest.approx(1.0, abs=0.03)
    assert result.refined_onset < result.asr_onset


def test_refine_onset_target_follows_other_speech_with_no_pause():
    # "Well, I think MY MIND rebels at stagnation" -- speech is continuous
    # from "Well" (0.3s) all the way through; the ASR timestamp for "My"
    # (the first TARGET word) is 1.10s. A naive/unbounded VAD pass over the
    # whole utterance would report onset at 0.3s ("Well"). The system must
    # keep 1.10s, never drift back to "Well"'s onset.
    clip = _clip(_silence(0.3), _tone(2.2))  # continuous speech from 0.3s to 2.5s
    asr_onset_for_my = 1.10
    result = refine_onset(clip, asr_onset=asr_onset_for_my)
    assert result.refined is False
    assert result.reason == "no_local_transition_continuous_speech"
    assert result.refined_onset == pytest.approx(asr_onset_for_my)
    # must not have snapped back toward "Well"'s general speech onset
    assert result.refined_onset != pytest.approx(0.3, abs=0.05)


def test_refine_onset_target_starts_amid_already_ongoing_speech():
    # a second speaker (or continuing dialogue) is already talking when the
    # target's first word begins -- no silence gap anywhere near it
    clip = _clip(_tone(3.0))  # speech from the very start of the clip
    asr_onset = 1.5
    result = refine_onset(clip, asr_onset=asr_onset)
    assert result.refined is False
    assert result.reason == "no_local_transition_continuous_speech"
    assert result.refined_onset == pytest.approx(asr_onset)


def test_refine_onset_falling_edge_within_window_leaves_asr_unchanged():
    # window captures the END of speech transitioning into silence (not a
    # silence -> speech rise) -- e.g. the ASR onset guess landed a little
    # late, so the bounded window mostly sees trailing speech then silence.
    # There IS a large dynamic range here, but no rising edge to snap to;
    # must not be mistaken for a detected transition.
    clip = _clip(_tone(1.1), _silence(1.0))
    result = refine_onset(clip, asr_onset=1.05)
    assert result.refined is False
    assert result.refined_onset == pytest.approx(1.05)


def test_refine_onset_no_speech_anywhere_in_window_leaves_asr_unchanged():
    clip = _clip(_silence(3.0))
    result = refine_onset(clip, asr_onset=1.5)
    assert result.refined is False
    assert result.reason == "no_speech_detected"
    assert result.refined_onset == pytest.approx(1.5)


def test_refine_onset_window_clipped_at_clip_start_does_not_error():
    clip = _clip(_silence(0.05), _tone(0.5))
    result = refine_onset(clip, asr_onset=0.06, config=SearchConfig(onset_pre_roll=0.15))
    assert result.refined_onset >= 0.0


def test_refine_onset_none_clip_leaves_asr_unchanged():
    result = refine_onset(None, asr_onset=1.23)
    assert result.refined is False
    assert result.reason == "clip_unavailable"
    assert result.refined_onset == pytest.approx(1.23)


def _make_search_result(first_word_start):
    verification = VerificationResult(
        window_start=0, window_end=0, alignment=[], lexical_similarity=1.0,
        coverage=1.0, contiguity=1.0, score=1.0, valid=True,
        first_word_position=0, first_word_start=first_word_start, first_word_end=first_word_start + 0.3,
    )
    return SearchResult(chosen=verification, other_valid=[], tier_used="exact_anchor", anchors_used=[], windows_verified=1)


def test_resolve_dialogue_onset_uses_chosen_first_word_start():
    clip = _clip(_silence(1.0), _tone(1.0))
    result = resolve_dialogue_onset(_make_search_result(1.08), clip)
    assert result is not None
    assert result.asr_onset == pytest.approx(1.08)


def test_resolve_dialogue_onset_returns_none_when_no_valid_occurrence():
    no_match = SearchResult(chosen=None, other_valid=[], tier_used="none", anchors_used=[], windows_verified=0)
    assert resolve_dialogue_onset(no_match, _clip(_tone(1.0))) is None
