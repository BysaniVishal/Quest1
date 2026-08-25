"""Dialogue-onset resolution: first-target-word ASR timestamp (primary) plus
bounded, directional local audio refinement (secondary).

The requirement is the onset of the target dialogue's FIRST WORD -- not
"where did speech begin" in the surrounding audio. VAD/energy analysis is
therefore never the primary signal and is never allowed to search backward
into unrelated earlier speech: the refinement window is small, starts at
most `onset_pre_roll` seconds before the ASR timestamp, and a rise (silence
-> speech transition) is required within that bounded window before the
timestamp is ever moved. If the window shows continuous speech throughout
(no local transition to snap to -- e.g. the target follows other dialogue
with no pause, or another speaker is already talking), refinement is a
deliberate no-op: the ASR timestamp is kept, with a reduced-confidence
reason recorded rather than a silent guess.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from .config import DEFAULT_CONFIG, SearchConfig
from .search import SearchResult


@dataclass(frozen=True)
class AudioClip:
    samples: np.ndarray  # 1D mono samples
    sample_rate: int

    @property
    def duration(self) -> float:
        return len(self.samples) / self.sample_rate if self.sample_rate else 0.0

    def slice(self, start: float, end: float) -> "AudioClip":
        start = max(0.0, start)
        end = min(self.duration, end)
        start_idx = int(round(start * self.sample_rate))
        end_idx = int(round(end * self.sample_rate))
        return AudioClip(samples=self.samples[start_idx:end_idx], sample_rate=self.sample_rate)


def _smooth(values: List[float], window: int = 3) -> List[float]:
    if len(values) < window:
        return list(values)
    half = window // 2
    n = len(values)
    return [
        sum(values[max(0, i - half):min(n, i + half + 1)])
        / (min(n, i + half + 1) - max(0, i - half))
        for i in range(n)
    ]


def frame_rms(clip: AudioClip, frame_ms: float, hop_ms: float) -> List[Tuple[float, float]]:
    """(frame_start_time_within_clip, rms) for consecutive, possibly
    overlapping frames covering the clip."""
    frame_len = max(1, int(round(clip.sample_rate * frame_ms / 1000.0)))
    hop_len = max(1, int(round(clip.sample_rate * hop_ms / 1000.0)))
    n = len(clip.samples)
    frames: List[Tuple[float, float]] = []
    start = 0
    while start < n:
        end = min(n, start + frame_len)
        segment = clip.samples[start:end]
        rms = float(np.sqrt(np.mean(np.square(segment)))) if len(segment) > 0 else 0.0
        frames.append((start / clip.sample_rate, rms))
        if end >= n:
            break
        start += hop_len
    return frames


@dataclass(frozen=True)
class OnsetRefinementResult:
    asr_onset: float
    refined_onset: float
    refined: bool
    reason: str
    # "local_transition_detected" | "no_local_transition_continuous_speech"
    # | "no_speech_detected" | "clip_unavailable"


def refine_onset(
    clip: Optional[AudioClip], asr_onset: float, config: SearchConfig = DEFAULT_CONFIG
) -> OnsetRefinementResult:
    if clip is None:
        return OnsetRefinementResult(asr_onset, asr_onset, False, "clip_unavailable")

    window_start = max(0.0, asr_onset - config.onset_pre_roll)
    window_end = min(clip.duration, asr_onset + config.onset_post_roll)
    if window_end <= window_start:
        return OnsetRefinementResult(asr_onset, asr_onset, False, "clip_unavailable")

    window_clip = clip.slice(window_start, window_end)
    frames = frame_rms(window_clip, config.onset_frame_ms, config.onset_hop_ms)
    if len(frames) < 2:
        return OnsetRefinementResult(asr_onset, asr_onset, False, "no_speech_detected")

    # Smooth before classifying: real speech onset is a SUSTAINED rise over
    # several consecutive frames, not a single-frame blip. Without
    # smoothing, ordinary frame-to-frame energy variance within continuous
    # speech (or within continuous silence) can look like a rise/fall under
    # naive min-max normalization over a short window.
    raw_energies = [e for _, e in frames]
    energies = _smooth(raw_energies, window=3)
    lo, hi = min(energies), max(energies)

    mean_energy = sum(energies) / len(energies)
    if mean_energy < config.onset_absolute_silence_floor:
        return OnsetRefinementResult(asr_onset, asr_onset, False, "no_speech_detected")

    # A genuine silence -> speech transition swings energy by an order of
    # magnitude; ordinary variance within uniform audio does not. Without
    # this gate, per-window min-max normalization alone would always find
    # SOME frame near the window's own minimum and mistake it for "silence"
    # relative to the window's own maximum, even in continuous speech.
    if hi < lo * config.onset_dynamic_range_ratio:
        return OnsetRefinementResult(asr_onset, asr_onset, False, "no_local_transition_continuous_speech")

    frame_times = [t for t, _ in frames]
    normalized = [(t, (e - lo) / (hi - lo)) for t, e in zip(frame_times, energies)]
    speech_like = [n >= config.onset_speech_ratio_threshold for _, n in normalized]

    for i in range(1, len(speech_like)):
        if speech_like[i] and not speech_like[i - 1]:
            refined_onset = window_start + normalized[i][0]
            return OnsetRefinementResult(asr_onset, refined_onset, True, "local_transition_detected")

    if speech_like[0]:
        # already speech-like at the very start of the bounded window --
        # speech was already ongoing; do not extrapolate further back than
        # the window itself.
        return OnsetRefinementResult(asr_onset, asr_onset, False, "no_local_transition_continuous_speech")

    return OnsetRefinementResult(asr_onset, asr_onset, False, "no_speech_detected")


def resolve_dialogue_onset(
    result: SearchResult, clip: Optional[AudioClip], config: SearchConfig = DEFAULT_CONFIG
) -> Optional[OnsetRefinementResult]:
    """Combine Phase 2's chosen occurrence (primary onset = first-target-word
    ASR timestamp) with bounded local audio refinement. Returns None if no
    valid occurrence was found -- there is no onset to refine."""
    if result.chosen is None or result.chosen.first_word_start is None:
        return None
    return refine_onset(clip, result.chosen.first_word_start, config)
