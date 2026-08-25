import wave
from fractions import Fraction

import numpy as np
import pytest

from dialogue_frame_finder.audio import extract_audio_clip

from video_fixtures import make_synthetic_video

pytestmark = pytest.mark.integration


def _make_wav(path, duration=1.0, source_rate=44100, channels=2, freq=440.0, amplitude=0.3):
    n = int(duration * source_rate)
    t = np.arange(n) / source_rate
    tone = (amplitude * np.sin(2 * np.pi * freq * t) * 32767).astype(np.int16)
    with wave.open(str(path), "w") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)
        w.setframerate(source_rate)
        if channels == 2:
            stereo = np.repeat(tone.reshape(-1, 1), 2, axis=1)
            w.writeframes(stereo.tobytes())
        else:
            w.writeframes(tone.tobytes())


def test_extract_audio_clip_resamples_to_target_rate(tmp_path):
    path = tmp_path / "tone.wav"
    _make_wav(path, duration=1.0, source_rate=44100)
    clip = extract_audio_clip(path, sample_rate=16000)
    assert clip.sample_rate == 16000
    assert clip.samples.shape[0] == pytest.approx(16000, abs=5)


def test_extract_audio_clip_duration_matches_source(tmp_path):
    path = tmp_path / "tone.wav"
    _make_wav(path, duration=2.5, source_rate=48000)
    clip = extract_audio_clip(path, sample_rate=16000)
    assert clip.duration == pytest.approx(2.5, abs=0.01)


def test_extract_audio_clip_samples_are_in_valid_range(tmp_path):
    path = tmp_path / "tone.wav"
    _make_wav(path, duration=0.5, amplitude=0.5)
    clip = extract_audio_clip(path, sample_rate=16000)
    assert clip.samples.max() <= 1.0
    assert clip.samples.min() >= -1.0
    assert clip.samples.max() > 0.1  # actually has signal, not silence


def test_extract_audio_clip_mono_source(tmp_path):
    path = tmp_path / "mono.wav"
    _make_wav(path, duration=0.5, channels=1)
    clip = extract_audio_clip(path, sample_rate=16000)
    assert clip.samples.ndim == 1


def test_extract_audio_clip_no_audio_stream_returns_empty_clip(tmp_path):
    path = tmp_path / "video_only.mp4"
    make_synthetic_video(path, fps=Fraction(10, 1), num_frames=5)
    clip = extract_audio_clip(path, sample_rate=16000)
    assert len(clip.samples) == 0
    assert clip.sample_rate == 16000
