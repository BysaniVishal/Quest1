"""Audio extraction: decode a media file's audio stream to mono, resampled
PCM samples -- feeds both onset refinement (Phase 3, onset.py) and ASR.

Uses PyAV directly rather than shelling out to an ffmpeg binary, so this has
no dependency on ffmpeg being present on PATH.
"""

from pathlib import Path
from typing import Union

import av
import numpy as np

from .onset import AudioClip


def extract_audio_clip(path: Union[str, Path], sample_rate: int = 16000) -> AudioClip:
    """Decode the first audio stream of `path`, downmixed to mono and
    resampled to `sample_rate`, as a float64 AudioClip in [-1, 1]."""
    path = str(path)
    with av.open(path) as container:
        if not container.streams.audio:
            return AudioClip(samples=np.array([], dtype=np.float64), sample_rate=sample_rate)

        stream = container.streams.audio[0]
        resampler = av.AudioResampler(format="s16", layout="mono", rate=sample_rate)

        chunks = []
        for frame in container.decode(stream):
            for resampled in resampler.resample(frame):
                chunks.append(resampled.to_ndarray())
        for resampled in resampler.resample(None):  # flush
            chunks.append(resampled.to_ndarray())

        if not chunks:
            return AudioClip(samples=np.array([], dtype=np.float64), sample_rate=sample_rate)

        raw = np.concatenate([c.reshape(-1) for c in chunks])
        samples = raw.astype(np.float64) / 32768.0
        return AudioClip(samples=samples, sample_rate=sample_rate)
