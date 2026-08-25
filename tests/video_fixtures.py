"""Synthetic video generation for frame-mapping tests.

Per test-plan.docx: don't commit real/copyrighted video; use small
synthetic/local fixtures with known FPS and known frame timestamps for exact
frame tests. Not a test module itself (no test_ functions) -- imported by
tests/integration/test_frame_mapping.py.
"""

from fractions import Fraction
from pathlib import Path
from typing import List, Optional, Union

import av
import numpy as np


def make_synthetic_video(
    path: Union[str, Path],
    fps: Fraction,
    num_frames: int,
    width: int = 32,
    height: int = 32,
    pts_values: Optional[List[int]] = None,
    codec: str = "mpeg4",
) -> None:
    """Encode a tiny synthetic video with distinguishable, deterministic
    frame content (frame index baked into a channel) and known timestamps.

    By default frames are evenly spaced (constant frame rate) at `1/fps`
    apart. Pass explicit `pts_values` (in the stream's time_base, i.e.
    1/fps units) for irregular/VFR-like spacing instead.
    """
    container = av.open(str(path), mode="w")
    stream = container.add_stream(codec, rate=fps)
    stream.width = width
    stream.height = height
    stream.pix_fmt = "yuv420p"
    stream.time_base = Fraction(1, 1) / fps

    for i in range(num_frames):
        array = np.zeros((height, width, 3), dtype=np.uint8)
        array[:, :, 0] = i % 256  # bake frame index into the red channel
        frame = av.VideoFrame.from_ndarray(array, format="rgb24")
        frame.pts = pts_values[i] if pts_values is not None else i
        for packet in stream.encode(frame):
            container.mux(packet)

    for packet in stream.encode():
        container.mux(packet)
    container.close()


def make_synthetic_media(
    path: Union[str, Path],
    duration: float,
    video_fps: Fraction,
    audio_regions: List[tuple],
    width: int = 32,
    height: int = 32,
    sample_rate: int = 16000,
    seed: int = 0,
) -> None:
    """Encode a combined video+audio synthetic media file: a video track at
    `video_fps` for the given duration, and an audio track that is silent
    except for `audio_regions` -- a list of (start_sec, end_sec, amplitude)
    noise bursts standing in for "speech." Used for a real, fully local
    integration test of decode -> onset refinement -> frame mapping working
    together on one actual media file (ASR itself is still injected as a
    fixture transcript matching these regions -- see test_full_pipeline.py).
    """
    container = av.open(str(path), mode="w")
    vstream = container.add_stream("mpeg4", rate=video_fps)
    vstream.width = width
    vstream.height = height
    vstream.pix_fmt = "yuv420p"

    astream = container.add_stream("aac", rate=sample_rate)

    total_samples = int(duration * sample_rate)
    audio = np.zeros(total_samples, dtype=np.float64)
    rng = np.random.default_rng(seed)
    for start, end, amplitude in audio_regions:
        s, e = int(start * sample_rate), int(end * sample_rate)
        audio[s:e] = amplitude * rng.standard_normal(e - s)
    pcm = (audio * 32767).astype(np.int16)

    frame_size = astream.codec_context.frame_size or 1024
    for i in range(0, len(pcm), frame_size):
        chunk = pcm[i:i + frame_size]
        if len(chunk) < frame_size:
            chunk = np.pad(chunk, (0, frame_size - len(chunk)))
        frame = av.AudioFrame.from_ndarray(chunk.reshape(1, -1), format="s16", layout="mono")
        frame.sample_rate = sample_rate
        frame.pts = i
        for packet in astream.encode(frame):
            container.mux(packet)
    for packet in astream.encode(None):
        container.mux(packet)

    num_video_frames = int(duration * video_fps)
    for i in range(num_video_frames):
        array = np.zeros((height, width, 3), dtype=np.uint8)
        frame = av.VideoFrame.from_ndarray(array, format="rgb24")
        frame.pts = i
        for packet in vstream.encode(frame):
            container.mux(packet)
    for packet in vstream.encode(None):
        container.mux(packet)

    container.close()
