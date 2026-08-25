"""PTS-aware video frame extraction via PyAV.

Frame selection must never be computed as timestamp * nominal_FPS -- variable
frame rates and encoder rounding make that unreliable (design.docx section
9). The Phase 0 feasibility spike confirmed this with real media: even a
standard 23.98fps source had a non-uniform first inter-frame gap. Instead:
seek to the nearest keyframe at/before the target, decode forward, and take
the FIRST decoded frame whose actual presentation timestamp (PTS) is >= the
target -- the same operational "first frame" rule from design.docx section 8
and exercised by test-plan.docx FR-01..FR-07.

For a long video, decoding from the very start to get an exact frame index
is wasteful (design.docx section 13: only do fine-grained work in a small
neighborhood around the identified onset). So frame_number is reported only
when the caller explicitly asks for exact counting (short clips, tests) --
otherwise it is honestly None, per "frame number where applicable."
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import av
import numpy as np
from PIL import Image


@dataclass(frozen=True)
class FrameResult:
    pts_seconds: float
    frame_number: Optional[int]
    image: np.ndarray  # HxWx3 uint8, RGB


def _pts_seconds(frame, time_base) -> Optional[float]:
    if frame.pts is None:
        return None
    return float(frame.pts * time_base)


def _to_result(frame, pts_seconds: float, frame_number: Optional[int]) -> FrameResult:
    image = frame.to_ndarray(format="rgb24")
    return FrameResult(pts_seconds=pts_seconds, frame_number=frame_number, image=image)


def locate_frame(
    path: Union[str, Path],
    target_seconds: float,
    seek_margin: float = 2.0,
    exact_frame_number: bool = False,
) -> FrameResult:
    """Return the first decoded frame whose PTS >= target_seconds.

    By default, seeks to the nearest keyframe at or before
    `target_seconds - seek_margin` and decodes forward from there -- cheap
    even for a long video, but frame_number is None since the true index
    from the start of the video isn't known without a full decode.

    Pass exact_frame_number=True to decode from the very start instead and
    get an exact, reliable frame index -- appropriate for short clips or
    tests, not for locating a frame deep into a long video.
    """
    path = str(path)
    with av.open(path) as container:
        if not container.streams.video:
            raise ValueError(f"no video stream found in {path}")
        stream = container.streams.video[0]
        time_base = stream.time_base

        if exact_frame_number:
            index = 0
            for frame in container.decode(stream):
                pts_time = _pts_seconds(frame, time_base)
                if pts_time is not None and pts_time >= target_seconds:
                    return _to_result(frame, pts_time, index)
                index += 1
            raise ValueError(f"no frame with pts >= {target_seconds}s found in {path}")

        seek_target = max(0.0, target_seconds - seek_margin)
        container.seek(int(seek_target / time_base), stream=stream, backward=True, any_frame=False)
        for frame in container.decode(stream):
            pts_time = _pts_seconds(frame, time_base)
            if pts_time is not None and pts_time >= target_seconds:
                return _to_result(frame, pts_time, None)
        raise ValueError(f"no frame with pts >= {target_seconds}s found in {path}")


def save_frame_image(image: np.ndarray, output_path: Union[str, Path]) -> None:
    Image.fromarray(image, mode="RGB").save(str(output_path))
