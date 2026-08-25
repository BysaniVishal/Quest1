"""Verifies run_pipeline's DEFAULT locate_frame_fn (used whenever a caller
doesn't inject its own, e.g. the CLI) actually reports a frame number rather
than always leaving it None. This is an output-contract requirement
(OUT-02, "frame number where applicable") that a purely mocked pipeline
test cannot catch, since Phase 5's test_pipeline.py always injects its own
locate_frame_fn -- real video decode is needed here, not the algorithm
change (frame_mapping.py itself is untouched)."""

from fractions import Fraction

import pytest

from dialogue_frame_finder.pipeline import _default_locate_frame

from video_fixtures import make_synthetic_video

pytestmark = pytest.mark.integration


def test_default_locate_frame_reports_exact_frame_number(tmp_path):
    path = tmp_path / "cfr.mp4"
    make_synthetic_video(path, fps=Fraction(10, 1), num_frames=20)

    result = _default_locate_frame(str(path), target_seconds=0.55)

    assert result.frame_number is not None
    assert result.frame_number == 6  # 0.6s at 10fps is decode index 6
    assert result.pts_seconds == pytest.approx(0.6)
