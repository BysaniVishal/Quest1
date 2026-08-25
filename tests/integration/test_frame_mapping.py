from fractions import Fraction

import pytest
from PIL import Image

from dialogue_frame_finder.frame_mapping import locate_frame, save_frame_image

from video_fixtures import make_synthetic_video

pytestmark = pytest.mark.integration


@pytest.fixture
def cfr_video(tmp_path):
    # 10fps, exact 0.1s spacing -- deliberately simple/exact for boundary tests
    path = tmp_path / "cfr.mp4"
    make_synthetic_video(path, fps=Fraction(10, 1), num_frames=20)
    return path


@pytest.fixture
def ntsc_video(tmp_path):
    # non-integer, real-world-style nominal FPS (matches the Phase 0
    # feasibility spike's actual OK.ru source rate)
    path = tmp_path / "ntsc.mp4"
    make_synthetic_video(path, fps=Fraction(24000, 1001), num_frames=30)
    return path


@pytest.fixture
def vfr_video(tmp_path):
    # irregular, non-uniform inter-frame spacing (milliseconds, via a 1000
    # time-base) -- simulates a variable frame rate source
    path = tmp_path / "vfr.mp4"
    pts_values = [0, 150, 340, 600, 1250, 1900]
    make_synthetic_video(path, fps=Fraction(1000, 1), num_frames=6, pts_values=pts_values)
    return path, [v / 1000.0 for v in pts_values]


def test_locate_frame_onset_exactly_equals_frame_pts(cfr_video):
    # FR-01
    result = locate_frame(cfr_video, target_seconds=0.6, seek_margin=0.5)
    assert result.pts_seconds == pytest.approx(0.6)


def test_locate_frame_onset_between_two_frame_pts(cfr_video):
    # FR-02: onset falls strictly between frame PTS 0.5 and 0.6 -- the
    # earlier frame must NOT be selected under the >= rule
    result = locate_frame(cfr_video, target_seconds=0.55, seek_margin=0.5)
    assert result.pts_seconds == pytest.approx(0.6)


def test_locate_frame_onset_before_first_frame(cfr_video):
    # FR-03
    result = locate_frame(cfr_video, target_seconds=-1.0, seek_margin=0.5)
    assert result.pts_seconds == pytest.approx(0.0)


def test_locate_frame_target_past_last_frame_raises(cfr_video):
    with pytest.raises(ValueError):
        locate_frame(cfr_video, target_seconds=100.0, seek_margin=0.5)


def test_locate_frame_different_nominal_fps_uses_actual_pts(ntsc_video):
    # FR-04: selection must come from decoder PTS, not a hardcoded FPS formula.
    # frame k's true pts = k * 1001/24000 s; verify against a mid-video onset.
    time_base = Fraction(1001, 24000)
    target = float(10 * time_base) + 0.001  # just after frame 10's pts
    result = locate_frame(ntsc_video, target_seconds=target, seek_margin=1.0)
    assert result.pts_seconds == pytest.approx(float(11 * time_base), abs=1e-4)


def test_locate_frame_variable_frame_rate_selection_is_correct(vfr_video):
    # FR-05
    path, pts_seconds = vfr_video
    # onset strictly between frame 2 (0.34s) and frame 3 (0.60s)
    result = locate_frame(path, target_seconds=0.5, seek_margin=1.0)
    assert result.pts_seconds == pytest.approx(pts_seconds[3])


def test_locate_frame_variable_frame_rate_exact_boundary(vfr_video):
    path, pts_seconds = vfr_video
    result = locate_frame(path, target_seconds=pts_seconds[4], seek_margin=1.0)
    assert result.pts_seconds == pytest.approx(pts_seconds[4])


def test_locate_frame_image_is_valid_and_reopenable(cfr_video, tmp_path):
    # FR-06
    result = locate_frame(cfr_video, target_seconds=0.3, seek_margin=0.5)
    assert result.image.shape == (32, 32, 3)
    out_path = tmp_path / "frame.png"
    save_frame_image(result.image, out_path)
    assert out_path.exists()
    reopened = Image.open(out_path)
    assert reopened.size == (32, 32)
    assert reopened.mode == "RGB"


def test_locate_frame_result_is_self_consistent(cfr_video):
    # FR-07: re-locating at the exact timestamp a prior call reported must
    # yield the same frame back -- if it didn't, that would mean the
    # reported timestamp and the actually-extracted frame disagree, which
    # must never silently pass.
    first = locate_frame(cfr_video, target_seconds=0.55, seek_margin=0.5)
    second = locate_frame(cfr_video, target_seconds=first.pts_seconds, seek_margin=0.5)
    assert second.pts_seconds == pytest.approx(first.pts_seconds)


def test_locate_frame_exact_frame_number_matches_known_index(cfr_video):
    result = locate_frame(cfr_video, target_seconds=0.55, exact_frame_number=True)
    assert result.frame_number == 6  # 0.6s at 10fps is decode index 6


def test_locate_frame_efficient_path_frame_number_is_none(cfr_video):
    result = locate_frame(cfr_video, target_seconds=0.55, seek_margin=0.5)
    assert result.frame_number is None


def test_locate_frame_seek_margin_zero_still_finds_correct_frame(cfr_video):
    result = locate_frame(cfr_video, target_seconds=0.55, seek_margin=0.0)
    assert result.pts_seconds == pytest.approx(0.6)


def test_locate_frame_no_video_stream_raises_clear_error(tmp_path):
    # audio-only media (e.g. an extracted audio track, or a source with no
    # video) must fail with a clear, documented error -- not a raw
    # IndexError from touching an empty stream list.
    import wave
    path = tmp_path / "audio_only.wav"
    with wave.open(str(path), "w") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes((b"\x00\x00") * 16000)

    with pytest.raises(ValueError, match="no video stream"):
        locate_frame(path, target_seconds=0.5)
