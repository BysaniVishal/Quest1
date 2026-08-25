"""Real, fully local integration test of the complete chain: PyAV audio
decode -> indexed retrieval/verification (Phase 1+2) -> bounded onset
refinement on real decoded audio (Phase 3) -> PTS-aware frame extraction on
real decoded video (Phase 4) -> output contract (Phase 5), all wired through
the actual `run_pipeline` orchestration -- not fully mocked like the
Phase 5 pipeline tests.

Per test-plan.docx, ASR itself is a fixture (a hand-built Transcript whose
word timestamps match the synthetic audio's actual noise-burst regions)
rather than a real model -- "mock or real small ASR run" -- since this test
is about the pipeline's plumbing and real signal-processing correctness,
not about ASR accuracy.
"""

from fractions import Fraction

import pytest
from PIL import Image

from dialogue_frame_finder.media_resolver import ResolvedMedia
from dialogue_frame_finder.output import MatchStatus
from dialogue_frame_finder.pipeline import run_pipeline
from dialogue_frame_finder.frame_mapping import locate_frame as real_locate_frame
from dialogue_frame_finder.transcript import Transcript

from video_fixtures import make_synthetic_media

pytestmark = pytest.mark.integration

VIDEO_FPS = 25


class _FakeMediaResolver:
    def __init__(self, local_path, duration):
        self._result = ResolvedMedia(local_path=str(local_path), duration=duration, provider="generic")

    def resolve(self, url):
        return self._result


class _FakeASR:
    def __init__(self, transcript):
        self._transcript = transcript

    def transcribe(self, audio_path):
        return self._transcript


def _exact_locate_frame(path, target_seconds):
    return real_locate_frame(path, target_seconds, exact_frame_number=True)


def _parse_timestamp(ts: str) -> float:
    h, m, s = ts.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


@pytest.fixture
def combined_media(tmp_path):
    path = tmp_path / "combined.mp4"
    make_synthetic_media(
        path,
        duration=3.0,
        video_fps=Fraction(VIDEO_FPS, 1),
        audio_regions=[
            (0.5, 1.1, 0.25),   # "well i think" -- continuous, no target words here
            (1.3, 2.6, 0.25),   # clean 0.2s pause, then "my mind rebels at stagnation"
        ],
    )
    return path


def _transcript_with_late_my_timestamp():
    # word timestamps deliberately a little late/rounded for "My" (1.35s)
    # vs the true acoustic onset baked into the audio (1.3s) -- exercises
    # real onset refinement, not just the ASR value passed straight through
    return Transcript.from_word_tuples([
        ("Well", 0.50, 0.65),
        ("I", 0.65, 0.75),
        ("think", 0.75, 1.10),
        ("My", 1.35, 1.55),
        ("mind", 1.55, 1.75),
        ("rebels", 1.75, 2.00),
        ("at", 2.00, 2.10),
        ("stagnation", 2.10, 2.60),
    ])


def test_full_pipeline_real_decode_match_onset_frame_chain(combined_media, tmp_path):
    output_dir = tmp_path / "outputs"
    result = run_pipeline(
        "https://example.com/video",
        "My mind rebels at stagnation",
        output_dir=output_dir,
        media_resolver=_FakeMediaResolver(combined_media, duration=3.0),
        asr_adapter=_FakeASR(_transcript_with_late_my_timestamp()),
        locate_frame_fn=_exact_locate_frame,
    )
    record = result.output

    assert record.status in (
        MatchStatus.HIGH_CONFIDENCE, MatchStatus.MEDIUM_CONFIDENCE, MatchStatus.LOW_CONFIDENCE,
    )
    assert record.text == "My mind rebels at stagnation"

    # onset refinement must have pulled the timestamp toward the real
    # acoustic transition (~1.3s), not left it at the late ASR value (1.35s)
    refined_seconds = _parse_timestamp(record.timestamp)
    assert refined_seconds == pytest.approx(1.3, abs=0.05)
    assert refined_seconds < 1.35
    assert result.output.diagnostics["onset_refined"] is True

    # frame number must be consistent with the real video's actual FPS/PTS
    # (first frame whose PTS >= refined onset, at 25fps)
    assert record.frame is not None
    assert abs(record.frame - refined_seconds * VIDEO_FPS) < 2

    assert record.image_path is not None
    img = Image.open(record.image_path)
    assert img.size == (32, 32)


def test_full_pipeline_no_match_produces_no_confident_match(combined_media, tmp_path):
    output_dir = tmp_path / "outputs"
    result = run_pipeline(
        "https://example.com/video",
        "goodbye cruel world",
        output_dir=output_dir,
        media_resolver=_FakeMediaResolver(combined_media, duration=3.0),
        asr_adapter=_FakeASR(_transcript_with_late_my_timestamp()),
        locate_frame_fn=_exact_locate_frame,
    )
    record = result.output
    assert record.status == MatchStatus.NO_CONFIDENT_MATCH
    assert record.timestamp is None
    assert record.frame is None
    assert record.image_path is None
    # no frame image should have been written anywhere under the output dir
    assert list(output_dir.glob("**/*.png")) == []
