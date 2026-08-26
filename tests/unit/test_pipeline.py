from pathlib import Path

import numpy as np
import pytest

from dialogue_frame_finder.frame_mapping import FrameResult
from dialogue_frame_finder.media_resolver import ResolvedMedia
from dialogue_frame_finder.onset import AudioClip
from dialogue_frame_finder.output import MatchStatus
from dialogue_frame_finder.pipeline import run_pipeline
from dialogue_frame_finder.semantic import SemanticMatchResult
from dialogue_frame_finder.transcript import Transcript

pytestmark = pytest.mark.unit


class _FakeMediaResolver:
    def __init__(self, local_path="fake.mp4", duration=10.0, provider="generic"):
        self._result = ResolvedMedia(local_path=local_path, duration=duration, provider=provider)
        self.calls = []

    def resolve(self, url):
        self.calls.append(url)
        return self._result


class _FakeASR:
    def __init__(self, transcript):
        self._transcript = transcript
        self.calls = []

    def transcribe(self, audio_path):
        self.calls.append(audio_path)
        return self._transcript


def _fake_audio_clip_fn(path):
    return AudioClip(samples=np.zeros(1600, dtype=np.float64), sample_rate=16000)


def _make_fake_locate_frame_fn(calls):
    def fn(path, target_seconds):
        calls.append((path, target_seconds))
        return FrameResult(pts_seconds=target_seconds, frame_number=99, image=np.zeros((2, 2, 3), dtype=np.uint8))
    return fn


def _make_fake_save_image_fn(calls):
    def fn(image, path):
        calls.append(path)
    return fn


def _make_fake_locate_frame_fn_with_offset_pts(calls, offset):
    # simulates the realistic case: the frame's own PTS is not exactly the
    # onset passed in ("first frame with PTS >= onset" generally lands
    # slightly after it)
    def fn(path, target_seconds):
        calls.append((path, target_seconds))
        pts = target_seconds + offset
        return FrameResult(pts_seconds=pts, frame_number=99, image=np.zeros((2, 2, 3), dtype=np.uint8))
    return fn


def _transcript_with_target():
    words = ["well", "i", "think", "my", "mind", "rebels", "at", "stagnation"]
    return Transcript.from_word_tuples([(w, i * 0.4, i * 0.4 + 0.35) for i, w in enumerate(words)])


class _FakeCaptionSource:
    def __init__(self, coarse_transcript):
        self._coarse_transcript = coarse_transcript
        self.calls = []

    def fetch_coarse_transcript(self, url):
        self.calls.append(url)
        return self._coarse_transcript


class _FakeCaptionSourceNoCaptions:
    def fetch_coarse_transcript(self, url):
        return None


def test_run_pipeline_success_case_wires_everything(tmp_path):
    media_resolver = _FakeMediaResolver(local_path="movie.mp4", provider="okru")
    asr = _FakeASR(_transcript_with_target())
    locate_calls, save_calls = [], []

    result = run_pipeline(
        "https://ok.ru/video/248244667877",
        "My mind rebels at stagnation",
        output_dir=tmp_path,
        media_resolver=media_resolver,
        asr_adapter=asr,
        extract_audio_clip_fn=_fake_audio_clip_fn,
        locate_frame_fn=_make_fake_locate_frame_fn(locate_calls),
        save_frame_image_fn=_make_fake_save_image_fn(save_calls),
    )

    assert media_resolver.calls == ["https://ok.ru/video/248244667877"]
    assert asr.calls == ["movie.mp4"]
    assert result.search_result.chosen is not None
    assert result.output.status in (
        MatchStatus.HIGH_CONFIDENCE, MatchStatus.MEDIUM_CONFIDENCE, MatchStatus.LOW_CONFIDENCE,
    )
    assert result.output.frame == 99
    assert result.output.text == "my mind rebels at stagnation"
    assert result.output.timestamp is not None
    assert result.output.image_path is not None
    assert len(locate_calls) == 1
    assert locate_calls[0][0] == "movie.mp4"
    assert save_calls == [result.output.image_path]
    assert result.transcript_source == "full_video_asr"


def test_run_pipeline_uses_caption_assisted_transcript_when_available(tmp_path):
    media_resolver = _FakeMediaResolver(local_path="movie.mp4", provider="youtube")
    asr = _FakeASR(_transcript_with_target())
    caption_source = _FakeCaptionSource(_transcript_with_target())
    locate_calls, save_calls = [], []

    result = run_pipeline(
        "https://www.youtube.com/watch?v=x",
        "My mind rebels at stagnation",
        output_dir=tmp_path,
        media_resolver=media_resolver,
        asr_adapter=asr,
        extract_audio_clip_fn=_fake_audio_clip_fn,
        locate_frame_fn=_make_fake_locate_frame_fn(locate_calls),
        save_frame_image_fn=_make_fake_save_image_fn(save_calls),
        caption_source=caption_source,
    )

    assert caption_source.calls == ["https://www.youtube.com/watch?v=x"]
    assert result.transcript_source == "captions_local_asr"
    assert result.output.diagnostics["transcript_source"] == "captions_local_asr"
    # The full-video ASR path (asr_adapter.transcribe("movie.mp4")) must be
    # skipped entirely when captions successfully provide a confirmed match --
    # only the local-window (temp-file) transcribe call should have happened.
    assert "movie.mp4" not in asr.calls
    assert result.search_result.chosen is not None


def test_run_pipeline_falls_back_to_full_video_asr_when_captions_unavailable(tmp_path):
    media_resolver = _FakeMediaResolver(local_path="movie.mp4", provider="youtube")
    asr = _FakeASR(_transcript_with_target())
    caption_source = _FakeCaptionSourceNoCaptions()
    locate_calls, save_calls = [], []

    result = run_pipeline(
        "https://www.youtube.com/watch?v=x",
        "My mind rebels at stagnation",
        output_dir=tmp_path,
        media_resolver=media_resolver,
        asr_adapter=asr,
        extract_audio_clip_fn=_fake_audio_clip_fn,
        locate_frame_fn=_make_fake_locate_frame_fn(locate_calls),
        save_frame_image_fn=_make_fake_save_image_fn(save_calls),
        caption_source=caption_source,
    )

    assert result.transcript_source == "full_video_asr"
    assert "movie.mp4" in asr.calls
    assert result.search_result.chosen is not None


def test_run_pipeline_default_has_captions_disabled(tmp_path):
    media_resolver = _FakeMediaResolver(local_path="movie.mp4", provider="youtube")
    asr = _FakeASR(_transcript_with_target())
    locate_calls, save_calls = [], []

    result = run_pipeline(
        "https://www.youtube.com/watch?v=x",
        "My mind rebels at stagnation",
        output_dir=tmp_path,
        media_resolver=media_resolver,
        asr_adapter=asr,
        extract_audio_clip_fn=_fake_audio_clip_fn,
        locate_frame_fn=_make_fake_locate_frame_fn(locate_calls),
        save_frame_image_fn=_make_fake_save_image_fn(save_calls),
    )

    assert result.transcript_source == "full_video_asr"
    assert "movie.mp4" in asr.calls


def test_run_pipeline_saves_image_under_images_subdirectory(tmp_path):
    media_resolver = _FakeMediaResolver(local_path="movie.mp4")
    asr = _FakeASR(_transcript_with_target())
    save_calls = []

    result = run_pipeline(
        "https://ok.ru/video/248244667877",
        "My mind rebels at stagnation",
        output_dir=tmp_path,
        media_resolver=media_resolver,
        asr_adapter=asr,
        extract_audio_clip_fn=_fake_audio_clip_fn,
        locate_frame_fn=_make_fake_locate_frame_fn([]),
        save_frame_image_fn=_make_fake_save_image_fn(save_calls),
    )

    image_path = Path(result.output.image_path)
    assert image_path.parent.name == "images"
    assert image_path.parent.parent == tmp_path
    assert image_path.name.startswith("frame_") and image_path.suffix == ".png"
    # relative to output_dir, the printed path reads as "images/frame_....png"
    assert image_path.relative_to(tmp_path) == Path("images") / image_path.name


def test_run_pipeline_image_path_is_project_relative_when_output_dir_is_relative(monkeypatch, tmp_path):
    # e.g. CLI default --output-dir "outputs" run from the repo root ->
    # printed Image path should read "outputs/images/frame_....png", not an
    # absolute filesystem path.
    monkeypatch.chdir(tmp_path)
    media_resolver = _FakeMediaResolver(local_path="movie.mp4")
    asr = _FakeASR(_transcript_with_target())

    result = run_pipeline(
        "https://ok.ru/video/248244667877",
        "My mind rebels at stagnation",
        output_dir="outputs",
        media_resolver=media_resolver,
        asr_adapter=asr,
        extract_audio_clip_fn=_fake_audio_clip_fn,
        locate_frame_fn=_make_fake_locate_frame_fn([]),
        save_frame_image_fn=_make_fake_save_image_fn([]),
    )

    image_path = Path(result.output.image_path)
    assert not image_path.is_absolute()
    assert image_path.parts[:2] == ("outputs", "images")


def test_run_pipeline_image_filename_matches_reported_timestamp_not_raw_onset(tmp_path):
    # OUT-05: the saved image's filename must agree with the Timestamp the
    # pipeline actually reports (the frame's own PTS), not the intermediate
    # onset estimate used only to locate that frame -- guards against the
    # displayed Timestamp and the image filename silently disagreeing.
    media_resolver = _FakeMediaResolver(local_path="movie.mp4")
    asr = _FakeASR(_transcript_with_target())
    locate_calls, save_calls = [], []

    result = run_pipeline(
        "https://example.com/video",
        "My mind rebels at stagnation",
        output_dir=tmp_path,
        media_resolver=media_resolver,
        asr_adapter=asr,
        extract_audio_clip_fn=_fake_audio_clip_fn,
        locate_frame_fn=_make_fake_locate_frame_fn_with_offset_pts(locate_calls, offset=0.025),
        save_frame_image_fn=_make_fake_save_image_fn(save_calls),
    )

    raw_onset = result.search_result.chosen.first_word_start  # onset refinement is a no-op on silent fake audio
    expected_frame_pts = raw_onset + 0.025
    assert f"{expected_frame_pts:.3f}" in result.output.image_path
    assert f"{raw_onset:.3f}" not in result.output.image_path
    assert save_calls[0] == result.output.image_path


def test_run_pipeline_no_match_case_skips_frame_extraction(tmp_path):
    media_resolver = _FakeMediaResolver()
    words = ["completely", "unrelated", "words", "here"]
    transcript = Transcript.from_word_tuples([(w, i * 0.4, i * 0.4 + 0.35) for i, w in enumerate(words)])
    asr = _FakeASR(transcript)
    locate_calls, save_calls = [], []

    result = run_pipeline(
        "https://example.com/video",
        "My mind rebels at stagnation",
        output_dir=tmp_path,
        media_resolver=media_resolver,
        asr_adapter=asr,
        extract_audio_clip_fn=_fake_audio_clip_fn,
        locate_frame_fn=_make_fake_locate_frame_fn(locate_calls),
        save_frame_image_fn=_make_fake_save_image_fn(save_calls),
    )

    assert result.output.status == MatchStatus.NO_CONFIDENT_MATCH
    assert result.output.timestamp is None
    assert result.output.frame is None
    assert result.output.image_path is None
    assert locate_calls == []  # never attempted frame extraction with no onset
    assert save_calls == []


class _FakeSemanticMatcher:
    def __init__(self, result):
        self._result = result
        self.calls = 0

    def match(self, target_text, candidates):
        self.calls += 1
        return self._result


def test_run_pipeline_uses_semantic_fallback_when_lexical_match_fails(tmp_path):
    media_resolver = _FakeMediaResolver(local_path="movie.mp4")
    # paraphrase, no lexical overlap worth matching on
    words = ["stagnation", "is", "something", "my", "thoughts", "resist", "deeply"]
    transcript = Transcript.from_word_tuples([(w, i * 0.4, i * 0.4 + 0.35) for i, w in enumerate(words)])
    asr = _FakeASR(transcript)
    locate_calls = []

    from dialogue_frame_finder.semantic import gather_semantic_candidates
    candidates = gather_semantic_candidates(["my", "mind", "rebels", "at", "stagnation"], transcript)
    matcher = _FakeSemanticMatcher(
        SemanticMatchResult(candidate_id=candidates[0].candidate_id, match_confidence=0.7, rationale="paraphrase")
    )

    result = run_pipeline(
        "https://example.com/video",
        "My mind rebels at stagnation",
        output_dir=tmp_path,
        media_resolver=media_resolver,
        asr_adapter=asr,
        extract_audio_clip_fn=_fake_audio_clip_fn,
        locate_frame_fn=_make_fake_locate_frame_fn(locate_calls),
        save_frame_image_fn=_make_fake_save_image_fn([]),
        semantic_matcher=matcher,
    )

    assert matcher.calls == 1
    assert result.search_result.tier_used == "semantic_fallback"
    assert result.output.status != MatchStatus.NO_CONFIDENT_MATCH
    assert result.output.frame == 99  # frame extraction did proceed, using the semantic pick's onset


def test_run_pipeline_semantic_matcher_not_consulted_on_confident_lexical_match(tmp_path):
    media_resolver = _FakeMediaResolver(local_path="movie.mp4")
    asr = _FakeASR(_transcript_with_target())
    matcher = _FakeSemanticMatcher(SemanticMatchResult(candidate_id="unused", match_confidence=0.99, rationale="n/a"))

    run_pipeline(
        "https://example.com/video",
        "My mind rebels at stagnation",
        output_dir=tmp_path,
        media_resolver=media_resolver,
        asr_adapter=asr,
        extract_audio_clip_fn=_fake_audio_clip_fn,
        locate_frame_fn=_make_fake_locate_frame_fn([]),
        save_frame_image_fn=_make_fake_save_image_fn([]),
        semantic_matcher=matcher,
    )

    assert matcher.calls == 0


def test_run_pipeline_creates_output_directory(tmp_path):
    out_dir = tmp_path / "nested" / "outputs"
    media_resolver = _FakeMediaResolver()
    asr = _FakeASR(_transcript_with_target())

    run_pipeline(
        "https://example.com/video",
        "My mind rebels at stagnation",
        output_dir=out_dir,
        media_resolver=media_resolver,
        asr_adapter=asr,
        extract_audio_clip_fn=_fake_audio_clip_fn,
        locate_frame_fn=_make_fake_locate_frame_fn([]),
        save_frame_image_fn=_make_fake_save_image_fn([]),
    )
    assert out_dir.is_dir()
