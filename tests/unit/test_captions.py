import io
import json

import numpy as np
import pytest

from dialogue_frame_finder.captions import (
    CaptionSource,
    compute_local_window,
    try_caption_assisted_transcript,
)
from dialogue_frame_finder.config import SearchConfig
from dialogue_frame_finder.onset import AudioClip
from dialogue_frame_finder.search import search_dialogue
from dialogue_frame_finder.transcript import Transcript

pytestmark = pytest.mark.unit


def _json3(events):
    return json.dumps({"events": events}).encode("utf-8")


def _event(start_s, dur_s, text):
    return {"tStartMs": int(start_s * 1000), "dDurationMs": int(dur_s * 1000), "segs": [{"utf8": text}]}


class _FakeResponse:
    def __init__(self, data: bytes):
        self._data = data

    def read(self):
        return self._data


def _make_fake_ydl(info):
    class FakeYDL:
        def __init__(self, opts):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc_info):
            return False

        def extract_info(self, url, download=False):
            return info

    return FakeYDL


def test_fetch_coarse_transcript_prefers_manual_over_auto():
    info = {
        "subtitles": {"en": [{"ext": "json3", "url": "manual-url"}]},
        "automatic_captions": {"en": [{"ext": "json3", "url": "auto-url"}]},
    }
    seen_urls = []

    def opener(url):
        seen_urls.append(url)
        return _FakeResponse(_json3([_event(0.0, 1.0, "hello world")]))

    source = CaptionSource(ydl_class=_make_fake_ydl(info), url_opener=opener)
    transcript = source.fetch_coarse_transcript("https://youtube.com/watch?v=x")
    assert seen_urls == ["manual-url"]
    assert [w.word for w in transcript.words] == ["hello", "world"]


def test_fetch_coarse_transcript_falls_back_to_auto_when_no_manual():
    info = {"automatic_captions": {"en": [{"ext": "json3", "url": "auto-url"}]}}
    source = CaptionSource(
        ydl_class=_make_fake_ydl(info),
        url_opener=lambda url: _FakeResponse(_json3([_event(5.0, 2.0, "go darcy")])),
    )
    transcript = source.fetch_coarse_transcript("https://youtube.com/watch?v=x")
    assert [w.word for w in transcript.words] == ["go", "darcy"]
    assert transcript.words[0].start == pytest.approx(5.0)
    assert transcript.words[0].end == pytest.approx(7.0)


def test_fetch_coarse_transcript_returns_none_when_no_tracks():
    info = {}
    source = CaptionSource(ydl_class=_make_fake_ydl(info), url_opener=lambda url: _FakeResponse(b""))
    assert source.fetch_coarse_transcript("https://youtube.com/watch?v=x") is None


def test_fetch_coarse_transcript_returns_none_when_no_json3_format():
    info = {"automatic_captions": {"en": [{"ext": "vtt", "url": "vtt-url"}]}}
    source = CaptionSource(ydl_class=_make_fake_ydl(info), url_opener=lambda url: _FakeResponse(b""))
    assert source.fetch_coarse_transcript("https://youtube.com/watch?v=x") is None


def test_fetch_coarse_transcript_returns_none_on_extract_info_failure():
    class FailingYDL:
        def __init__(self, opts):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc_info):
            return False

        def extract_info(self, url, download=False):
            raise RuntimeError("network error")

    source = CaptionSource(ydl_class=FailingYDL, url_opener=lambda url: _FakeResponse(b""))
    assert source.fetch_coarse_transcript("https://youtube.com/watch?v=x") is None


def test_fetch_coarse_transcript_returns_none_on_malformed_json():
    info = {"automatic_captions": {"en": [{"ext": "json3", "url": "auto-url"}]}}
    source = CaptionSource(ydl_class=_make_fake_ydl(info), url_opener=lambda url: _FakeResponse(b"not json"))
    assert source.fetch_coarse_transcript("https://youtube.com/watch?v=x") is None


def test_fetch_coarse_transcript_matches_language_prefix():
    info = {"automatic_captions": {"en-US": [{"ext": "json3", "url": "auto-url"}]}}
    source = CaptionSource(
        ydl_class=_make_fake_ydl(info),
        url_opener=lambda url: _FakeResponse(_json3([_event(0.0, 1.0, "hi")])),
    )
    transcript = source.fetch_coarse_transcript("https://youtube.com/watch?v=x")
    assert transcript is not None
    assert [w.word for w in transcript.words] == ["hi"]


def _words_transcript(words):
    return Transcript.from_word_tuples(words)


def test_compute_local_window_pads_around_block_span():
    transcript = _words_transcript([("go", 100.0, 100.3), ("darcy", 100.3, 100.6)])
    result = search_dialogue("go darcy", transcript, SearchConfig(valid_score_threshold=0.0))
    config = SearchConfig(caption_window_pre_pad=2.0, caption_window_post_pad=3.0, caption_window_max_seconds=60.0)
    start, end = compute_local_window(result.chosen, transcript, config)
    assert start == pytest.approx(98.0)
    assert end == pytest.approx(103.6)


def test_compute_local_window_clips_negative_start_to_zero():
    transcript = _words_transcript([("go", 0.5, 0.8), ("darcy", 0.8, 1.0)])
    result = search_dialogue("go darcy", transcript, SearchConfig(valid_score_threshold=0.0))
    config = SearchConfig(caption_window_pre_pad=2.0, caption_window_post_pad=3.0, caption_window_max_seconds=60.0)
    start, end = compute_local_window(result.chosen, transcript, config)
    assert start == 0.0


def test_compute_local_window_respects_max_seconds_cap():
    transcript = _words_transcript([("go", 100.0, 200.0), ("darcy", 200.0, 300.0)])
    result = search_dialogue("go darcy", transcript, SearchConfig(valid_score_threshold=0.0))
    config = SearchConfig(caption_window_pre_pad=2.0, caption_window_post_pad=3.0, caption_window_max_seconds=60.0)
    start, end = compute_local_window(result.chosen, transcript, config)
    assert end - start == pytest.approx(60.0)


class _FakeASRAdapter:
    """Returns a fixed Transcript regardless of the (temp file) path given,
    with timestamps relative to 0 -- exercises transcribe_local_window's
    timestamp-shifting without needing real audio decoding."""

    def __init__(self, words):
        self._words = words

    def transcribe(self, audio_path):
        return Transcript.from_word_tuples(self._words)


def _silence_clip(duration_s, sample_rate=16000):
    return AudioClip(samples=np.zeros(int(duration_s * sample_rate)), sample_rate=sample_rate)


def test_try_caption_assisted_transcript_full_success_path():
    coarse_info = {"automatic_captions": {"en": [{"ext": "json3", "url": "auto-url"}]}}
    caption_source = CaptionSource(
        ydl_class=_make_fake_ydl(coarse_info),
        url_opener=lambda url: _FakeResponse(_json3([_event(100.0, 2.0, "let's go darcy")])),
    )
    # Local ASR (fake) reports the phrase at t=0.5 relative to the local window
    asr_adapter = _FakeASRAdapter([("let's", 0.5, 0.7), ("go", 0.7, 0.9), ("darcy", 0.9, 1.1)])
    audio_clip = _silence_clip(200.0)

    result = try_caption_assisted_transcript(
        "https://youtube.com/watch?v=x", "let's go darcy", audio_clip, caption_source, asr_adapter,
        SearchConfig(),
    )
    assert result.used is True
    assert result.reason == "captions_localized_local_asr_confirmed"
    # window start = 100 - pre_pad(2.0) = 98.0; local ASR word at rel 0.5 -> abs 98.5
    assert result.transcript.words[0].start == pytest.approx(98.5)


def test_try_caption_assisted_transcript_no_captions_falls_back():
    caption_source = CaptionSource(ydl_class=_make_fake_ydl({}), url_opener=lambda url: _FakeResponse(b""))
    asr_adapter = _FakeASRAdapter([("irrelevant", 0.0, 0.2)])
    result = try_caption_assisted_transcript(
        "https://youtube.com/watch?v=x", "let's go darcy", _silence_clip(10.0), caption_source, asr_adapter,
    )
    assert result.used is False
    assert result.reason == "no_usable_captions"


def test_try_caption_assisted_transcript_no_coarse_match_falls_back():
    info = {"automatic_captions": {"en": [{"ext": "json3", "url": "auto-url"}]}}
    caption_source = CaptionSource(
        ydl_class=_make_fake_ydl(info),
        url_opener=lambda url: _FakeResponse(_json3([_event(0.0, 1.0, "completely unrelated text")])),
    )
    asr_adapter = _FakeASRAdapter([("irrelevant", 0.0, 0.2)])
    result = try_caption_assisted_transcript(
        "https://youtube.com/watch?v=x", "let's go darcy", _silence_clip(10.0), caption_source, asr_adapter,
    )
    assert result.used is False
    assert result.reason == "captions_no_confident_coarse_match"


class _SequenceASRAdapter:
    """Returns a different fixed Transcript per call, in order -- lets a
    test simulate the coarse pass finding several valid candidates before
    the true one, and confirm the code recovers by trying the next
    candidate rather than giving up after the first."""

    def __init__(self, transcripts_by_call):
        self._transcripts = transcripts_by_call
        self._i = 0

    def transcribe(self, audio_path):
        t = self._transcripts[min(self._i, len(self._transcripts) - 1)]
        self._i += 1
        return t


def test_try_caption_assisted_transcript_recovers_past_earlier_false_positive_coarse_candidate():
    # Two coarse "Darcy" mentions -- an earlier spurious one (coverage
    # capped, only "darcy" itself matches) and the true, later one.
    info = {
        "automatic_captions": {
            "en": [{"ext": "json3", "url": "auto-url"}],
        }
    }
    filler = " ".join(f"filler{i}" for i in range(20))
    caption_source = CaptionSource(
        ydl_class=_make_fake_ydl(info),
        url_opener=lambda url: _FakeResponse(_json3([
            _event(10.0, 1.0, "let's darcy go"),  # scrambled: scores 0.75, above default threshold
            _event(15.0, 5.0, filler),
            _event(200.0, 2.0, "let's go darcy"),
        ])),
    )
    # First local ASR attempt (around the spurious candidate) doesn't
    # confirm the target; second attempt (around the true candidate) does.
    asr_adapter = _SequenceASRAdapter([
        Transcript.from_word_tuples([("unrelated", 0.0, 0.2)]),
        Transcript.from_word_tuples([("let's", 0.0, 0.2), ("go", 0.2, 0.4), ("darcy", 0.4, 0.6)]),
    ])
    result = try_caption_assisted_transcript(
        "https://youtube.com/watch?v=x", "let's go darcy", _silence_clip(300.0), caption_source, asr_adapter,
    )
    assert result.used is True
    assert result.reason == "captions_localized_local_asr_confirmed"


def test_try_caption_assisted_transcript_local_asr_not_confirming_falls_back():
    info = {"automatic_captions": {"en": [{"ext": "json3", "url": "auto-url"}]}}
    caption_source = CaptionSource(
        ydl_class=_make_fake_ydl(info),
        url_opener=lambda url: _FakeResponse(_json3([_event(5.0, 2.0, "let's go darcy")])),
    )
    # Local ASR (fake) returns something that doesn't actually match the target
    asr_adapter = _FakeASRAdapter([("completely", 0.0, 0.2), ("different", 0.2, 0.4)])
    result = try_caption_assisted_transcript(
        "https://youtube.com/watch?v=x", "let's go darcy", _silence_clip(20.0), caption_source, asr_adapter,
    )
    assert result.used is False
    assert result.reason == "local_asr_did_not_confirm_match"
