import pytest

from dialogue_frame_finder.media_resolver import MediaResolutionError, MediaResolver, detect_provider

pytestmark = pytest.mark.unit


def test_detect_provider_okru():
    assert detect_provider("https://ok.ru/video/248244667877") == "okru"


def test_detect_provider_youtube_full_domain():
    assert detect_provider("https://www.youtube.com/watch?v=abc123") == "youtube"


def test_detect_provider_youtube_short_domain():
    assert detect_provider("https://youtu.be/abc123") == "youtube"


def test_detect_provider_generic_for_unknown_host():
    assert detect_provider("https://example.com/video/1") == "generic"


def _make_fake_ydl(opts_log, info=None, fail_times=0):
    state = {"fail_remaining": fail_times}

    class FakeYDL:
        def __init__(self, opts):
            opts_log.append(opts)

        def __enter__(self):
            return self

        def __exit__(self, *exc_info):
            return False

        def extract_info(self, url, download=True):
            if state["fail_remaining"] > 0:
                state["fail_remaining"] -= 1
                raise RuntimeError("simulated network failure")
            return info or {"id": "abc123", "ext": "mp4", "duration": 42.0}

        def prepare_filename(self, extracted_info):
            return f"/fake/{extracted_info['id']}.{extracted_info['ext']}"

    return FakeYDL


def test_media_resolver_resolve_success(tmp_path):
    opts_log = []
    resolver = MediaResolver(tmp_path, ydl_class=_make_fake_ydl(opts_log))
    result = resolver.resolve("https://ok.ru/video/248244667877")
    assert result.local_path == "/fake/abc123.mp4"
    assert result.duration == pytest.approx(42.0)
    assert result.provider == "okru"


def test_media_resolver_applies_okru_specific_options(tmp_path):
    opts_log = []
    resolver = MediaResolver(tmp_path, ydl_class=_make_fake_ydl(opts_log))
    resolver.resolve("https://ok.ru/video/248244667877")
    opts = opts_log[0]
    assert opts["downloader_args"]["ffmpeg_i"] == ["-tls_verify", "0"]
    assert opts["nocheckcertificate"] is True
    assert opts["retries"] == 5


def test_media_resolver_applies_youtube_specific_options(tmp_path):
    opts_log = []
    resolver = MediaResolver(tmp_path, ydl_class=_make_fake_ydl(opts_log))
    resolver.resolve("https://www.youtube.com/watch?v=abc123")
    opts = opts_log[0]
    assert opts["js_runtimes"] == {"node": {}}
    assert opts["extractor_args"]["youtube"]["player_client"] == ["android", "web_safari"]


def test_media_resolver_generic_provider_gets_no_extra_options(tmp_path):
    opts_log = []
    resolver = MediaResolver(tmp_path, ydl_class=_make_fake_ydl(opts_log))
    resolver.resolve("https://example.com/video/1")
    opts = opts_log[0]
    assert "downloader_args" not in opts
    assert "js_runtimes" not in opts


def test_media_resolver_retries_transient_failure_then_succeeds(tmp_path):
    opts_log = []
    resolver = MediaResolver(
        tmp_path, ydl_class=_make_fake_ydl(opts_log, fail_times=2),
        max_attempts=3, retry_backoff_seconds=0,
    )
    result = resolver.resolve("https://ok.ru/video/248244667877")
    assert result.local_path == "/fake/abc123.mp4"
    assert len(opts_log) == 3  # two failed attempts + one success


def test_media_resolver_format_selector_is_passed_through(tmp_path):
    opts_log = []
    resolver = MediaResolver(tmp_path, ydl_class=_make_fake_ydl(opts_log), format_selector="worst")
    resolver.resolve("https://ok.ru/video/248244667877")
    assert opts_log[0]["format"] == "worst"


def test_media_resolver_no_format_key_when_selector_not_given(tmp_path):
    opts_log = []
    resolver = MediaResolver(tmp_path, ydl_class=_make_fake_ydl(opts_log))
    resolver.resolve("https://ok.ru/video/248244667877")
    assert "format" not in opts_log[0]


def test_media_resolver_raises_after_exhausting_retries(tmp_path):
    opts_log = []
    resolver = MediaResolver(
        tmp_path, ydl_class=_make_fake_ydl(opts_log, fail_times=10),
        max_attempts=3, retry_backoff_seconds=0,
    )
    with pytest.raises(MediaResolutionError):
        resolver.resolve("https://ok.ru/video/248244667877")
    assert len(opts_log) == 3
