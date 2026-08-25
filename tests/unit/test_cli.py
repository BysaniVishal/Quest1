import numpy as np
import pytest

from dialogue_frame_finder import cli
from dialogue_frame_finder.output import MatchStatus, OutputRecord
from dialogue_frame_finder.pipeline import PipelineResult
from dialogue_frame_finder.media_resolver import ResolvedMedia
from dialogue_frame_finder.transcript import Transcript
from dialogue_frame_finder.search import SearchResult

pytestmark = pytest.mark.unit


def _fake_pipeline_result(record: OutputRecord) -> PipelineResult:
    return PipelineResult(
        output=record,
        resolved_media=ResolvedMedia(local_path="fake.mp4", duration=10.0, provider="generic"),
        transcript=Transcript(words=[]),
        search_result=SearchResult(chosen=None, other_valid=[], tier_used="none", anchors_used=[], windows_verified=0),
    )


def test_cli_prints_all_four_required_fields_with_frame_number(monkeypatch, capsys):
    record = OutputRecord(
        status=MatchStatus.HIGH_CONFIDENCE,
        timestamp="00:05:25.305",
        frame=7812,
        text="My mind rebels at stagnation",
        image_path="outputs/frame_325.305.png",
        confidence_score=0.88,
        diagnostics={},
    )
    monkeypatch.setattr(cli, "run_pipeline", lambda *a, **k: _fake_pipeline_result(record))

    exit_code = cli.main(["https://ok.ru/video/248244667877", "My mind rebels at stagnation"])
    out = capsys.readouterr().out

    assert "Timestamp : 00:05:25.305" in out
    assert "Frame     : 7812" in out
    assert 'Text      : "My mind rebels at stagnation"' in out
    assert "Image     : outputs/frame_325.305.png" in out
    assert exit_code == 0


def test_cli_documents_unavailable_frame_number_explicitly(monkeypatch, capsys):
    record = OutputRecord(
        status=MatchStatus.MEDIUM_CONFIDENCE,
        timestamp="00:00:01.500",
        frame=None,
        text="some text",
        image_path="outputs/frame_1.500.png",
        confidence_score=0.7,
        diagnostics={},
    )
    monkeypatch.setattr(cli, "run_pipeline", lambda *a, **k: _fake_pipeline_result(record))

    cli.main(["https://example.com/video", "some text"])
    out = capsys.readouterr().out

    frame_line = next(line for line in out.splitlines() if line.startswith("Frame"))
    assert "None" not in frame_line  # must not print the bare Python None
    assert "N/A" in frame_line
    assert "not available" in frame_line


def test_cli_default_run_passes_no_custom_media_resolver(monkeypatch):
    record = OutputRecord(
        status=MatchStatus.NO_CONFIDENT_MATCH,
        timestamp=None, frame=None, text=None, image_path=None,
        confidence_score=None, diagnostics={},
    )
    calls = []

    def fake_run_pipeline(*args, **kwargs):
        calls.append(kwargs)
        return _fake_pipeline_result(record)

    monkeypatch.setattr(cli, "run_pipeline", fake_run_pipeline)
    cli.main(["https://example.com/video", "some text"])

    assert calls[0]["media_resolver"] is None


def test_cli_format_flag_builds_custom_media_resolver(monkeypatch):
    record = OutputRecord(
        status=MatchStatus.NO_CONFIDENT_MATCH,
        timestamp=None, frame=None, text=None, image_path=None,
        confidence_score=None, diagnostics={},
    )
    calls = []

    def fake_run_pipeline(*args, **kwargs):
        calls.append(kwargs)
        return _fake_pipeline_result(record)

    monkeypatch.setattr(cli, "run_pipeline", fake_run_pipeline)
    cli.main(["https://ok.ru/video/1", "some text", "--format", "worst", "--retries", "10"])

    resolver = calls[0]["media_resolver"]
    assert resolver is not None
    assert resolver.format_selector == "worst"
    assert resolver.max_attempts == 10


def test_cli_returns_nonzero_exit_code_on_no_confident_match(monkeypatch):
    record = OutputRecord(
        status=MatchStatus.NO_CONFIDENT_MATCH,
        timestamp=None, frame=None, text=None, image_path=None,
        confidence_score=None, diagnostics={},
    )
    monkeypatch.setattr(cli, "run_pipeline", lambda *a, **k: _fake_pipeline_result(record))

    exit_code = cli.main(["https://example.com/video", "unfindable phrase"])
    assert exit_code == 1
