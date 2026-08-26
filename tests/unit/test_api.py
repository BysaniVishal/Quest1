import time

import pytest

fastapi_testclient = pytest.importorskip("fastapi.testclient")
TestClient = fastapi_testclient.TestClient

from dialogue_frame_finder import api as api_mod
from dialogue_frame_finder.media_resolver import MediaResolutionError
from dialogue_frame_finder.output import MatchStatus, OutputRecord
from dialogue_frame_finder.pipeline import PipelineResult

pytestmark = pytest.mark.unit


def _fake_output_record(status=MatchStatus.HIGH_CONFIDENCE, image_path="fake_frame.png"):
    return OutputRecord(
        status=status,
        timestamp="00:00:17.640",
        frame=529,
        text="one small step for man",
        image_path=image_path,
        confidence_score=0.879,
        diagnostics={},
    )


def _fake_run_pipeline_success(*args, **kwargs):
    on_stage = kwargs.get("on_stage")
    if on_stage:
        on_stage("Downloading video...")
        on_stage("Transcribing audio (full video)...")
        on_stage("Verifying dialogue...")
    return PipelineResult(
        output=_fake_output_record(),
        resolved_media=None,
        transcript=None,
        search_result=None,
        transcript_source="full_video_asr",
    )


def _fake_run_pipeline_no_match(*args, **kwargs):
    return PipelineResult(
        output=_fake_output_record(status=MatchStatus.NO_CONFIDENT_MATCH, image_path=None),
        resolved_media=None,
        transcript=None,
        search_result=None,
        transcript_source="full_video_asr",
    )


def _fake_run_pipeline_download_failure(*args, **kwargs):
    raise MediaResolutionError("simulated failure")


def _fake_run_pipeline_unexpected_error(*args, **kwargs):
    raise RuntimeError("something internal broke")


@pytest.fixture()
def client():
    api_mod._JOBS.clear()
    return TestClient(api_mod.app)


def _wait_for_completion(client, job_id, timeout=5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        resp = client.get(f"/api/search/{job_id}")
        data = resp.json()
        if data["status"] != "running":
            return data
        time.sleep(0.02)
    raise TimeoutError("job did not complete in time")


def test_start_search_rejects_invalid_url(client):
    resp = client.post("/api/search", json={"url": "not-a-url", "dialogue": "hello"})
    assert resp.status_code == 400
    assert "valid" in resp.json()["detail"].lower()


def test_start_search_rejects_empty_dialogue(client):
    resp = client.post("/api/search", json={"url": "https://www.youtube.com/watch?v=x", "dialogue": "   "})
    assert resp.status_code == 400


def test_get_search_status_unknown_job_returns_404(client):
    resp = client.get("/api/search/does-not-exist")
    assert resp.status_code == 404


def test_full_success_flow_reports_result(client, monkeypatch):
    monkeypatch.setattr(api_mod, "run_pipeline", _fake_run_pipeline_success)
    resp = client.post("/api/search", json={"url": "https://www.youtube.com/watch?v=x", "dialogue": "one small step"})
    assert resp.status_code == 200
    job_id = resp.json()["job_id"]

    data = _wait_for_completion(client, job_id)
    assert data["status"] == "done"
    assert data["result"]["status"] == "HIGH_CONFIDENCE"
    assert data["result"]["timestamp"] == "00:00:17.640"
    assert data["result"]["image_url"] == f"/api/search/{job_id}/image"


def test_no_confident_match_is_a_result_not_an_error(client, monkeypatch):
    monkeypatch.setattr(api_mod, "run_pipeline", _fake_run_pipeline_no_match)
    resp = client.post("/api/search", json={"url": "https://www.youtube.com/watch?v=x", "dialogue": "nothing here"})
    job_id = resp.json()["job_id"]

    data = _wait_for_completion(client, job_id)
    assert data["status"] == "done"
    assert data["result"]["status"] == "NO_CONFIDENT_MATCH"
    assert data["result"]["image_url"] is None
    assert data["error"] is None


def test_download_failure_becomes_human_readable_error(client, monkeypatch):
    monkeypatch.setattr(api_mod, "run_pipeline", _fake_run_pipeline_download_failure)
    resp = client.post("/api/search", json={"url": "https://www.youtube.com/watch?v=x", "dialogue": "hello"})
    job_id = resp.json()["job_id"]

    data = _wait_for_completion(client, job_id)
    assert data["status"] == "error"
    assert "download" in data["error"]["message"].lower()
    assert "Traceback" not in data["error"]["message"]


def test_unexpected_exception_becomes_generic_human_readable_error(client, monkeypatch):
    monkeypatch.setattr(api_mod, "run_pipeline", _fake_run_pipeline_unexpected_error)
    resp = client.post("/api/search", json={"url": "https://www.youtube.com/watch?v=x", "dialogue": "hello"})
    job_id = resp.json()["job_id"]

    data = _wait_for_completion(client, job_id)
    assert data["status"] == "error"
    assert "something internal broke" not in data["error"]["message"]
    assert "went wrong" in data["error"]["message"].lower()


def test_image_endpoint_404_when_no_image(client, monkeypatch):
    monkeypatch.setattr(api_mod, "run_pipeline", _fake_run_pipeline_no_match)
    resp = client.post("/api/search", json={"url": "https://www.youtube.com/watch?v=x", "dialogue": "hello"})
    job_id = resp.json()["job_id"]
    _wait_for_completion(client, job_id)

    img_resp = client.get(f"/api/search/{job_id}/image")
    assert img_resp.status_code == 404


def test_progress_stage_is_visible_while_running(client, monkeypatch):
    import threading

    release = threading.Event()

    def slow_pipeline(*args, **kwargs):
        on_stage = kwargs.get("on_stage")
        if on_stage:
            on_stage("Downloading video...")
        release.wait(timeout=2.0)
        return _fake_run_pipeline_success(*args, **kwargs)

    monkeypatch.setattr(api_mod, "run_pipeline", slow_pipeline)
    resp = client.post("/api/search", json={"url": "https://www.youtube.com/watch?v=x", "dialogue": "hello"})
    job_id = resp.json()["job_id"]

    deadline = time.time() + 2.0
    seen_stage = None
    while time.time() < deadline:
        data = client.get(f"/api/search/{job_id}").json()
        if data["stage"]:
            seen_stage = data["stage"]
            break
        time.sleep(0.01)

    release.set()
    assert seen_stage == "Downloading video..."
