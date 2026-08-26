"""Thin HTTP API for the web UI, wrapping the existing pipeline exactly as
the CLI does -- no matching/verification/ASR logic lives here, only request
handling, background job bookkeeping (for long-running requests), and
translating results/errors into JSON. `run_pipeline` (pipeline.py) remains
the single source of truth for the actual algorithm.

Each search runs in a background thread with an in-memory job-state dict --
deliberately no task queue/database, since this is a single-user local tool
and that would be unjustified complexity for what's needed here.

`fastapi`/`uvicorn` are optional dependencies (requirements-web.txt, not
requirements.txt) -- importing this module is not required to run the
offline test suite or the CLI, matching how faster-whisper/anthropic are
already handled as lazy/optional elsewhere in this package.
"""

import re
import threading
import uuid
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .asr import FasterWhisperASR
from .captions import CaptionSource
from .media_resolver import MediaResolutionError, MediaResolver
from .pipeline import run_pipeline

app = FastAPI(title="Dialogue-to-Exact-Video-Frame Finder")

OUTPUT_DIR = Path("outputs") / "web_jobs"

# Accepts YouTube/OK.ru URLs specifically -- MediaResolver's own
# detect_provider() is intentionally more permissive (a "generic" provider
# path exists for anything yt-dlp might resolve), but the web form's own
# validation message should match what this project actually targets.
URL_RE = re.compile(r"^https?://(www\.)?(youtube\.com|youtu\.be|ok\.ru|odnoklassniki\.ru)/", re.I)

_JOBS_LOCK = threading.Lock()
_JOBS: dict = {}


class SearchRequest(BaseModel):
    url: str
    dialogue: str
    use_captions: bool = False
    asr_model: Optional[str] = None


def _human_error(exc: Exception) -> str:
    """Never let a raw traceback reach the client -- map what we can to a
    specific, actionable message; anything unrecognized gets a generic one
    (the real exception is still in _JOBS server-side for debugging)."""
    if isinstance(exc, MediaResolutionError):
        return "Could not download this video. Check the URL, or the source may be temporarily unavailable."
    return "Something went wrong while processing this video. Please try again."


def _run_job(job_id: str, req: SearchRequest) -> None:
    def on_stage(message: str) -> None:
        with _JOBS_LOCK:
            _JOBS[job_id]["stage"] = message

    job_dir = OUTPUT_DIR / job_id
    try:
        media_resolver = MediaResolver(job_dir, format_selector="worst")
        asr_adapter = FasterWhisperASR(model_size=req.asr_model) if req.asr_model else None
        caption_source = CaptionSource() if req.use_captions else None

        result = run_pipeline(
            req.url, req.dialogue, job_dir,
            media_resolver=media_resolver,
            asr_adapter=asr_adapter,
            caption_source=caption_source,
            on_stage=on_stage,
        )
        record = result.output
        with _JOBS_LOCK:
            _JOBS[job_id].update(
                status="done",
                stage=None,
                image_path=record.image_path,
                result={
                    "status": record.status.value,
                    "timestamp": record.timestamp,
                    "frame": record.frame,
                    "text": record.text,
                    "confidence": record.confidence_score,
                    "transcript_source": result.transcript_source,
                    "image_url": f"/api/search/{job_id}/image" if record.image_path else None,
                },
            )
    except Exception as exc:  # noqa: BLE001 -- deliberately broad: any pipeline
        # failure must become a clean job-error state, never crash the thread
        # or leak a traceback to the client.
        with _JOBS_LOCK:
            _JOBS[job_id].update(status="error", stage=None, error={"message": _human_error(exc)})


@app.post("/api/search")
def start_search(req: SearchRequest):
    url = req.url.strip()
    dialogue = req.dialogue.strip()
    if not URL_RE.match(url):
        raise HTTPException(400, "Please enter a valid YouTube or OK.ru video URL.")
    if not dialogue:
        raise HTTPException(400, "Please enter the dialogue you're looking for.")

    job_id = uuid.uuid4().hex
    with _JOBS_LOCK:
        _JOBS[job_id] = {"status": "running", "stage": "Starting...", "result": None, "error": None, "image_path": None}
    threading.Thread(target=_run_job, args=(job_id, req), daemon=True).start()
    return {"job_id": job_id}


@app.get("/api/search/{job_id}")
def get_search_status(job_id: str):
    with _JOBS_LOCK:
        job = _JOBS.get(job_id)
    if job is None:
        raise HTTPException(404, "Unknown job id.")
    return {
        "job_id": job_id,
        "status": job["status"],
        "stage": job["stage"],
        "result": job["result"],
        "error": job["error"],
    }


@app.get("/api/search/{job_id}/image")
def get_search_image(job_id: str):
    with _JOBS_LOCK:
        job = _JOBS.get(job_id)
    if job is None or not job.get("image_path"):
        raise HTTPException(404, "No image available for this job.")
    return FileResponse(job["image_path"])


# Static frontend (web/index.html, style.css, app.js), mounted last so it
# doesn't shadow the /api routes above. Resolved relative to this file (not
# the process's current working directory) so `uvicorn` works regardless of
# where it's launched from.
_WEB_DIR = Path(__file__).resolve().parents[2] / "web"
app.mount("/", StaticFiles(directory=str(_WEB_DIR), html=True), name="web")
