# Dialogue-to-Exact-Video-Frame Finder

Given a publicly accessible video URL and a target spoken dialogue, locates the exact/first
video frame where that dialogue begins, and returns its timestamp, frame number (where
applicable), the identified dialogue text, and the corresponding frame as an image.

The dialogue is located by its **spoken audio**, not by on-screen text/subtitles — see
[APPROACH.md](APPROACH.md) for the full design and algorithm.

Two ways to run it: a **CLI** (`python -m dialogue_frame_finder ...`) and a **browser-based
Web UI** — both call the exact same underlying pipeline, no logic duplicated between them.

## Setup

Requires Python 3.10+.

```bash
pip install -r requirements.txt
```

`requirements.txt` covers everything needed to run the pipeline, including `yt-dlp`
(media resolution), `av`/`Pillow` (audio/video decode and image saving), `rapidfuzz`
(fuzzy text matching), and `faster-whisper` (speech-to-text). The optional LLM semantic
fallback additionally needs `anthropic` (already listed) and an API key — see
[Optional: semantic fallback](#optional-semantic-fallback) below; nothing else in the
pipeline requires it.

No `ffmpeg` binary on `PATH` is required — audio/video decoding goes through PyAV
directly.

**Windows note:** if `faster-whisper`'s model download fails with a `WinError 1314`
(symlink privilege), the pipeline already sets `HF_HUB_DISABLE_SYMLINKS=1` automatically
before loading the model — no manual step needed.

**Before running the CLI**, the package needs `src/` on `PYTHONPATH` (it isn't pip-installed
as a package — `pytest` already handles this itself via `pyproject.toml`, but a direct
`python -m` invocation needs it set explicitly, once per terminal session):

```powershell
# PowerShell (Windows)
$env:PYTHONPATH = "src"
```

```bash
# bash/zsh (macOS/Linux)
export PYTHONPATH=src
```

## Usage

```bash
python -m dialogue_frame_finder <video_url> "<target dialogue>" [--output-dir outputs] [--format worst]
```

Example, using the supplied Quest1 reference case:

```bash
python -m dialogue_frame_finder \
  "https://ok.ru/video/248244667877" \
  "My mind rebels at stagnation" \
  --format worst
```

`--format worst` is recommended: faster, smaller download, no effect on correctness
(resolution doesn't matter to the pipeline — see `--help` for all flags). Omit it for a
higher-quality source frame image.

Example output (real run, default config — `base.en` ASR, `valid_score_threshold=0.70`):

```
Status    : HIGH_CONFIDENCE
Timestamp : 00:05:25.013
Frame     : 7792
Text      : "My mind rebels its stagnation."
Image     : outputs\images\frame_325.013.png
Confidence: 0.832
```

`Text` is what was actually transcribed (here, a real ASR error — "its" instead of "at" —
tolerated correctly), not the target phrase echoed back; see APPROACH.md §5. Exact
`Frame`/`Timestamp` can shift slightly run-to-run; the identified *location* does not.

The same command works unmodified against a YouTube URL — provider-specific handling is
isolated to media resolution only (APPROACH.md §3):

```bash
python -m dialogue_frame_finder \
  "https://www.youtube.com/watch?v=J6jplPkbe8g" \
  "one small step for man" \
  --format worst
```

## Optional: caption-assisted latency optimization

`--use-captions` is opt-in (off by default) — when a video has captions, they're used to
locate a candidate region, then ASR runs on just that local audio window instead of the
whole video. Captions are **never** used for the final timestamp/frame — only real ASR is.
Falls back automatically to the normal full-video path if captions are unavailable or don't
confirm a match. See APPROACH.md §10 for the design, the real measured speedup, and a real
measured limitation.

```bash
python -m dialogue_frame_finder "<url>" "<target dialogue>" --use-captions
```

Exit code is `0` on a confident/ambiguous match, `1` on `NO_CONFIDENT_MATCH`.

## Web UI

A browser-based alternative to the CLI. It's a thin layer over the same `run_pipeline` the
CLI uses — no matching/verification/ASR logic is duplicated. Optional dependency, kept out
of `requirements.txt`:

```bash
pip install -r requirements-web.txt
$env:PYTHONPATH = "src"   # PowerShell; use `export PYTHONPATH=src` on bash
python -m uvicorn dialogue_frame_finder.api:app --reload
```

Then open `http://127.0.0.1:8000` — paste a video URL, enter the dialogue, click **Find
Frame**. Progress is shown as real pipeline stage names, never a fabricated percentage.
"Use captions" and the ASR model selector are tucked behind an **Advanced** section — the
default flow only needs a URL and a dialogue. See APPROACH.md §11 for the API/UI
architecture and §10 for the caption-assisted path the toggle enables.

## Output contract

```
Timestamp : HH:MM:SS.sss
Frame     : <frame number, where applicable>
Text      : "<extracted/identified dialogue>"
Image     : <path to the extracted frame image>
```

Timestamp, Frame, and Image always refer to the same video instant by construction (the
Timestamp reported is the *extracted frame's own* presentation timestamp, not an
intermediate estimate). `Frame` is `N/A (not available for this source/timestamp)` in the
rare case an exact frame number can't be determined — the required Timestamp and Image are
still produced. `Status` and `Confidence` are supplementary diagnostic fields, not part of
the required minimum output.

Frame images are saved under `<output_dir>/images/` (default `outputs/images/`), named
after the extracted frame's timestamp, e.g. `outputs/images/frame_325.305.png`.

### Status values

`HIGH_CONFIDENCE`, `MEDIUM_CONFIDENCE`, `LOW_CONFIDENCE`, `AMBIGUOUS_MATCH` (another
occurrence scored nearly as well as the one selected), `NO_CONFIDENT_MATCH` (all four
required fields are `None` — never a fabricated guess). See APPROACH.md §6.

## Optional: semantic fallback

Consulted only when lexical/fuzzy matching alone found no confident or an ambiguous
result (APPROACH.md §6). Off by default; enable by passing a matcher to `run_pipeline`:

```python
from dialogue_frame_finder.pipeline import run_pipeline
from dialogue_frame_finder.semantic import ClaudeSemanticMatcher

run_pipeline(url, target_text, "outputs", semantic_matcher=ClaudeSemanticMatcher())
```

Requires an `ANTHROPIC_API_KEY` environment variable (or pass `api_key=` explicitly). The
exact prompt used is recorded in [prompts.txt](prompts.txt).

## Running the tests

```bash
pytest                 # default: unit + integration, offline, ~2s
pytest -m unit          # pure logic only, no filesystem/media
pytest -m integration   # real local audio/video fixtures, no network
pytest -m e2e            # real network + real ASR against OK.ru and YouTube (slow, opt-in)
pytest --cov=dialogue_frame_finder --cov-report=term-missing   # coverage report
```

The default run (`pytest`, no flags) never touches the network — `pyproject.toml` excludes
`e2e` tests unless explicitly requested. Test fixtures are synthetic (`tests/video_fixtures.py`
generates small local audio/video files with known, exact timestamps) — no video is
committed to the repository.

## Project structure

See APPROACH.md §13.

## Design and algorithm

See [APPROACH.md](APPROACH.md) for the full architecture, the indexed-retrieval algorithm,
onset refinement, PTS-aware frame mapping, ambiguity handling, ASR model comparison, and
threshold calibration rationale.
