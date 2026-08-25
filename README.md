# Dialogue-to-Exact-Video-Frame Finder

Given a publicly accessible video URL and a target spoken dialogue, locates the exact/first
video frame where that dialogue begins, and returns its timestamp, frame number (where
applicable), the identified dialogue text, and the corresponding frame as an image.

The dialogue is located by its **spoken audio**, not by on-screen text/subtitles — see
[APPROACH.md](APPROACH.md) for the full design and algorithm.

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

`--format worst` is recommended: it makes yt-dlp pull the smallest available stream
instead of the best-quality one, which downloads much faster and is far less likely to
hit a connection reset on a flaky source (this project only needs the audio to transcribe
and the frames to extract a still from, so resolution doesn't affect correctness — see
`python -m dialogue_frame_finder --help` for the full flag description). Omit it if you
want the highest-quality source frame image instead.

Example output (real run, default config — `base.en` ASR, `valid_score_threshold=0.70`):

```
Status    : HIGH_CONFIDENCE
Timestamp : 00:05:25.013
Frame     : 7792
Text      : "My mind rebels its stagnation."
Image     : outputs\images\frame_325.013.png
Confidence: 0.832
```

`Text` shows what the system actually transcribed and matched against (here, a real ASR
error — "its" instead of "at" — that the matching pipeline correctly tolerated), not the
target phrase echoed back; see APPROACH.md §5. Exact `Frame`/`Timestamp` values can shift
slightly run-to-run (different `--format` selection changes the video's own frame grid;
ASR word timing has small run-to-run variance) — the identified *location* in the video
does not.

The same command works unmodified against a YouTube URL — provider-specific handling is
isolated to media resolution only (APPROACH.md §3):

```bash
python -m dialogue_frame_finder \
  "https://www.youtube.com/watch?v=J6jplPkbe8g" \
  "one small step for man" \
  --format worst
```

Exit code is `0` on a confident/ambiguous match, `1` on `NO_CONFIDENT_MATCH`.

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

See APPROACH.md §11.

## Design and algorithm

See [APPROACH.md](APPROACH.md) for the full architecture, the indexed-retrieval algorithm,
onset refinement, PTS-aware frame mapping, ambiguity handling, ASR model comparison, and
threshold calibration rationale.
