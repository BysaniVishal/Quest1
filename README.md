# Dialogue-to-Exact-Video-Frame Finder

Given a publicly accessible video URL and a target spoken dialogue, this system finds
the **earliest occurrence of that dialogue**, determines where it begins, and returns:

- **Timestamp**
- **Frame number** (where available)
- **Actually transcribed dialogue**
- **Extracted frame image**
- **Confidence / status**

The system searches the **spoken audio**, not on-screen text.

It is designed around three core ideas:

1. **Efficient retrieval** — build an inverted index over the transcript and search
   using rare, selective anchors instead of scanning every transcript window.
2. **Accuracy-aware verification** — candidate matches are verified using word-level
   alignment, coverage, contiguity, confidence thresholds, and earliest-occurrence
   selection.
3. **Accurate frame mapping** — the final dialogue onset is mapped to the video's
   actual frame timestamps using PTS rather than assuming a constant FPS.

An optional **caption-assisted path** can reduce ASR latency when captions are available,
while preserving ASR as the source of final timing.

The project is available through both a **CLI** and a **browser-based Web UI**.
Both interfaces call the same underlying pipeline — there is no duplicated search or
frame-finding logic.

See [`APPROACH.md`](APPROACH.md) for the detailed architecture, algorithmic decisions,
accuracy trade-offs, profiling results, and design rationale.

---

## What the system does

```text
Video URL + target dialogue
        │
        ▼
   Media resolution
        │
        ▼
   Audio extraction
        │
        ▼
 ┌───────────────────────────────┐
 │ Caption-assisted path?        │
 │                               │
 │ Captions → coarse location    │
 │          → local ASR          │
 │          → confirmation       │
 └──────────────┬────────────────┘
                │
       unavailable / unconfirmed
                │
                ▼
       Full-video ASR
                │
                ▼
       Transcript indexing
                │
                ▼
      Candidate retrieval
       exact → fuzzy → scan
                │
                ▼
       Phrase verification
                │
                ▼
      Earliest valid match
                │
                ▼
       Onset refinement
                │
                ▼
      PTS-aware frame mapping
                │
                ▼
        Extract + save frame
                │
                ▼
 Timestamp / Frame / Text / Image
```

### Important design boundary

Captions are **not trusted as the final timing source**.

When captions are available, they are used only to identify a coarse region where the
dialogue may occur. The actual audio in that region is then transcribed using the same
ASR pipeline used elsewhere.

The final timestamp and frame therefore still come from **real ASR timestamps + audio
onset refinement + PTS-aware frame mapping**.

If captions are unavailable or cannot be independently confirmed, the system falls back
to the original full-video ASR pipeline.

---

# Setup

Requires **Python 3.10+**.

From the project root:

```bash
pip install -r requirements.txt
```

The core dependencies include:

* `yt-dlp` — video/media resolution
* `faster-whisper` — speech recognition
* `av` / PyAV — audio/video decoding
* `Pillow` — frame image saving
* `rapidfuzz` — fuzzy matching
* `anthropic` — optional semantic fallback

No `ffmpeg` binary on `PATH` is required. Audio/video decoding is handled through PyAV.

### Windows

If the faster-whisper model download encounters:

```text
WinError 1314
```

the application already disables Hugging Face symlink creation automatically, so no
manual symlink configuration should be necessary.

### Set `PYTHONPATH`

The project is run directly from the `src/` layout rather than installed as a package.

#### PowerShell

```powershell
$env:PYTHONPATH = "src"
```

#### bash / zsh

```bash
export PYTHONPATH=src
```

Set this once per terminal session.

---

# Quick Start — CLI

From the project root:

```bash
python -m dialogue_frame_finder "<video_url>" "<target dialogue>"
```

For example:

```bash
python -m dialogue_frame_finder \
  "https://ok.ru/video/248244667877" \
  "My mind rebels at stagnation"
```

The supplied Quest1 reference case can also be run with:

```bash
python -m dialogue_frame_finder \
  "https://ok.ru/video/248244667877" \
  "My mind rebels at stagnation" \
  --format worst
```

`--format worst` is useful when you want a smaller/faster media download.
It does not change the dialogue-search algorithm.

---

# Example output

```text
Status     : HIGH_CONFIDENCE

Timestamp  : 00:05:25.013
Frame      : 7792
Text       : "My mind rebels its stagnation."
Image      : outputs/images/frame_325.013.png

Confidence : 0.832
```

`Text` is the **actual ASR transcription**, not the target dialogue echoed back.

In the run above, the target dialogue was "My mind rebels **at** stagnation," but ASR
actually transcribed "rebels **its** stagnation" — a real, minor ASR error. The system
still identified the correct occurrence because matching is designed to tolerate this
kind of imperfection, rather than requiring an exact transcript match.

The extracted frame and timestamp correspond to the same video instant.

---

# Caption-assisted latency optimization

Caption assistance is **opt-in**:

```bash
python -m dialogue_frame_finder \
  "<video_url>" \
  "<target dialogue>" \
  --use-captions
```

When captions are available:

```text
Captions
   ↓
Coarse dialogue localization
   ↓
Identify candidate caption block
   ↓
Extract a small local audio window
   ↓
Run real ASR on that window
   ↓
Verify the dialogue again
   ↓
Continue through the normal pipeline
```

This avoids transcribing the entire video when captions can successfully localize the
dialogue.

### Accuracy boundary

The caption timestamp itself is **not used as the final timestamp**.

The local audio is re-transcribed using the same ASR adapter, and the resulting
word-level ASR timestamps are used for the final onset and frame mapping.

### Fallback

If:

* captions are unavailable,
* caption fetching fails,
* the target cannot be located in the captions, or
* local ASR cannot independently confirm the candidate,

the system automatically falls back to:

```text
Full-video ASR
     ↓
Normal indexed retrieval
     ↓
Normal verification
     ↓
Normal onset/frame mapping
```

Therefore the caption feature is a latency optimization, not a replacement for the
accuracy-critical ASR pipeline.

See [`APPROACH.md`](APPROACH.md) §10 for the implementation details and measured results.

---

# Why indexed retrieval?

A naive approach would compare the target dialogue against every possible transcript
window.

Instead, the system builds an in-memory inverted index:

```text
word → transcript positions
```

Target words are ranked by how selective they are in the transcript.

For example:

```text
target:
"the spacecraft entered lunar orbit"

common word:
"the"        → many occurrences

selective word:
"spacecraft" → few occurrences

rare n-gram:
"lunar orbit" → potentially very selective
```

The system uses selective anchors to retrieve a small set of candidate locations.

Candidate retrieval has multiple tiers:

```text
Exact anchor
     ↓
Fuzzy anchor
     ↓
Bounded scan
```

This gives the system a fast path when the ASR transcript contains useful anchor words,
while still providing recall-oriented fallbacks when ASR corrupts those anchors.

---

# Why verification is separate from retrieval

An index hit is **not considered a match**.

Each candidate is verified against the entire target phrase using word-level alignment.

The verification score combines:

```text
Lexical similarity
Coverage
Contiguity
```

A candidate must pass the configured validity threshold before it can be selected.

This separation is important because an isolated matching word should not be enough to
declare that the entire dialogue was found.

---

# Multiple occurrences

The target dialogue may occur more than once.

The system therefore:

1. Finds all valid candidate occurrences.
2. Filters candidates using the validity threshold.
3. Selects the **earliest valid occurrence by video time**.
4. Does not simply choose the highest-scoring occurrence.

This distinction matters for repeated phrases such as dialogue in a chorus, refrain,
interview question, or repeated sentence.

If another occurrence has a sufficiently similar score to the selected occurrence,
the result can be reported as:

```text
AMBIGUOUS_MATCH
```

rather than pretending the system has certainty.

---

# Exact frame selection

The final dialogue onset comes from the ASR word timestamp corresponding to the first
target word.

A bounded audio refinement step can then adjust that timestamp when a genuine acoustic
transition is present.

The system does **not** calculate:

```text
frame = timestamp × FPS
```

Instead, the video is decoded using its actual presentation timestamps (PTS), and the
system selects:

```text
first decoded frame with PTS >= refined dialogue onset
```

This matters for sources where frame timing is not perfectly represented by a simple
constant-FPS calculation.

---

# Accuracy and uncertainty handling

The system explicitly distinguishes between different confidence states:

```text
HIGH_CONFIDENCE
MEDIUM_CONFIDENCE
LOW_CONFIDENCE
AMBIGUOUS_MATCH
NO_CONFIDENT_MATCH
```

A `NO_CONFIDENT_MATCH` result does not fabricate a timestamp or frame.

The semantic fallback, when enabled, is also constrained:

* It is used only when lexical matching fails or is ambiguous.
* It chooses among candidates already located by the system.
* It cannot invent a timestamp.
* The final onset is still calculated using the normal verification/onset pipeline.

The exact semantic fallback prompt is recorded in [`prompts.txt`](prompts.txt).

---

# ASR model

The current default is:

```text
faster-whisper
model: base.en
VAD: enabled
word-level timestamps: enabled
```

VAD filtering is important for long videos because it avoids processing large amounts of
silence as one continuous audio input.

The ASR model remains configurable internally, and the Web UI exposes model selection
through its Advanced options.

The CLI currently uses the default model and does not expose an `--asr-model` flag.

See [`APPROACH.md`](APPROACH.md) §8 for the model comparison and design rationale.

---

# Performance

The system was profiled using real video runs rather than assuming that indexing or
retrieval would be the bottleneck.

A controlled profiling run measured approximately:

| Stage                                           |     No captions |        Captions |
| ----------------------------------------------- | --------------: | --------------: |
| Total                                           |          86.17s |          32.88s |
| ASR                                             | 60.975s / 70.8% |  3.426s / 10.4% |
| Download                                        | 16.815s / 19.5% | 16.653s / 50.6% |
| Frame extraction                                |   7.190s / 8.3% |  7.349s / 22.3% |
| Caption fetch                                   |               — |  4.264s / 13.0% |
| Indexing + retrieval + verification + selection |         ~0.006s |         ~0.007s |

The important architectural finding is:

> **ASR, not indexing/retrieval, is the dominant computational bottleneck when
> captions are unavailable.**

The indexed retrieval and verification stage took only a few milliseconds in this
measurement.

Once caption-assisted local ASR removes most of the full-video transcription cost,
network download and frame extraction become relatively more significant.

These measurements are from real runs and should be treated as representative evidence,
not universal timing guarantees.

---

# Web UI

The project also includes a browser-based interface.

The Web UI is intentionally a thin layer over the same pipeline used by the CLI.

There is no separate implementation of:

* dialogue retrieval
* matching
* verification
* ASR
* onset detection
* frame mapping

### Install web dependencies

```bash
pip install -r requirements-web.txt
```

Set `PYTHONPATH`:

#### PowerShell

```powershell
$env:PYTHONPATH = "src"
```

#### bash / zsh

```bash
export PYTHONPATH=src
```

Start the server:

```bash
python -m uvicorn dialogue_frame_finder.api:app --reload
```

Open:

```text
http://127.0.0.1:8000
```

Then:

1. Paste the video URL.
2. Enter the dialogue.
3. Click **Find Frame**.
4. Watch the actual pipeline stages progress.
5. View the extracted frame and result.

The UI also exposes an **Advanced** section containing options such as caption
assistance and ASR model selection.

### Progress reporting

The backend reports actual pipeline stages rather than inventing a percentage.

The browser polls the API for the current job state while the same `run_pipeline()`
function executes in the background.

This keeps the UI simple while ensuring that CLI and Web UI results come from exactly
the same implementation.

---

# Semantic fallback

The semantic fallback is optional and disabled by default.

It can be enabled programmatically:

```python
from dialogue_frame_finder.pipeline import run_pipeline
from dialogue_frame_finder.semantic import ClaudeSemanticMatcher

run_pipeline(
    url,
    target_text,
    "outputs",
    semantic_matcher=ClaudeSemanticMatcher()
)
```

An Anthropic API key is required:

```text
ANTHROPIC_API_KEY
```

The LLM does **not** generate timestamps or frames.

It can only select among candidate regions that the deterministic pipeline has already
identified.

---

# Testing

Run the default test suite:

```bash
pytest
```

The default suite is offline and does not download videos or call external APIs.

Useful test commands:

```bash
pytest -m unit
```

```bash
pytest -m integration
```

For the opt-in real-world tests:

```bash
pytest -m e2e
```

Coverage:

```bash
pytest --cov=dialogue_frame_finder --cov-report=term-missing
```

The repository currently contains **235 passing tests**.

The test suite covers:

* transcript normalization
* indexing
* anchor selection
* retrieval
* fuzzy matching
* bounded scanning
* word alignment
* verification
* earliest-occurrence selection
* confidence classification
* onset refinement
* PTS-aware frame mapping
* caption-assisted orchestration
* API behavior
* provider independence
* local audio/video fixtures

No real videos are committed to the repository.

---

# Project structure

```text
Quest1/
│
├── src/
│   └── dialogue_frame_finder/
│       ├── api.py
│       ├── cli.py
│       ├── pipeline.py
│       ├── config.py
│       │
│       ├── media_resolver.py
│       ├── audio.py
│       ├── asr.py
│       ├── captions.py
│       │
│       ├── transcript.py
│       ├── normalization.py
│       ├── index.py
│       ├── anchors.py
│       ├── retrieval.py
│       ├── fallback.py
│       ├── neighborhood.py
│       ├── align.py
│       ├── search.py
│       ├── verification.py
│       ├── selection.py
│       ├── semantic.py
│       │
│       ├── onset.py
│       ├── frame_mapping.py
│       ├── timeformat.py
│       └── output.py
│
├── web/
│   ├── index.html
│   ├── style.css
│   └── app.js
│
├── tests/
│   ├── unit/
│   ├── integration/
│   ├── e2e/
│   └── video_fixtures.py
│
├── outputs/
│   └── images/
│
├── README.md
├── APPROACH.md
├── prompts.txt
├── requirements.txt
├── requirements-web.txt
└── pyproject.toml
```

---

# Design summary

The project deliberately separates the problem into independent stages:

```text
Media resolution
      ↓
Audio / caption localization
      ↓
ASR
      ↓
Transcript representation
      ↓
Indexed retrieval
      ↓
Candidate verification
      ↓
Earliest-valid selection
      ↓
Onset refinement
      ↓
PTS-aware frame mapping
      ↓
Output
```

This separation makes the system:

* **Provider-independent** after media resolution
* **Testable** through dependency injection
* **Robust to ASR errors**
* **Explicit about uncertainty**
* **Efficient for long transcripts**
* **Accurate at the frame-selection stage**
* **Extensible without duplicating the core pipeline**

The caption-assisted path is an optimization layered on top of the same architecture,
rather than a separate implementation.

For the detailed engineering rationale, experiments, trade-offs, and known limitations,
see [`APPROACH.md`](APPROACH.md).
