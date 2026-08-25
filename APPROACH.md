# Approach — Dialogue-to-Exact-Video-Frame Finder

This document explains the design, algorithms, assumptions, and trade-offs behind the
solution, and directly answers the evaluation questions from the Quest1 problem statement:
(3) how the solution determines where to look in the video, (4) how it determines the
relevant frame, (5) how it extracts the text, and (6) how ambiguous/uncertain results are
handled. LLM prompts (evaluation item 1) are recorded verbatim in `prompts.txt`.

## 1. Problem framing

The target dialogue is **spoken**, not necessarily displayed as on-screen text. The core
signal is therefore audio, not video/OCR. The primary problem is speech-to-video temporal
alignment: locate where in the audio the target phrase is spoken, refine that to a precise
onset, then map that onset to the video's own frame grid. No OCR or full-video visual
analysis is used anywhere in the pipeline.

## 2. End-to-end architecture

```
URL + target dialogue
  -> MediaResolver            (yt-dlp; ONLY provider-specific component)
  -> audio extraction          (PyAV, mono 16kHz)
  -> ASR                       (faster-whisper, word-level timestamps)
  -> indexed retrieval          (inverted index, rarity-ranked anchors)
  -> local verification         (word alignment, lexical+coverage+contiguity)
  -> earliest-valid selection    (by TIME, never by score)
  -> [optional semantic fallback, only if lexical retrieval was weak/ambiguous]
  -> onset refinement            (first-target-word ASR timestamp, primary;
                                   bounded local audio refinement, secondary)
  -> PTS-aware frame mapping      (first decoded frame with PTS >= onset)
  -> frame extraction + save
  -> output contract (Timestamp / Frame / Text / Image + confidence)
```

Everything after MediaResolver operates on a plain decoded media source and is identical
for every provider (see §3). Source modules: `src/dialogue_frame_finder/`.

## 3. How the solution determines where to look in the video (evaluation item 3)

**Provider isolation.** `media_resolver.py` is the *only* component that knows which
platform a URL belongs to. It resolves OK.ru/YouTube/other URLs to a plain local media
file via yt-dlp, with provider-specific quirks (found empirically, not assumed) isolated
here only:
- OK.ru: the initial page fetch resets intermittently — retried with backoff (default 6
  attempts); its CDN's TLS chain isn't trusted by ffmpeg's bundled gnutls even though
  yt-dlp's own Python layer tolerates it — `-tls_verify 0` passed through to the ffmpeg
  downloader.
- YouTube: needs a JS runtime (Node) and, as of this build, `player_client=android,
  web_safari` extractor args to clear a bot-check without cookies.

Everything downstream — indexing, matching, onset refinement, frame mapping — takes no
provider-specific arguments at all (enforced by
`tests/unit/test_provider_independence.py`, a structural check that these functions
cannot even see which provider produced their input).

**Indexed candidate retrieval, not brute-force scanning.** The transcript (from ASR, see
§5) is built into a lightweight in-memory inverted index (`index.py`): a dict from
normalized word to the list of transcript positions where it occurs, plus a flat
position-ordered word array. This is *not* a database — it exists only in memory per run.

Rather than scanning the whole transcript for the target phrase, the target's own words
and short n-grams (up to trigrams) are ranked by how rare they are *within this
transcript* (`anchors.py`) — a word appearing once is a far better search anchor than one
appearing fifty times. Retrieval (`retrieval.py`) tries the several rarest single words
(not just the single rarest anchor overall) plus the single most selective anchor overall,
and unions their index postings into candidate positions. Trying multiple unigrams, not
just the best one, matters in practice: if ASR corrupts a *different* word in each
occurrence of the target, the single most selective anchor (often a rare n-gram) may exist
in only one of them, silently hiding the rest — this was an actual bug found and fixed
during development (`test_retrieve_candidates_recall_survives_different_word_corrupted_per_occurrence`).

**Fallback tiers**, escalated only when the previous tier finds nothing:
1. `exact_anchor` — the index lookup above.
2. `fuzzy_anchor` — approximate (edit-distance) vocabulary lookup, for when ASR corrupts
   the anchor word itself beyond exact matching (`fallback.py`).
3. `bounded_scan` — tiles the whole transcript into verification windows as a last
   resort, when no anchor (exact or fuzzy) exists anywhere.
4. `semantic_fallback` — optional, LLM-based, see §6 below.

Every result reports which tier actually produced it (`SearchResult.tier_used`) — never
silent.

**Local verification, not string equality.** Index hits are candidate *locations*, not
matches. For each, a small neighborhood around it (`neighborhood.py`, target length plus a
tolerance margin, clipped to transcript bounds) is compared against the *entire* target
phrase via word-level alignment (`align.py`): a small dynamic-programming alignment with
free leading/trailing skips (filler words around the target cost nothing) but a real
internal gap penalty using **signed** match scores (`2*similarity - 1`), so a poor pairing
actively costs almost as much as a true gap — without this, two unrelated words could look
"free" to align rather than being correctly treated as a mismatch. This is what makes word
order matter without a separate "sequence similarity" metric: a scrambled candidate cannot
achieve a good monotonic alignment path.

`verification.py` turns that alignment into: lexical similarity (average per-word
similarity), coverage (fraction of target words actually matched above a per-word
threshold), and contiguity (how close together the matched words are in the window), then
a weighted score. A window is valid only above a configurable threshold.

**Earliest-valid selection — the target may occur more than once.** All valid windows
across the whole transcript are kept (`selection.py`), then sorted by **video time**, not
score, and the earliest wins. A later, higher-scoring occurrence is retained only as
diagnostic metadata (`SearchResult.other_valid`), feeding the ambiguity check in §6 — it
never overrides the earliest-time choice.

## 4. How the solution determines the relevant frame (evaluation item 4)

**Onset: the ASR timestamp of the target's own first word is primary — not general
speech onset.** The word-level alignment above identifies which transcript word matches
the target's first word specifically; that word's own ASR timestamp is the coarse onset.
This deliberately is *not* "where does speech begin nearby" — those are different
questions. If the target follows other dialogue with no pause, or starts while another
speaker is already talking, a naive VAD pass over the whole utterance would report the
wrong (earlier) instant.

**Bounded, directional local refinement (`onset.py`).** A short audio window — at most
`onset_pre_roll` (0.15s default) before the ASR timestamp to `onset_post_roll` (0.30s) after
— is inspected for a genuine acoustic transition to snap to, correcting ASR's typical
late/rounded bias. Two safeguards prevent this from becoming the "general VAD" mistake
above:
- Energies are smoothed (3-tap moving average) before classification, since a real onset
  is a sustained multi-frame rise, not a single-frame blip.
- A hard gate requires the window's energy dynamic range to clear `onset_dynamic_range_ratio`
  (3x default) before any transition-hunting runs at all — a genuine silence/speech
  boundary swings energy by an order of magnitude; ordinary frame-to-frame variance within
  otherwise-uniform audio does not. Without this gate, per-window min-max normalization
  alone would always find *some* frame near the window's own minimum and misclassify it as
  "silence" relative to the window's own maximum, even in continuous speech — a real bug
  caught by testing the exact continuous-speech scenarios below, not a hypothetical.

If the window shows continuous speech throughout (no pause before the target, or another
speaker already talking) or no speech at all, refinement is a deliberate no-op: the ASR
timestamp is kept unchanged, with a diagnostic reason recorded
(`no_local_transition_continuous_speech` / `no_speech_detected`) rather than a guess.
Directly tested: `test_refine_onset_target_follows_other_speech_with_no_pause` and
`test_refine_onset_target_starts_amid_already_ongoing_speech`.

**PTS-aware frame mapping, never `timestamp * FPS`.** `frame_mapping.py` decodes the video
(PyAV) and selects the **first decoded frame whose actual presentation timestamp (PTS) is
>= the refined onset** — the operational definition of "first frame." This is necessary,
not a nicety: the Phase 0 feasibility spike found real inter-frame PTS gaps in the actual
supplied OK.ru source are *not* perfectly uniform even at a nominal 23.98fps, and a
dedicated test fixture exercises a genuinely irregular (VFR-like) PTS sequence to confirm
selection is PTS-driven, not FPS-arithmetic. By default this decodes from the start of the
video up to the target instant to obtain an exact frame number (OUT-02, "frame number
where applicable") — a real, one-time cost proportional to how far into the video the
match falls, not the file's total length, and bounded because it only runs once, after the
target instant is already known from every prior stage. A caller processing very long
sources with very late targets can inject a different `locate_frame_fn` (e.g. the
efficient keyframe-seek mode, which reports no frame number) if that cost becomes an issue
— the pipeline's dependency-injection points exist exactly for this.

Video-less (audio-only) media raises a clear `ValueError` rather than an opaque
`IndexError` — a small but deliberate compliance fix (test-plan.docx's negative-test
requirement that failures be explicit, not silent).

## 5. How the solution extracts the text (evaluation item 5)

Word-level ASR (faster-whisper, see §8 for the model choice) produces the transcript that
both indexing (§3) and onset extraction (§4) operate on. The reported `Text` field is the
**actual transcribed words** spanning the matched region (`output.py:
extract_matched_text`) — i.e. what the system actually heard and matched against, in
original casing, not the target text echoed back. This is deliberate: on the real supplied
OK.ru example, ASR mis-heard "rebels at" as "reveals its," and the reported Text correctly
shows "My mind reveals its stagnation." — letting an evaluator see exactly what evidence
the match is based on, rather than silently substituting the target phrase back into the
output.

## 6. How ambiguous or uncertain results are handled (evaluation item 6)

Status classification (`output.py: classify_confidence`) is checked in a specific order
and is deliberately independent of *which* occurrence gets selected (§3's earliest-time
rule always applies first):

1. `NO_CONFIDENT_MATCH` — no valid occurrence found by any retrieval tier. All four output
   fields are `None` rather than a fabricated placeholder.
2. `AMBIGUOUS_MATCH` — checked next, and can override an otherwise-high score: if the best
   *other* valid occurrence's score is within `ambiguity_score_margin` (0.05 default) of
   the chosen (earliest) one's score, that's flagged explicitly. This directly implements
   the "same phrase occurs twice" case (test-plan.docx TM-07: "return strongest candidate
   or `AMBIGUOUS_MATCH` when scores are too close") without reopening the earliest-time
   selection rule — ambiguity is reported *alongside* the earliest choice, never used to
   swap it for a different one.
3. `HIGH_CONFIDENCE` / `MEDIUM_CONFIDENCE` / `LOW_CONFIDENCE` — absolute score bands
   (0.85 / 0.65 defaults) otherwise. Calibration rationale in §9.

Confidence and diagnostic fields (`OutputRecord.confidence_score`, `.diagnostics`) are
supplementary to the four required fields, never a substitute — per requirements.docx
OUT-06, "supplementary."

**Optional LLM semantic fallback** (`semantic.py`), consulted *only* when the lexical/fuzzy
pipeline produces `NO_CONFIDENT_MATCH` or an ambiguous result — never when a confident
lexical match already exists (verified: the matcher is not even invoked in that case). It
selects among candidate windows already located by the same `bounded_scan_windows` used by
the lexical fallback tier — it is never asked for, and structurally cannot supply, a
timestamp or frame number:
- The model is offered candidates by an opaque, code-generated id (`c0`, `c1`, ...), not a
  timestamp or transcript excerpt — nothing timestamp-shaped for it to hallucinate a
  variant of.
- If its answer names an id that was never actually offered, the answer is rejected
  outright (`parse_semantic_response`), not just discouraged by prompt wording.
- On a confident pick, the actual onset still comes from re-running the same
  `verify_window` word-alignment used by the lexical path on the selected window — the LLM
  never contributes a time value itself, only a selection.
- The exact prompt template is recorded in `prompts.txt`, per the Submission Process
  Flowchart requirement.

## 7. Efficiency

For a long video, the design deliberately avoids expensive frame-by-frame visual analysis
across the whole file: ASR runs once, retrieval narrows to a handful of candidate windows
via the index (§3), and audio/frame-level work only happens in small neighborhoods around
those candidates. The one exception, PTS-aware frame extraction's default exact-numbering
mode (§4), is bounded by construction: it runs once, after every prior stage has already
narrowed the target down to a single instant.

## 8. ASR model choice

Default: **faster-whisper `tiny.en`**, with VAD filtering enabled (`vad_filter=True`).

VAD filtering is not just an accuracy nicety here: without it, faster-whisper's feature
extractor builds one array covering the *entire* audio at once, which is memory-prohibitive
for a long source — this exhausted available RAM on the real 54-minute OK.ru video during
validation. VAD filtering (skip silence, chunk by detected speech) fixed this and is also
a genuine accuracy improvement independent of the memory issue.

A controlled comparison against `base.en` (same 60-second real audio neighborhood around
the actual target occurrence, isolated subprocess measurement) found: `base.en` cost 3.3x
the runtime and ~17% more peak memory, improved transcription of unrelated words elsewhere
in the passage, but made the **identical** error on the target phrase itself ("reveals its
stagnation" instead of "rebels at stagnation") — this was a consistent misrecognition
across both model sizes, not a fluke of the smaller one. `small.en` could not be evaluated
in this environment (disk space). Given no measured improvement on the thing actually being
measured, `tiny.en` remains the default; **this stayed a documented recommendation, not a
requirement** — a deployment with more compute/storage headroom, or a fix to source audio
quality (a plausible alternate cause: the validation used a deliberately low-bitrate
format to conserve bandwidth), may reasonably choose a larger model. `asr.py`'s
`FasterWhisperASR(model_size=...)` parameter makes this a one-line change, not an
architecture change.

Critically, **localization was unaffected either way** — both `tiny.en` and `base.en`
transcripts, fed through the same unmodified `search_dialogue`, produced the identical
match (score 0.8135, same tier, same validity). This is the intended behavior: the
fuzzy/lexical matching pipeline is designed to tolerate ASR error, not to depend on a
particular model getting every word right.

## 9. Threshold calibration

Per test-plan.docx's explicit instruction not to hardcode arbitrary universal tolerances
before measuring real behavior, defaults in `config.py`'s `SearchConfig` were validated
against evidence from two independent sources, not chosen a priori:

- The full synthetic unit-test suite (TM-01..09, FR-01..07 style cases): exact matches
  score 1.0; a single-word ASR substitution scores ~0.8 and remains valid; scrambled word
  order scores ~0.4 and is correctly rejected — establishing that `valid_score_threshold`
  (0.6) sits in a real gap between "corrupted but genuine" and "wrong" matches, not an
  arbitrary midpoint.
- Two real end-to-end runs (OK.ru: 2 of 5 target words ASR-corrupted, scored 0.8135,
  correctly landed in `MEDIUM_CONFIDENCE`; YouTube/Apollo 11: 1 of 5 words corrupted,
  scored 0.879, correctly landed in `HIGH_CONFIDENCE`) confirm the confidence bands
  (0.65 / 0.85) grade realistic ASR corruption sensibly — more corruption maps to a lower
  band — rather than being tuned only to the one supplied example.

No changes were made to any threshold as a result of this review; the evidence supports
the values already in place. All of them remain configurable (`SearchConfig`) for future
recalibration if broader evaluation data becomes available.

## 10. Known limitations and assumptions

- ASR accuracy is inherently model- and audio-quality-dependent; the system is designed to
  *tolerate* ASR error, not eliminate it (see §8).
- `MediaResolver`'s YouTube bot-check workaround (`player_client=android,web_safari`) is a
  currently-effective mitigation for a widely-known, evolving yt-dlp/YouTube friction
  point — it could require updating if YouTube changes its detection again; this is an
  external dependency risk, not a defect in this codebase.
- Exact frame numbering's decode-from-start cost (§4) scales with how far into a video the
  target falls; a caller with a very long source and a very late target should inject a
  bounded `locate_frame_fn` if that cost matters for their use case.
- Semantic fallback (§6) requires network access and an Anthropic API key; it is optional
  and the pipeline works fully without it (lexical/fuzzy retrieval alone).

## 11. Repository structure

```
src/dialogue_frame_finder/   complete source
tests/unit/                  pure-logic tests, no network/media (pytest -m unit)
tests/integration/           real local audio/video fixtures, no network (pytest -m integration)
tests/e2e/                   real network + real ASR, opt-in only (pytest -m e2e)
tests/video_fixtures.py      synthetic media generators shared by integration/e2e tests
outputs/images/              runtime output location (gitignored; generated per run)
README.md                    setup, usage, examples
APPROACH.md                  this document
prompts.txt                  every LLM prompt that materially influences the solution
requirements.txt             dependencies
pyproject.toml               pytest configuration (default run excludes e2e)
```
