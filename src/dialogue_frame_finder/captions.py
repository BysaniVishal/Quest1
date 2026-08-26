"""Caption-assisted coarse localization -- a latency optimization only.

Real investigation (empirical, on the actual videos used in this project)
found that YouTube auto-caption word-level offsets (`tOffsetMs` in the
`json3` format) are absent entirely for real videos we tested: every word
silently inherits its caption block's own start time, and a block can span
many seconds. Naively trusting a caption timestamp as the dialogue onset
was off by 9.3s and 2.1s in the two real cases measured -- far outside the
bounded onset-refinement window (onset.py) tuned for faster-whisper's own
(much smaller, characterized) bias.

So captions are used here for exactly one purpose: identifying WHICH short
region of a (possibly very long) video plausibly contains the target
dialogue, via the *exact same* `search_dialogue` used everywhere else in
this pipeline -- text-based verification doesn't care about timestamp
precision, only timestamp *presence*, so a coarse (block-level) transcript
is perfectly usable for this step even though it is not usable for onset
timing. Once a candidate region is found, real ASR (the same ASRAdapter
used for the full-video path, unmodified) still produces the actual
word-level timestamps that onset refinement and frame mapping operate on --
just for a small local audio window instead of the entire video. This
keeps `search_dialogue`, `verification.py`, `onset.py`, and
`frame_mapping.py` completely untouched: captions only ever change WHICH
audio gets transcribed, never how transcripts get matched or timed.

Both manual (human-uploaded) and auto-generated caption tracks are
accepted for this coarse step -- manual captions, when available, actually
have *better* text fidelity (no ASR error) which only helps the coarse
match; word-level precision is irrelevant here since it's never used for
timing.
"""

import tempfile
import wave
from dataclasses import dataclass
from typing import Any, Callable, Optional
from urllib.request import urlopen

import numpy as np

from .asr import ASRAdapter
from .config import DEFAULT_CONFIG, SearchConfig
from .media_resolver import _provider_ydl_opts, detect_provider
from .onset import AudioClip
from .search import search_dialogue
from .transcript import Transcript, WordEntry
from .verification import VerificationResult


def _parse_json3(data: dict) -> Transcript:
    words = []
    for event in data.get("events", []):
        start_ms = event.get("tStartMs")
        dur_ms = event.get("dDurationMs")
        if start_ms is None or dur_ms is None:
            continue
        start = start_ms / 1000.0
        end = (start_ms + dur_ms) / 1000.0
        text = "".join(seg.get("utf8", "") for seg in event.get("segs", []) if "utf8" in seg)
        for token in text.replace("\n", " ").split():
            words.append(WordEntry(word=token, normalized=token.lower().strip(".,!?'\""), start=start, end=end))
    return Transcript(words=words)


def _pick_track(tracks_by_lang: dict, language: str) -> Optional[dict]:
    candidates = tracks_by_lang.get(language) or next(
        (v for k, v in tracks_by_lang.items() if k.startswith(language)), None
    )
    if not candidates:
        return None
    return next((t for t in candidates if t.get("ext") == "json3"), None)


class CaptionSource:
    """Fetches a coarse, block-level Transcript from a video's caption
    track, if one exists. Never raises for "no usable captions" -- only
    for genuine, unexpected errors during a real fetch attempt would
    propagate; anything about caption *availability* returns None so the
    pipeline can fall through to the full-video ASR path cleanly."""

    def __init__(
        self,
        ydl_class: Optional[Callable] = None,
        url_opener: Callable[[str], Any] = urlopen,
        language: str = "en",
    ):
        if ydl_class is None:
            import yt_dlp
            ydl_class = yt_dlp.YoutubeDL
        self._ydl_class = ydl_class
        self._url_opener = url_opener
        self._language = language

    def fetch_coarse_transcript(self, url: str) -> Optional[Transcript]:
        # Reuse MediaResolver's own provider-specific yt-dlp options (e.g.
        # YouTube's bot-check workaround) rather than duplicating that
        # knowledge here -- without it, this extract_info call can fail
        # silently on the same friction MediaResolver already solves,
        # which would otherwise look indistinguishable from "no captions
        # exist" (a real gap found and fixed during real E2E validation).
        provider = detect_provider(url)
        opts = {"quiet": True, "skip_download": True, **_provider_ydl_opts(provider)}
        try:
            with self._ydl_class(opts) as ydl:
                info = ydl.extract_info(url, download=False)
        except Exception:
            return None
        if not info:
            return None

        # Prefer manual (human-uploaded) captions for text fidelity; fall
        # back to auto-generated. Word-level precision is irrelevant here
        # (see module docstring) -- only text correctness matters for the
        # coarse localization step.
        track = _pick_track(info.get("subtitles") or {}, self._language)
        if track is None:
            track = _pick_track(info.get("automatic_captions") or {}, self._language)
        if track is None:
            return None

        try:
            raw = self._url_opener(track["url"]).read()
            import json
            data = json.loads(raw)
        except Exception:
            return None

        transcript = _parse_json3(data)
        return transcript if len(transcript) > 0 else None


def compute_local_window(chosen: VerificationResult, coarse_transcript: Transcript, config: SearchConfig) -> "tuple[float, float]":
    """The candidate's own matched-block span, padded and capped -- not a
    fixed radius around a point. Real testing found the true onset falls
    WITHIN the caption block's own span (the block's duration reflects how
    long that speech took), offset from the block's start by several
    seconds -- so the block's own end matters as much as its start."""
    block_start = chosen.first_word_start
    block_end = coarse_transcript.words[chosen.window_end].end
    start = max(0.0, block_start - config.caption_window_pre_pad)
    end = block_end + config.caption_window_post_pad
    if end - start > config.caption_window_max_seconds:
        end = start + config.caption_window_max_seconds
    return start, end


def _write_temp_wav(samples: np.ndarray, sample_rate: int) -> str:
    pcm16 = np.clip(samples * 32767.0, -32768, 32767).astype(np.int16)
    fd, path = tempfile.mkstemp(suffix=".wav")
    import os
    os.close(fd)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm16.tobytes())
    return path


def transcribe_local_window(
    audio_clip: AudioClip, window: "tuple[float, float]", asr_adapter: ASRAdapter
) -> Transcript:
    """Slice the already-decoded full-video audio to `window`, transcribe
    just that slice with the SAME (unmodified) ASRAdapter used for the
    full-video path, then shift the resulting word timestamps back to
    absolute video time. Reuses ASRAdapter.transcribe(path) exactly as-is
    (via a small temp WAV file) rather than extending the ASR interface --
    any injected adapter, real or fake, works unchanged."""
    import os

    start, end = window
    start_idx = max(0, int(start * audio_clip.sample_rate))
    end_idx = min(len(audio_clip.samples), int(end * audio_clip.sample_rate))
    slice_samples = audio_clip.samples[start_idx:end_idx]

    temp_path = _write_temp_wav(slice_samples, audio_clip.sample_rate)
    try:
        local_transcript = asr_adapter.transcribe(temp_path)
    finally:
        try:
            os.remove(temp_path)
        except OSError:
            pass

    shifted_words = [
        WordEntry(word=w.word, normalized=w.normalized, start=w.start + start, end=w.end + start, confidence=w.confidence)
        for w in local_transcript.words
    ]
    return Transcript(words=shifted_words)


@dataclass(frozen=True)
class CaptionAssistResult:
    used: bool
    transcript: Optional[Transcript]
    reason: str


def try_caption_assisted_transcript(
    url: str,
    target_text: str,
    audio_clip: AudioClip,
    caption_source: CaptionSource,
    asr_adapter: ASRAdapter,
    config: SearchConfig = DEFAULT_CONFIG,
    on_stage: Optional[Callable[[str], None]] = None,
) -> CaptionAssistResult:
    """Attempt the full captions -> coarse candidate -> local ASR path.
    Returns used=False (never raises) for any of: no captions, no coarse
    match, or the real local ASR result not independently confirming a
    valid match through the same search_dialogue gate used everywhere else
    -- in every used=False case the caller should fall back to full-video
    ASR exactly as it did before this optimization existed.

    `on_stage` (optional): same purely-additive progress hook as
    pipeline.py's run_pipeline -- no-op by default."""
    def _stage(message: str) -> None:
        if on_stage is not None:
            on_stage(message)

    coarse_transcript = caption_source.fetch_coarse_transcript(url)
    if coarse_transcript is None:
        return CaptionAssistResult(False, None, "no_usable_captions")

    _stage("Finding candidate region...")
    coarse_result = search_dialogue(target_text, coarse_transcript, config)
    if coarse_result.chosen is None:
        return CaptionAssistResult(False, None, "captions_no_confident_coarse_match")

    # Try coarse candidates earliest-first (chosen, then other_valid --
    # already time-sorted), up to the configured cap. The coarse pass can
    # hit the same earliest-valid-wins-vs-false-positive pattern the rest
    # of this pipeline is designed around (e.g. a proper noun mentioned
    # several times before the true occurrence) -- never trust the first
    # candidate blindly; confirm each via real local ASR through the same
    # threshold gate, and move on to the next candidate if it doesn't
    # confirm, rather than giving up immediately.
    coarse_candidates = [coarse_result.chosen] + list(coarse_result.other_valid)
    for coarse_candidate in coarse_candidates[: config.caption_max_coarse_candidates]:
        window = compute_local_window(coarse_candidate, coarse_transcript, config)
        _stage("Transcribing audio (fast local pass)...")
        local_transcript = transcribe_local_window(audio_clip, window, asr_adapter)
        local_result = search_dialogue(target_text, local_transcript, config)
        if local_result.chosen is not None:
            return CaptionAssistResult(True, local_transcript, "captions_localized_local_asr_confirmed")

    return CaptionAssistResult(False, None, "local_asr_did_not_confirm_match")
