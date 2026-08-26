"""Full pipeline assembly: URL + target dialogue -> required output.

MediaResolver -> audio extraction -> ASR -> indexed retrieval/verification
(Phase 1+2) -> onset refinement (Phase 3) -> PTS-aware frame extraction
(Phase 4) -> output contract (Phase 5).

Every I/O-heavy step (media resolution, ASR, audio extraction, frame
location, image save) is independently injectable so the orchestration
itself is unit-testable without real network, ffmpeg, or a real ASR model --
per test-plan.docx's instruction to use dependency injection for exactly
these. Real implementations are the defaults, so calling this with just a
URL and target text works end to end unmodified.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Union

from .asr import ASRAdapter, FasterWhisperASR
from .audio import extract_audio_clip as _extract_audio_clip
from .captions import CaptionSource, try_caption_assisted_transcript
from .config import DEFAULT_CONFIG, SearchConfig
from .frame_mapping import FrameResult, locate_frame as _locate_frame, save_frame_image as _save_frame_image
from .media_resolver import MediaResolver, ResolvedMedia
from .onset import AudioClip, resolve_dialogue_onset
from .output import OutputRecord, build_output_record
from .search import SearchResult, search_dialogue
from .semantic import SemanticMatcher, apply_semantic_fallback
from .transcript import Transcript


def _default_locate_frame(path: str, target_seconds: float) -> FrameResult:
    """Default frame lookup used by run_pipeline: requests an exact frame
    number (OUT-02, "frame number where applicable") rather than the bare
    keyframe-seek path, which never reports one. This decodes from the
    start of the video up to target_seconds -- cheap relative to the
    video's length for how far into it the match actually falls (here,
    once, after the target instant is already known -- not a full-video
    scan), but still a real decode cost for a very late target in a very
    long video. locate_frame's efficient (exact_frame_number=False) mode
    remains available to any caller that wants to bound that cost by
    passing its own locate_frame_fn -- this default only changes which of
    frame_mapping.locate_frame's two existing, already-tested modes the
    pipeline reaches for out of the box."""
    return _locate_frame(path, target_seconds, exact_frame_number=True)


@dataclass(frozen=True)
class PipelineResult:
    output: OutputRecord
    resolved_media: ResolvedMedia
    transcript: Transcript
    search_result: SearchResult
    transcript_source: str = "full_video_asr"


def run_pipeline(
    url: str,
    target_text: str,
    output_dir: Union[str, Path],
    media_resolver: Optional[MediaResolver] = None,
    asr_adapter: Optional[ASRAdapter] = None,
    extract_audio_clip_fn: Callable[[str], AudioClip] = _extract_audio_clip,
    locate_frame_fn: Callable[..., FrameResult] = _default_locate_frame,
    save_frame_image_fn: Callable[..., None] = _save_frame_image,
    semantic_matcher: Optional[SemanticMatcher] = None,
    caption_source: Optional[CaptionSource] = None,
    config: SearchConfig = DEFAULT_CONFIG,
    on_stage: Optional[Callable[[str], None]] = None,
) -> PipelineResult:
    """`on_stage` (optional): called with a short human-readable label at
    each real stage boundary below -- purely additive, no-op by default, so
    every existing caller (CLI, tests) is unaffected. Added for the web UI's
    progress display; carries no timing/percentage claim, only real stage
    names, since the pipeline has no way to estimate completion percentage."""
    def _stage(message: str) -> None:
        if on_stage is not None:
            on_stage(message)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if media_resolver is None:
        media_resolver = MediaResolver(output_dir)
    if asr_adapter is None:
        asr_adapter = FasterWhisperASR()

    _stage("Downloading video...")
    resolved = media_resolver.resolve(url)
    _stage("Extracting audio...")
    audio_clip = extract_audio_clip_fn(resolved.local_path)

    # Caption-assisted local ASR (latency optimization, off by default --
    # pass a real CaptionSource() to opt in): captions only ever narrow
    # WHICH audio gets transcribed, never how it's matched or timed --
    # see captions.py's module docstring for the full rationale and the
    # real investigation behind it. Any failure at any stage (no captions,
    # no coarse match, local ASR not independently confirming a match)
    # falls through to the exact same full-video ASR path used when this
    # feature is disabled entirely.
    transcript = None
    transcript_source = "full_video_asr"
    if caption_source is not None:
        _stage("Checking captions...")
        assist = try_caption_assisted_transcript(
            url, target_text, audio_clip, caption_source, asr_adapter, config, on_stage=_stage
        )
        if assist.used:
            transcript = assist.transcript
            transcript_source = "captions_local_asr"

    if transcript is None:
        _stage("Transcribing audio (full video)...")
        transcript = asr_adapter.transcribe(resolved.local_path)

    _stage("Verifying dialogue...")
    search_result = search_dialogue(target_text, transcript, config)
    if semantic_matcher is not None:
        search_result = apply_semantic_fallback(search_result, target_text, transcript, semantic_matcher, config)
    onset_result = resolve_dialogue_onset(search_result, audio_clip, config)

    frame_result = None
    image_path = None
    if onset_result is not None:
        _stage("Locating exact frame...")
        frame_result = locate_frame_fn(resolved.local_path, onset_result.refined_onset)
        # Runtime output images live under <output_dir>/images/ (not mixed
        # in with downloaded media or Claude's scratchpad), e.g.
        # outputs/images/frame_325.305.png -- name the file after the
        # frame's own PTS (what build_output_record also reports as
        # Timestamp), not the pre-quantization onset estimate used only to
        # locate it, so the displayed Timestamp and the image filename
        # agree rather than differing by a fraction of a frame.
        images_dir = output_dir / "images"
        images_dir.mkdir(parents=True, exist_ok=True)
        image_path = str(images_dir / f"frame_{frame_result.pts_seconds:.3f}.png")
        save_frame_image_fn(frame_result.image, image_path)

    output = build_output_record(
        search_result, transcript, onset_result, frame_result, image_path, config
    )
    output.diagnostics["transcript_source"] = transcript_source

    return PipelineResult(
        output=output,
        resolved_media=resolved,
        transcript=transcript,
        search_result=search_result,
        transcript_source=transcript_source,
    )
