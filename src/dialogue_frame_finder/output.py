"""Final output assembly: the required four-field contract (Timestamp,
Frame, Text, Image) plus confidence/diagnostic metadata.

Confidence status (design.docx section 11): HIGH_CONFIDENCE, MEDIUM_
CONFIDENCE, LOW_CONFIDENCE, AMBIGUOUS_MATCH, NO_CONFIDENT_MATCH. Ambiguity
is checked first and can override an otherwise-high score: if another valid
occurrence scores nearly as well as the chosen (earliest) one, that is
reported explicitly rather than presenting an uncertain pick as certain.
Earliest-occurrence selection (Phase 2) and confidence classification are
deliberately separate concerns -- which occurrence is chosen never depends
on this status, only how confidently it is reported.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional

from .config import DEFAULT_CONFIG, SearchConfig
from .frame_mapping import FrameResult
from .onset import OnsetRefinementResult
from .search import SearchResult
from .timeformat import format_timestamp
from .transcript import Transcript
from .verification import VerificationResult


class MatchStatus(Enum):
    HIGH_CONFIDENCE = "HIGH_CONFIDENCE"
    MEDIUM_CONFIDENCE = "MEDIUM_CONFIDENCE"
    LOW_CONFIDENCE = "LOW_CONFIDENCE"
    AMBIGUOUS_MATCH = "AMBIGUOUS_MATCH"
    NO_CONFIDENT_MATCH = "NO_CONFIDENT_MATCH"


@dataclass(frozen=True)
class OutputRecord:
    status: MatchStatus
    timestamp: Optional[str]
    frame: Optional[int]
    text: Optional[str]
    image_path: Optional[str]
    confidence_score: Optional[float]
    diagnostics: Dict[str, Any] = field(default_factory=dict)


def classify_confidence(search_result: SearchResult, config: SearchConfig = DEFAULT_CONFIG) -> MatchStatus:
    if search_result.chosen is None:
        return MatchStatus.NO_CONFIDENT_MATCH

    chosen_score = search_result.chosen.score
    if search_result.other_valid:
        best_other = max(r.score for r in search_result.other_valid)
        if (chosen_score - best_other) < config.ambiguity_score_margin:
            return MatchStatus.AMBIGUOUS_MATCH

    if chosen_score >= config.confidence_high_threshold:
        return MatchStatus.HIGH_CONFIDENCE
    if chosen_score >= config.confidence_medium_threshold:
        return MatchStatus.MEDIUM_CONFIDENCE
    return MatchStatus.LOW_CONFIDENCE


def extract_matched_text(chosen: VerificationResult, transcript: Transcript) -> str:
    """Reconstruct the actual transcribed words (original casing/form, not
    the normalized target) spanning from the first to the last matched
    target word -- the literal ASR text this match is based on."""
    matched = [e for e in chosen.alignment if e.window_index is not None]
    if not matched:
        return ""
    first_local = min(e.window_index for e in matched)
    last_local = max(e.window_index for e in matched)
    start_pos = chosen.window_start + first_local
    end_pos = chosen.window_start + last_local
    return " ".join(transcript.words[p].word for p in range(start_pos, end_pos + 1))


def build_output_record(
    search_result: SearchResult,
    transcript: Transcript,
    onset_result: Optional[OnsetRefinementResult],
    frame_result: Optional[FrameResult],
    image_path: Optional[str],
    config: SearchConfig = DEFAULT_CONFIG,
) -> OutputRecord:
    status = classify_confidence(search_result, config)

    if search_result.chosen is None:
        return OutputRecord(
            status=status,
            timestamp=None,
            frame=None,
            text=None,
            image_path=None,
            confidence_score=None,
            diagnostics={"tier_used": search_result.tier_used},
        )

    chosen = search_result.chosen
    # OUT-05: Timestamp, Frame and Image must refer to the SAME instant.
    # frame_result.pts_seconds is the actual PTS of the extracted frame
    # (discrete); onset_result.refined_onset is the continuous signal-
    # processing estimate used to locate it, and can differ from the
    # frame's own PTS by a fraction of a frame under the "first frame with
    # PTS >= onset" rule. Prefer the frame's own PTS whenever a frame was
    # actually extracted, so the reported timestamp exactly matches the
    # image/frame number rather than merely being close to them.
    if frame_result is not None:
        final_onset = frame_result.pts_seconds
    elif onset_result is not None:
        final_onset = onset_result.refined_onset
    else:
        final_onset = chosen.first_word_start

    diagnostics: Dict[str, Any] = {
        "tier_used": search_result.tier_used,
        "match_score": chosen.score,
        "other_valid_count": len(search_result.other_valid),
    }
    if onset_result is not None:
        diagnostics["onset_refined"] = onset_result.refined
        diagnostics["onset_reason"] = onset_result.reason
        diagnostics["asr_onset"] = onset_result.asr_onset
    if frame_result is not None:
        diagnostics["frame_pts_seconds"] = frame_result.pts_seconds

    return OutputRecord(
        status=status,
        timestamp=format_timestamp(final_onset) if final_onset is not None else None,
        frame=frame_result.frame_number if frame_result is not None else None,
        text=extract_matched_text(chosen, transcript),
        image_path=image_path,
        confidence_score=chosen.score,
        diagnostics=diagnostics,
    )
