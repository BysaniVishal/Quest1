"""Configurable thresholds/weights for retrieval, verification and fallback.

None of these defaults are claimed to be universally correct -- Phase 8
(threshold calibration) is expected to tune them against real transcripts.
They live in one place so calibration touches one file, not scattered
constants.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class SearchConfig:
    # neighborhood extraction
    neighborhood_tolerance: int = 3
    max_ngram_anchor: int = 3
    max_unigram_anchors: int = 3

    # word alignment
    gap_penalty: float = -0.6
    per_word_match_threshold: float = 0.72

    # verification scoring weights (must sum to 1.0)
    weight_lexical: float = 0.40
    weight_coverage: float = 0.35
    weight_contiguity: float = 0.25

    # Calibrated 2026-08-25 against real E2E scores (OK.ru 0.8135, Apollo 11
    # 0.879) and a real false-positive case (spurious short-phrase matches
    # topping out at 0.62 vs. the true match at 0.809) -- 0.70 rejects every
    # known false positive with margin while staying below every known
    # legitimate ASR-corrupted match (lowest observed: 0.80).
    valid_score_threshold: float = 0.70

    # fallback escalation
    fuzzy_anchor_max_distance: int = 2

    # onset refinement (Phase 3): the window is intentionally small and
    # asymmetric-by-default -- it exists to nudge the ASR timestamp to a
    # genuine local acoustic transition, never to search for "where did
    # speech begin" over an arbitrary span.
    onset_pre_roll: float = 0.15
    onset_post_roll: float = 0.30
    onset_frame_ms: float = 20.0
    onset_hop_ms: float = 10.0
    onset_speech_ratio_threshold: float = 0.5
    # A real silence -> speech transition swings energy by an order of
    # magnitude or more; ordinary frame-to-frame RMS variance within
    # otherwise-uniform audio does not. Requiring the window's energy
    # dynamic range to clear this ratio before hunting for a transition
    # point prevents that ordinary variance from being mistaken for one.
    onset_dynamic_range_ratio: float = 3.0
    onset_absolute_silence_floor: float = 0.01

    # confidence/status classification (Phase 5 output)
    # Lowered 2026-08-25: known false positives cap at 0.62 (see
    # valid_score_threshold note above), so anything >=0.80 is comfortably
    # clear of coincidental matches -- real single-word-ASR-error cases
    # (0.80-0.83) deserve HIGH_CONFIDENCE, not MEDIUM.
    confidence_high_threshold: float = 0.80
    # Raised 2026-08-25 to sit strictly above valid_score_threshold (0.70):
    # previously (0.65) every valid match automatically scored above this,
    # making LOW_CONFIDENCE unreachable through the real pipeline. Now
    # [0.70, 0.75) genuinely means "barely cleared the bar."
    confidence_medium_threshold: float = 0.75
    ambiguity_score_margin: float = 0.05

    # semantic fallback (Phase 7): only ever consulted when lexical/fuzzy
    # retrieval found no confident match or a genuinely ambiguous one --
    # never responsible for timing, only for selecting among candidates
    # already located by bounded_scan_windows.
    semantic_min_confidence: float = 0.6
    semantic_max_candidates: int = 8

    # Caption-assisted local ASR (latency optimization): captions are used
    # ONLY for coarse candidate localization via the same search_dialogue
    # used everywhere else -- never for final word timing (real, measured
    # investigation found YouTube auto-caption word offsets are absent for
    # real videos in this project; block-level timestamps can be several
    # seconds away from the true onset). Once a coarse candidate block is
    # found, real ASR still runs -- just on a small local audio window
    # instead of the whole video. Padding is asymmetric-by-default and
    # span-based (added to the candidate's own matched-block start/end, not
    # a fixed radius around a point) because real testing showed the true
    # onset falls within the caption block's own span, offset from its
    # start by several seconds -- a fixed small radius around the start
    # alone would have missed it.
    caption_window_pre_pad: float = 2.0
    caption_window_post_pad: float = 3.0
    # Defensive cap: an unusually long single caption block (or several
    # merged ones) should not silently recreate the full-video-ASR cost
    # this optimization exists to avoid.
    caption_window_max_seconds: float = 60.0
    # Real testing found the coarse caption pass is subject to the SAME
    # earliest-valid-wins-vs-false-positive pattern as full ASR (a short,
    # coincidentally-repeated word like a proper noun can produce several
    # valid-but-wrong coarse candidates before the true one). Rather than
    # giving up after the first coarse candidate's local ASR fails to
    # confirm, try the next-earliest valid coarse candidates in order
    # (already time-sorted via SearchResult.other_valid), up to this cap,
    # before falling back to full-video ASR -- bounds worst-case latency.
    caption_max_coarse_candidates: int = 3


DEFAULT_CONFIG = SearchConfig()
