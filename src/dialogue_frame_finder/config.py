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

    valid_score_threshold: float = 0.6

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
    confidence_high_threshold: float = 0.85
    confidence_medium_threshold: float = 0.65
    ambiguity_score_margin: float = 0.05

    # semantic fallback (Phase 7): only ever consulted when lexical/fuzzy
    # retrieval found no confident match or a genuinely ambiguous one --
    # never responsible for timing, only for selecting among candidates
    # already located by bounded_scan_windows.
    semantic_min_confidence: float = 0.6
    semantic_max_candidates: int = 8


DEFAULT_CONFIG = SearchConfig()
