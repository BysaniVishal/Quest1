"""Semantic fallback: LLM-based candidate selection, consulted only when
lexical/fuzzy retrieval (Phase 1+2) found no confident match or a genuinely
ambiguous one.

Per design.docx section 15 ("AI/LLM Usage") this stays strictly
non-authoritative on timing: the LLM never invents a timestamp or frame
number. Its only role is picking which ALREADY-LOCATED candidate window (if
any) most plausibly represents the target dialogue; the actual onset
timestamp still comes from that candidate's own word-level ASR alignment,
exactly as in the lexical path (verify_window). If the LLM's answer names a
candidate_id that was never offered to it, that answer is rejected outright
-- this is the concrete guard against a hallucinated match, not just a
documentation promise.

Every prompt that materially influences the solution is recorded in
prompts.txt (Submission Process Flowchart requirement), including this
exact template.
"""

import json
import re
from dataclasses import dataclass
from typing import List, Optional, Protocol, Tuple

from .config import DEFAULT_CONFIG, SearchConfig
from .fallback import bounded_scan_windows
from .index import TranscriptIndex
from .normalization import normalize_words
from .search import SearchResult
from .transcript import Transcript
from .verification import VerificationResult, verify_window

PROMPT_TEMPLATE = """Given a target dialogue and timestamped transcript candidates, select the candidate that most likely represents the target. Do not invent words or timestamps. Return structured JSON containing candidate_id, match_confidence, and rationale. If the evidence is insufficient, return NO_CONFIDENT_MATCH.

Target dialogue: "{target_text}"

Candidates:
{candidate_list}

Respond with ONLY a JSON object, no other text, in exactly this shape:
{{"candidate_id": "<id or null>", "match_confidence": <0.0-1.0>, "rationale": "<short reason>"}}
Use candidate_id null and match_confidence 0.0 if none of the candidates plausibly represent the target (NO_CONFIDENT_MATCH)."""


@dataclass(frozen=True)
class SemanticCandidateInput:
    candidate_id: str
    text: str
    start_time: float
    window: Tuple[int, int]


@dataclass(frozen=True)
class SemanticMatchResult:
    candidate_id: Optional[str]
    match_confidence: float
    rationale: str


class SemanticMatcher(Protocol):
    def match(self, target_text: str, candidates: List[SemanticCandidateInput]) -> SemanticMatchResult: ...


def build_semantic_prompt(target_text: str, candidates: List[SemanticCandidateInput]) -> str:
    candidate_list = "\n".join(
        f'- id={c.candidate_id} start={c.start_time:.3f}s text="{c.text}"' for c in candidates
    )
    return PROMPT_TEMPLATE.format(target_text=target_text, candidate_list=candidate_list)


def parse_semantic_response(raw_text: str, valid_candidate_ids: List[str]) -> SemanticMatchResult:
    """Parse the LLM's JSON response, tolerating surrounding text/code
    fences. A candidate_id not among those actually offered is rejected as
    NO_CONFIDENT_MATCH -- the LLM does not get to name a candidate that was
    never given to it."""
    match = re.search(r"\{.*\}", raw_text, re.DOTALL)
    if not match:
        return SemanticMatchResult(candidate_id=None, match_confidence=0.0, rationale="unparseable response")

    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return SemanticMatchResult(candidate_id=None, match_confidence=0.0, rationale="invalid JSON response")

    candidate_id = data.get("candidate_id")
    confidence = float(data.get("match_confidence", 0.0) or 0.0)
    rationale = str(data.get("rationale", ""))

    if candidate_id is None or candidate_id not in valid_candidate_ids:
        return SemanticMatchResult(candidate_id=None, match_confidence=0.0, rationale=rationale or "no valid candidate")

    return SemanticMatchResult(candidate_id=candidate_id, match_confidence=confidence, rationale=rationale)


class ClaudeSemanticMatcher:
    """Real matcher backed by the Anthropic API. Lazily imports the SDK so
    importing this module never requires it to be installed -- only using
    this specific class does."""

    def __init__(self, model: str = "claude-sonnet-5", api_key: Optional[str] = None):
        self._model = model
        self._api_key = api_key
        self._client = None

    def _ensure_client(self):
        if self._client is None:
            import anthropic
            self._client = anthropic.Anthropic(api_key=self._api_key) if self._api_key else anthropic.Anthropic()
        return self._client

    def match(self, target_text: str, candidates: List[SemanticCandidateInput]) -> SemanticMatchResult:
        client = self._ensure_client()
        prompt = build_semantic_prompt(target_text, candidates)
        response = client.messages.create(
            model=self._model,
            max_tokens=300,
            temperature=0,
            messages=[{"role": "user", "content": prompt}],
        )
        raw_text = "".join(block.text for block in response.content if getattr(block, "type", None) == "text")
        return parse_semantic_response(raw_text, [c.candidate_id for c in candidates])


def gather_semantic_candidates(
    target_words: List[str], transcript: Transcript, config: SearchConfig = DEFAULT_CONFIG
) -> List[SemanticCandidateInput]:
    windows = bounded_scan_windows(len(target_words), len(transcript), config.neighborhood_tolerance)
    candidates = []
    for i, (start, end) in enumerate(windows[: config.semantic_max_candidates]):
        text = " ".join(w.word for w in transcript.words[start:end + 1])
        candidates.append(
            SemanticCandidateInput(
                candidate_id=f"c{i}", text=text, start_time=transcript.words[start].start, window=(start, end)
            )
        )
    return candidates


def should_use_semantic_fallback(search_result: SearchResult, config: SearchConfig = DEFAULT_CONFIG) -> bool:
    if search_result.chosen is None:
        return True
    if search_result.other_valid:
        best_other = max(r.score for r in search_result.other_valid)
        return (search_result.chosen.score - best_other) < config.ambiguity_score_margin
    return False


def apply_semantic_fallback(
    search_result: SearchResult,
    target_text: str,
    transcript: Transcript,
    matcher: SemanticMatcher,
    config: SearchConfig = DEFAULT_CONFIG,
) -> SearchResult:
    """If lexical/fuzzy retrieval found nothing confident or something
    ambiguous, consult the semantic matcher. On a confident match, re-run
    verify_window on the LLM-selected candidate to obtain a real word-level
    alignment (and therefore a real, non-invented first-word timestamp),
    reporting the semantic matcher's own confidence as the score since the
    lexical score was already established as insufficient by construction.
    Otherwise returns search_result unchanged."""
    if not should_use_semantic_fallback(search_result, config):
        return search_result

    target_words = normalize_words(target_text)
    candidates = gather_semantic_candidates(target_words, transcript, config)
    if not candidates:
        return search_result

    match = matcher.match(target_text, candidates)
    if match.candidate_id is None or match.match_confidence < config.semantic_min_confidence:
        return search_result

    selected = next(c for c in candidates if c.candidate_id == match.candidate_id)
    index = TranscriptIndex.build(transcript)
    verification = verify_window(target_words, index, selected.window, config)

    if verification.first_word_position is None:
        return search_result  # LLM picked a region with no recoverable word alignment at all

    semantic_result = VerificationResult(
        window_start=verification.window_start,
        window_end=verification.window_end,
        alignment=verification.alignment,
        lexical_similarity=verification.lexical_similarity,
        coverage=verification.coverage,
        contiguity=verification.contiguity,
        score=match.match_confidence,
        valid=True,
        first_word_position=verification.first_word_position,
        first_word_start=verification.first_word_start,
        first_word_end=verification.first_word_end,
    )

    return SearchResult(
        chosen=semantic_result,
        other_valid=search_result.other_valid,
        tier_used="semantic_fallback",
        anchors_used=search_result.anchors_used,
        windows_verified=search_result.windows_verified + len(candidates),
    )
