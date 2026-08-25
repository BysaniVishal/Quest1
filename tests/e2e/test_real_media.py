"""Real end-to-end validation against the actual supplied Quest1 example
(OK.ru) and a real YouTube video, per requirements.docx's acceptance
criteria and test-plan.docx section 6 ("Media/Platform Tests").

Excluded from the default suite (network + real ASR compute; run explicitly
with `pytest -m e2e`). Both scenarios below were manually validated during
Phase 8 -- see the plan/APPROACH notes for the actual observed output,
including the real frame images. Assertions here are intentionally loose
around exact scores/timestamps (a different faster-whisper version or model
size could shift them slightly) and strict about the things that must never
regress: a confident match is found, the reported text is real transcript
content (not fabricated), and the image file is a genuine, valid frame.

Uses the tiny.en model with VAD filtering for tractable run time; a
production run would use a larger model for better accuracy (see Phase 9
threshold-calibration notes).
"""

import pytest
from PIL import Image

from dialogue_frame_finder.asr import FasterWhisperASR
from dialogue_frame_finder.media_resolver import MediaResolver
from dialogue_frame_finder.output import MatchStatus
from dialogue_frame_finder.pipeline import run_pipeline

pytestmark = pytest.mark.e2e


def _asr():
    return FasterWhisperASR(model_size="tiny.en", device="cpu", compute_type="int8")


def test_okru_reference_example(tmp_path):
    """The supplied Quest1 example: requirements.docx explicitly names this
    URL and dialogue and requires the four-field output be produced from it."""
    url = "https://ok.ru/video/248244667877"
    target = "My mind rebels at stagnation"

    resolver = MediaResolver(tmp_path, format_selector="worst")
    result = run_pipeline(url, target, tmp_path, media_resolver=resolver, asr_adapter=_asr())
    record = result.output

    assert record.status != MatchStatus.NO_CONFIDENT_MATCH
    assert record.timestamp is not None
    assert record.text  # non-empty -- real transcribed text, not fabricated
    # the ASR may not transcribe every word perfectly, but real recognizable
    # words from the target phrase must appear
    assert any(w in record.text.lower() for w in ("mind", "stagnation"))

    assert record.image_path is not None
    image = Image.open(record.image_path)
    assert image.size[0] > 0 and image.size[1] > 0


def test_youtube_reference_example(tmp_path):
    """A real YouTube source, verifying provider-independent behavior
    through the same core pipeline (design.docx section 12). Target
    confirmed present via a prior transcribe-only run against this exact
    clip, not guessed."""
    url = "https://www.youtube.com/watch?v=J6jplPkbe8g"
    target = "one small step for man"

    resolver = MediaResolver(tmp_path, format_selector="worst[height>=240]/worst")
    result = run_pipeline(url, target, tmp_path, media_resolver=resolver, asr_adapter=_asr())
    record = result.output

    assert record.status != MatchStatus.NO_CONFIDENT_MATCH
    assert record.timestamp is not None
    assert record.text
    assert any(w in record.text.lower() for w in ("one", "small", "step"))

    assert record.image_path is not None
    image = Image.open(record.image_path)
    assert image.size[0] > 0 and image.size[1] > 0
