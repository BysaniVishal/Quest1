"""ASR adapter: word-level timestamped transcription, isolated behind an
interface so the rest of the pipeline never depends on which engine
produced the Transcript.

`transcript_from_whisper_words` is pure conversion logic and is unit-tested
without the real engine installed (it accepts anything duck-typed with
.word/.start/.end/.probability, or an equivalent dict). `FasterWhisperASR`
lazily imports faster-whisper only when actually used, so importing this
module -- and running the offline test suite -- never requires the real
(heavy: torch/ctranslate2) dependency to be installed.
"""

from typing import Any, Iterable, Optional, Protocol

from .transcript import Transcript


class ASRAdapter(Protocol):
    def transcribe(self, audio_path: str) -> Transcript: ...


def _get(word: Any, key: str, default: Optional[Any] = None) -> Any:
    if isinstance(word, dict):
        return word.get(key, default)
    return getattr(word, key, default)


def transcript_from_whisper_words(words: Iterable[Any]) -> Transcript:
    """Convert an iterable of faster-whisper-style word objects (or
    equivalent dicts) -- each carrying word/start/end and optionally
    probability -- into a Transcript."""
    tuples = []
    for w in words:
        text = str(_get(w, "word")).strip()
        if not text:
            continue
        start = float(_get(w, "start"))
        end = float(_get(w, "end"))
        confidence = _get(w, "probability")
        tuples.append((text, start, end, confidence) if confidence is not None else (text, start, end))
    return Transcript.from_word_tuples(tuples)


class FasterWhisperASR:
    """Real ASR adapter backed by faster-whisper. Not exercised by the
    offline test suite -- see test-plan.docx: ASR is tested through the
    conversion logic above plus mocks, with real invocation reserved for
    E2E (Phase 7)."""

    def __init__(
        self, model_size: str = "base", device: str = "cpu", compute_type: str = "int8",
        vad_filter: bool = True,
    ):
        self._model_size = model_size
        self._device = device
        self._compute_type = compute_type
        # VAD-based chunking, not just an accuracy nicety: without it,
        # faster-whisper's feature extractor builds one array covering the
        # ENTIRE audio at once, which is memory-prohibitive for a long
        # video (real-world finding: a 54-minute source exhausted available
        # RAM in the Phase 8 E2E validation environment). VAD filtering also
        # skips silence, which is a net accuracy improvement, not just a
        # workaround.
        self._vad_filter = vad_filter
        self._model = None

    def _ensure_model(self):
        if self._model is None:
            import os
            # huggingface_hub's default model cache uses symlinks, which
            # requires a Windows privilege not granted by default (real
            # finding from Phase 8 E2E validation: model download failed
            # with WinError 1314 without this). Set only if the user hasn't
            # already expressed a preference.
            os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")
            from faster_whisper import WhisperModel
            self._model = WhisperModel(
                self._model_size, device=self._device, compute_type=self._compute_type
            )
        return self._model

    def transcribe(self, audio_path: str) -> Transcript:
        model = self._ensure_model()
        segments, _info = model.transcribe(
            audio_path, word_timestamps=True, vad_filter=self._vad_filter
        )
        words = []
        for segment in segments:
            if segment.words:
                words.extend(segment.words)
        return transcript_from_whisper_words(words)
