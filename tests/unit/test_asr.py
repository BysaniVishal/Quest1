from types import SimpleNamespace

import pytest

from dialogue_frame_finder.asr import transcript_from_whisper_words

pytestmark = pytest.mark.unit


def test_transcript_from_whisper_words_object_style():
    words = [
        SimpleNamespace(word=" My", start=0.7, end=1.0, probability=0.95),
        SimpleNamespace(word=" mind", start=1.0, end=1.3, probability=0.9),
    ]
    transcript = transcript_from_whisper_words(words)
    assert [w.word for w in transcript.words] == ["My", "mind"]
    assert transcript.words[0].start == pytest.approx(0.7)
    assert transcript.words[0].confidence == pytest.approx(0.95)
    assert transcript.words[0].normalized == "my"


def test_transcript_from_whisper_words_dict_style():
    words = [{"word": " stagnation", "start": 1.8, "end": 2.3}]
    transcript = transcript_from_whisper_words(words)
    assert transcript.words[0].word == "stagnation"
    assert transcript.words[0].confidence is None


def test_transcript_from_whisper_words_strips_whitespace():
    words = [SimpleNamespace(word="  hello  ", start=0.0, end=0.5, probability=None)]
    transcript = transcript_from_whisper_words(words)
    assert transcript.words[0].word == "hello"


def test_transcript_from_whisper_words_skips_empty_words():
    words = [
        SimpleNamespace(word=" ", start=0.0, end=0.1, probability=None),
        SimpleNamespace(word=" hi", start=0.1, end=0.3, probability=None),
    ]
    transcript = transcript_from_whisper_words(words)
    assert len(transcript.words) == 1
    assert transcript.words[0].word == "hi"


def test_transcript_from_whisper_words_empty_input():
    assert transcript_from_whisper_words([]).words == []
