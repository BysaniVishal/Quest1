import pytest

from dialogue_frame_finder.transcript import Transcript, WordEntry

pytestmark = pytest.mark.unit


def test_from_word_tuples_normalizes_each_word():
    t = Transcript.from_word_tuples([("My", 0.0, 0.2), ("Mind.", 0.2, 0.5)])
    assert t.words[0].normalized == "my"
    assert t.words[1].normalized == "mind"


def test_from_word_tuples_preserves_original_word_text():
    t = Transcript.from_word_tuples([("My", 0.0, 0.2)])
    assert t.words[0].word == "My"


def test_from_word_tuples_optional_confidence():
    t = Transcript.from_word_tuples([("My", 0.0, 0.2, 0.87)])
    assert t.words[0].confidence == 0.87
    t2 = Transcript.from_word_tuples([("My", 0.0, 0.2)])
    assert t2.words[0].confidence is None


def test_len_matches_word_count():
    t = Transcript.from_word_tuples([("a", 0.0, 0.1), ("b", 0.1, 0.2), ("c", 0.2, 0.3)])
    assert len(t) == 3


def test_to_dict_from_dict_roundtrip():
    t = Transcript.from_word_tuples([("My", 0.0, 0.2), ("mind", 0.2, 0.5, 0.9)])
    restored = Transcript.from_dict(t.to_dict())
    assert restored.words == t.words


def test_to_json_from_json_roundtrip():
    t = Transcript.from_word_tuples([("stagnation", 12.05, 12.55, 0.75)])
    restored = Transcript.from_json(t.to_json())
    assert restored.words == t.words


def test_word_entry_from_dict_derives_normalized_when_missing():
    entry = WordEntry.from_dict({"word": "Stagnation.", "start": 1.0, "end": 1.5})
    assert entry.normalized == "stagnation"


def test_empty_transcript_has_zero_length():
    assert len(Transcript()) == 0
