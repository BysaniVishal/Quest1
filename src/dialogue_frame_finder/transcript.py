"""Transcript data model: a flat, position-indexed array of timestamped words.

This is the ASR output contract downstream stages build on. It is provider-
and ASR-engine-independent -- whatever produces word-level timestamps (real
ASR, or a synthetic fixture in tests) constructs a Transcript the same way.
"""

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .normalization import normalize_word


@dataclass(frozen=True)
class WordEntry:
    word: str
    normalized: str
    start: float
    end: float
    confidence: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "word": self.word,
            "normalized": self.normalized,
            "start": self.start,
            "end": self.end,
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "WordEntry":
        word = d["word"]
        normalized = d.get("normalized") or normalize_word(word)
        return cls(
            word=word,
            normalized=normalized,
            start=float(d["start"]),
            end=float(d["end"]),
            confidence=d.get("confidence"),
        )


@dataclass
class Transcript:
    words: List[WordEntry] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.words)

    def to_dict(self) -> Dict[str, Any]:
        return {"words": [w.to_dict() for w in self.words]}

    def to_json(self, indent: Optional[int] = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Transcript":
        return cls(words=[WordEntry.from_dict(w) for w in data.get("words", [])])

    @classmethod
    def from_json(cls, text: str) -> "Transcript":
        return cls.from_dict(json.loads(text))

    @classmethod
    def from_word_tuples(
        cls, tuples: Sequence[Tuple[str, float, float]]
    ) -> "Transcript":
        """Build a Transcript from (word, start, end[, confidence]) tuples.

        Primarily a fixture/test helper for constructing synthetic transcripts
        without going through an ASR engine or JSON.
        """
        words: List[WordEntry] = []
        for t in tuples:
            word, start, end = t[0], t[1], t[2]
            confidence = t[3] if len(t) > 3 else None
            words.append(
                WordEntry(
                    word=word,
                    normalized=normalize_word(word),
                    start=float(start),
                    end=float(end),
                    confidence=confidence,
                )
            )
        return cls(words=words)
