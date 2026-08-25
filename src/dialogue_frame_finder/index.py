"""In-memory inverted transcript index.

Maps normalized words to their transcript-position postings. This is the
primary retrieval structure: it is a lightweight dict-of-lists built once per
transcript, not a database. Word frequency (postings-list length) doubles as
the transcript-local rarity signal used for anchor selection.
"""

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Sequence

from .normalization import normalize_word
from .transcript import Transcript, WordEntry


@dataclass
class TranscriptIndex:
    transcript: Transcript
    postings: Dict[str, List[int]]

    @classmethod
    def build(cls, transcript: Transcript) -> "TranscriptIndex":
        postings: Dict[str, List[int]] = defaultdict(list)
        for position, entry in enumerate(transcript.words):
            if entry.normalized:
                postings[entry.normalized].append(position)
        return cls(transcript=transcript, postings=dict(postings))

    def frequency(self, word: str) -> int:
        """Transcript-local occurrence count for a word (0 if never seen)."""
        return len(self.postings.get(normalize_word(word), ()))

    def positions(self, word: str) -> List[int]:
        """All transcript positions where this word occurs, in ascending order."""
        return list(self.postings.get(normalize_word(word), ()))

    def word_entry(self, position: int) -> WordEntry:
        return self.transcript.words[position]

    @property
    def vocabulary(self):
        return self.postings.keys()

    def ngram_positions(self, ngram_words: Sequence[str]) -> List[int]:
        """Positions where the given sequence of already-normalized words
        occurs contiguously in the transcript. Empty input yields no matches."""
        if not ngram_words:
            return []
        n = len(self.transcript.words)
        matches: List[int] = []
        for pos in self.positions(ngram_words[0]):
            ok = True
            for offset in range(1, len(ngram_words)):
                p = pos + offset
                if p >= n or self.transcript.words[p].normalized != ngram_words[offset]:
                    ok = False
                    break
            if ok:
                matches.append(pos)
        return matches
