"""BM25 keyword search, written from scratch (no search library).

BM25 is the ranking formula behind classic search engines like Elasticsearch.
For each query word it rewards chunks that:
  * contain the word often   (term frequency, with diminishing returns via k1)
  * contain a RARE word      (inverse document frequency: "forklift" matters
                              more than "employee")
  * are not just long        (length normalisation via b)

score(chunk, query) = sum over query words w of
    idf(w) * tf * (k1 + 1) / (tf + k1 * (1 - b + b * len / avg_len))
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from .ingest import Chunk

STOPWORDS = frozenset(
    ("a an and are as at be by can do does for from has have how i if in is it its me my "
     "of on or our so than that the their them then there these this to us was we what "
     "when where which who why will with you your").split()
)


def _stem(word: str) -> str:
    """Tiny stemmer so different forms of a word match: 'days' -> 'day',
    'reimbursed' / 'reimburse' -> 'reimburs', 'expenses' / 'expense' -> 'expens'.

    Real stemmers (Porter, Snowball) handle many more cases. This one only strips
    plural and verb endings, because over-stemming hurts: an earlier version
    stripped "-ation" and turned "vacation" into "vac" (the unit tests caught it).
    """
    if len(word) <= 3:
        return word
    if word.endswith("ies") and len(word) > 4:
        word = word[:-3] + "y"
    elif word.endswith("s") and not word.endswith("ss"):
        word = word[:-1]
    for suffix in ("ing", "ed"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            word = word[: -len(suffix)]
            break
    if word.endswith("e") and len(word) > 4:
        word = word[:-1]
    return word


def tokenize(text: str, stem: bool = True) -> list[str]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    words = [w for w in words if w not in STOPWORDS]
    return [_stem(w) for w in words] if stem else words


@dataclass
class Hit:
    chunk: Chunk
    score: float


class BM25Index:
    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75, stem: bool = True):
        self.chunks = chunks
        self.k1, self.b, self.stem = k1, b, stem
        self._tf = [Counter(tokenize(c.text, stem)) for c in chunks]
        self._len = [sum(tf.values()) for tf in self._tf]
        self._avg_len = (sum(self._len) / len(self._len)) if chunks else 0.0
        doc_freq = Counter(word for tf in self._tf for word in tf)
        n = len(chunks)
        # "+1" inside the log keeps idf positive even for very common words.
        self._idf = {w: math.log(1 + (n - df + 0.5) / (df + 0.5)) for w, df in doc_freq.items()}

    def search(self, query: str, k: int = 5) -> list[Hit]:
        terms = tokenize(query, self.stem)
        scored = []
        for i, tf in enumerate(self._tf):
            score = 0.0
            for w in terms:
                if w not in tf:
                    continue
                f = tf[w]
                norm = self.k1 * (1 - self.b + self.b * self._len[i] / self._avg_len)
                score += self._idf[w] * f * (self.k1 + 1) / (f + norm)
            if score > 0:
                scored.append(Hit(self.chunks[i], score))
        scored.sort(key=lambda h: h.score, reverse=True)
        return scored[:k]

    # --- persistence: the index is rebuilt from chunks, so we only store chunks ---
    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps([c.to_dict() for c in self.chunks], indent=1), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path, **kwargs) -> BM25Index:
        path = Path(path)
        chunks = [Chunk(**d) for d in json.loads(path.read_text(encoding="utf-8"))] if path.exists() else []
        return cls(chunks, **kwargs)
