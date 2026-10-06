"""Deterministic Okapi BM25, independent of datasets and external services."""
from collections import Counter
from math import isfinite, log
import re
import unicodedata


def tokenize(text: str) -> tuple[str, ...]:
    return tuple(re.findall(r"[^\W_]+", unicodedata.normalize("NFKC", text).casefold()))


class BM25Index:
    def __init__(self, texts, *, k1: float = 1.5, b: float = .75):
        if not isfinite(k1) or k1 <= 0 or not isfinite(b) or not 0 <= b <= 1:
            raise ValueError("Invalid BM25 parameters")
        self.k1, self.b = k1, b
        tokens = tuple(tokenize(text) for text in texts)
        self.frequencies = tuple(Counter(row) for row in tokens)
        self.lengths = tuple(len(row) for row in tokens)
        self.count = len(tokens)
        self.average_length = sum(self.lengths) / self.count if self.count else 0
        self.document_frequency = Counter(token for row in self.frequencies for token in row)

    def scores(self, query: str) -> tuple[float, ...]:
        terms = sorted(set(tokenize(query)))
        scores = []
        for frequency, length in zip(self.frequencies, self.lengths):
            score = 0.0
            for term in terms:
                tf = frequency[term]
                if not tf:
                    continue
                df = self.document_frequency[term]
                idf = log(1 + (self.count - df + .5) / (df + .5))
                norm = 1 - self.b + self.b * length / self.average_length
                score += idf * tf * (self.k1 + 1) / (tf + self.k1 * norm)
            scores.append(score)
        return tuple(scores)
