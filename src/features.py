from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class TextSimilarityModel:
    vectorizer: TfidfVectorizer

    @classmethod
    def fit(cls, corpus: Iterable[str], max_features: int = 50000, ngram_range: tuple[int, int] = (1, 2)) -> "TextSimilarityModel":
        vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=2,
        )
        vectorizer.fit(list(corpus))
        return cls(vectorizer=vectorizer)

    def pairwise_scores(self, left_texts: Iterable[str], right_texts: Iterable[str]) -> np.ndarray:
        left = self.vectorizer.transform([_safe_text(x) for x in left_texts])
        right = self.vectorizer.transform([_safe_text(x) for x in right_texts])
        scores = np.asarray(left.multiply(right).sum(axis=1)).ravel()
        # Because TF-IDF vectors are L2-normalized by default, dot product equals cosine similarity.
        return scores

    def query_to_candidates(self, query_text: str, candidate_texts: Iterable[str]) -> np.ndarray:
        q = self.vectorizer.transform([_safe_text(query_text)])
        c = self.vectorizer.transform([_safe_text(x) for x in candidate_texts])
        return cosine_similarity(q, c).ravel()


def _safe_text(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value)


def minmax_normalize(values: Iterable[float]) -> np.ndarray:
    arr = np.asarray(list(values), dtype=float)
    if len(arr) == 0:
        return arr
    finite = np.isfinite(arr)
    if not finite.any():
        return np.zeros_like(arr)
    mn = arr[finite].min()
    mx = arr[finite].max()
    out = np.zeros_like(arr)
    if mx == mn:
        out[finite] = 0.0
    else:
        out[finite] = (arr[finite] - mn) / (mx - mn)
    return out


def log_normalize_count(values: Iterable[float]) -> np.ndarray:
    arr = np.asarray(list(values), dtype=float)
    arr = np.where(np.isfinite(arr), arr, 0)
    arr = np.maximum(arr, 0)
    return minmax_normalize(np.log1p(arr))


def graph_score_from_hop(hops: Iterable[float]) -> np.ndarray:
    arr = np.asarray(list(hops), dtype=float)
    arr = np.where(np.isfinite(arr), arr, 999)
    arr = np.maximum(arr, 1)
    scores = 1.0 / arr
    scores[arr >= 999] = 0.0
    return scores


def mmr_diversity_scores(
    relevance_scores: np.ndarray,
    candidate_texts: list[str],
    vectorizer: TfidfVectorizer,
    lambda_: float = 0.75,
) -> np.ndarray:
    """Return approximate diversity contribution based on MMR selection order.

    The score is higher for candidates selected earlier by MMR. This lets us blend
    diversity into the final ranking without changing the output format.
    """
    n = len(candidate_texts)
    if n == 0:
        return np.array([])
    if n == 1:
        return np.array([1.0])

    x = vectorizer.transform(candidate_texts)
    sim = cosine_similarity(x)
    selected: list[int] = []
    remaining = set(range(n))
    order_score = np.zeros(n, dtype=float)

    for rank in range(n):
        best_idx = None
        best_score = -np.inf
        for i in remaining:
            if not selected:
                div_penalty = 0.0
            else:
                div_penalty = max(sim[i, j] for j in selected)
            mmr = lambda_ * relevance_scores[i] - (1 - lambda_) * div_penalty
            if mmr > best_score:
                best_score = mmr
                best_idx = i
        if best_idx is None:
            break
        selected.append(best_idx)
        remaining.remove(best_idx)
        order_score[best_idx] = 1.0 / (rank + 1)

    return minmax_normalize(order_score)
