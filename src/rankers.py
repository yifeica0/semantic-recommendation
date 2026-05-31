"""Two-stage relation-aware ranking.

Stage 1 (cheap, all candidates): TF-IDF semantic similarity + citation-graph
proximity + citation prominence + novelty produce a ``retrieval_score`` used to
prefilter the top-K candidates per target.

Stage 2 (expensive, survivors only): the real LLM relation scorer judges those
top-K, and the final score blends semantic + graph + relation-satisfaction +
citation + novelty + diversity. Because the LLM only ever sees the K survivors,
cost scales with ``targets * top_k`` instead of the full candidate pool.

The split lets the pipeline insert the LLM call between the two stages::

    survivors, text_model = retrieve_candidates(pairs, papers, weights, top_k)
    relation_scores = score_pairs(survivors, desired_relation, client)
    ranked = finalize_ranking(survivors, relation_scores, desired_relation, weights)
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from .features import (
    TextSimilarityModel,
    graph_score_from_hop,
    log_normalize_count,
    minmax_normalize,
    mmr_diversity_scores,
)
from .relation_scorer import merge_relation_scores


@dataclass
class RankingWeights:
    semantic: float = 0.30
    graph: float = 0.20
    relation: float = 0.30
    citation: float = 0.08
    novelty: float = 0.05
    diversity: float = 0.07

    @classmethod
    def from_dict(cls, d: Dict[str, float]) -> "RankingWeights":
        return cls(**{k: float(v) for k, v in (d or {}).items() if hasattr(cls, k)})

    def retrieval_part(self) -> Dict[str, float]:
        """Non-relation weights, used for the stage-1 prefilter."""
        return {"semantic": self.semantic, "graph": self.graph,
                "citation": self.citation, "novelty": self.novelty}


# --------------------------------------------------------------------------- #
# Stage 1: features + retrieval prefilter                                     #
# --------------------------------------------------------------------------- #
def compute_features(
    pairs: pd.DataFrame, papers: pd.DataFrame, full_corpus: bool = True
) -> Tuple[pd.DataFrame, TextSimilarityModel]:
    """Attach semantic / graph / citation / novelty features to every pair.

    ``full_corpus=True`` fits the TF-IDF vocabulary/IDF on every paper (stable
    statistics, but slow on a large corpus). ``full_corpus=False`` fits only on
    the texts in ``pairs`` — far faster for single-query inference, at the cost of
    IDF estimated from just the candidate set.
    """
    df = pairs.copy()
    df["target_text"] = (df["target_title"].fillna("") + ". " + df["target_abstract"].fillna("")).str.strip()
    df["candidate_text"] = (df["candidate_title"].fillna("") + ". " + df["candidate_abstract"].fillna("")).str.strip()

    fit_texts = [df["target_text"], df["candidate_text"]]
    if full_corpus:
        fit_texts.insert(0, papers.get("text_for_retrieval", papers["title"].fillna("") + ". " + papers["abstract"].fillna("")))
    corpus = pd.concat(fit_texts, ignore_index=True).fillna("").astype(str)
    text_model = TextSimilarityModel.fit(corpus)

    df["semantic_score"] = text_model.pairwise_scores(df["target_text"], df["candidate_text"])
    df["graph_score"] = graph_score_from_hop(df["hop"])
    df["citation_score"] = log_normalize_count(df.get("candidate_citationCount", pd.Series([0] * len(df))))
    df["novelty_score"] = _novelty_score(df)
    return df, text_model


def _novelty_score(df: pd.DataFrame) -> np.ndarray:
    """Favor recent and not-yet-saturated papers (optional novelty term)."""
    years = pd.to_numeric(df.get("candidate_year", pd.Series([np.nan] * len(df))), errors="coerce")
    recency = minmax_normalize(years.fillna(years.median() if years.notna().any() else 0).to_numpy(dtype=float))
    citation_saturation = log_normalize_count(df.get("candidate_citationCount", pd.Series([0] * len(df))))
    return 0.5 * recency + 0.5 * (1.0 - np.asarray(citation_saturation, dtype=float))


def retrieval_score(df: pd.DataFrame, weights: RankingWeights) -> pd.Series:
    parts = weights.retrieval_part()
    total = sum(parts.values()) or 1.0
    score = (
        parts["semantic"] * df["semantic_score"].astype(float)
        + parts["graph"] * df["graph_score"].astype(float)
        + parts["citation"] * df["citation_score"].astype(float)
        + parts["novelty"] * df["novelty_score"].astype(float)
    ) / total
    return score


def retrieve_candidates(
    pairs: pd.DataFrame,
    papers: pd.DataFrame,
    weights: Optional[RankingWeights] = None,
    top_k: Optional[int] = None,
    full_corpus: bool = True,
) -> Tuple[pd.DataFrame, TextSimilarityModel]:
    """Stage 1: compute features and keep the top-K candidates per target.

    ``top_k=None`` keeps everything (no prefilter). Returns the survivors plus
    the fitted text model (so stage 2 can reuse its vocabulary for diversity).
    ``full_corpus=False`` fits TF-IDF only on the pair texts (fast single-query path).
    """
    weights = weights or RankingWeights()
    if pairs.empty:
        return pairs.copy(), TextSimilarityModel.fit([""])

    df, text_model = compute_features(pairs, papers, full_corpus=full_corpus)
    df["retrieval_score"] = retrieval_score(df, weights)
    df["retrieval_rank"] = (
        df.groupby("target_paper_id")["retrieval_score"].rank(method="first", ascending=False).astype(int)
    )
    if top_k is not None:
        df = df[df["retrieval_rank"] <= int(top_k)].copy()
    return df.reset_index(drop=True), text_model


# --------------------------------------------------------------------------- #
# Stage 2: relation rerank + final blend                                      #
# --------------------------------------------------------------------------- #
def _add_diversity(df: pd.DataFrame, text_model: Optional[TextSimilarityModel], mmr_lambda: float) -> pd.DataFrame:
    df = df.copy()
    df["diversity_score"] = 0.0
    if text_model is None:
        text_model = TextSimilarityModel.fit(df["candidate_text"].fillna("").astype(str).tolist() or [""])
    for target_id, idx in df.groupby("target_paper_id").groups.items():
        sub = df.loc[idx]
        div = mmr_diversity_scores(
            relevance_scores=sub["semantic_score"].to_numpy(dtype=float),
            candidate_texts=sub["candidate_text"].fillna("").astype(str).tolist(),
            vectorizer=text_model.vectorizer,
            lambda_=mmr_lambda,
        )
        df.loc[idx, "diversity_score"] = div
    return df


def finalize_ranking(
    df: pd.DataFrame,
    relation_scores: Optional[pd.DataFrame] = None,
    desired_relation: str = "extension",
    weights: Optional[RankingWeights] = None,
    mmr_lambda: float = 0.75,
    text_model: Optional[TextSimilarityModel] = None,
) -> pd.DataFrame:
    """Stage 2: merge relation scores, add diversity, blend, and rank."""
    weights = weights or RankingWeights()
    if df.empty:
        return df.copy()
    if "semantic_score" not in df.columns:
        raise ValueError("finalize_ranking expects feature columns; run retrieve_candidates / compute_features first.")

    df = merge_relation_scores(df, relation_scores=relation_scores, desired_relation=desired_relation)
    df = _add_diversity(df, text_model, mmr_lambda)

    df["final_score"] = (
        weights.semantic * df["semantic_score"].astype(float)
        + weights.graph * df["graph_score"].astype(float)
        + weights.relation * df["relation_score"].astype(float)
        + weights.citation * df["citation_score"].astype(float)
        + weights.novelty * df["novelty_score"].astype(float)
        + weights.diversity * df["diversity_score"].astype(float)
    )
    df["rank"] = df.groupby("target_paper_id")["final_score"].rank(method="first", ascending=False).astype(int)
    df = df.sort_values(["target_paper_id", "rank"]).reset_index(drop=True)
    return df


def rank_candidate_pairs(
    pairs: pd.DataFrame,
    papers: pd.DataFrame,
    relation_scores: Optional[pd.DataFrame] = None,
    desired_relation: str = "extension",
    weights: RankingWeights | None = None,
    mmr_lambda: float = 0.75,
    top_k: Optional[int] = None,
) -> pd.DataFrame:
    """Convenience: run both stages when relation scores are already available.

    (The pipeline uses ``retrieve_candidates`` + ``finalize_ranking`` directly so
    it can insert the LLM call in between; this helper covers the simple path and
    the ablation baselines.)
    """
    weights = weights or RankingWeights()
    survivors, text_model = retrieve_candidates(pairs, papers, weights=weights, top_k=top_k)
    return finalize_ranking(
        survivors, relation_scores=relation_scores, desired_relation=desired_relation,
        weights=weights, mmr_lambda=mmr_lambda, text_model=text_model,
    )


def make_baseline_rankings(pairs: pd.DataFrame, papers: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Separate baseline rankings for ablation studies."""
    baselines = {}
    for name, weights in {
        "semantic_only": RankingWeights(semantic=1, graph=0, relation=0, citation=0, novelty=0, diversity=0),
        "graph_only": RankingWeights(semantic=0, graph=1, relation=0, citation=0, novelty=0, diversity=0),
        "citation_only": RankingWeights(semantic=0, graph=0, relation=0, citation=1, novelty=0, diversity=0),
        "semantic_graph": RankingWeights(semantic=0.6, graph=0.4, relation=0, citation=0, novelty=0, diversity=0),
    }.items():
        ranked = rank_candidate_pairs(pairs, papers, relation_scores=None, desired_relation="extension", weights=weights)
        ranked["baseline"] = name
        baselines[name] = ranked
    return baselines
