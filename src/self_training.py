"""Self-training utilities for zero-manual-label ranking adjustment.

The LLM judge produces relation labels, satisfaction scores, confidence, and a
short reason. This module treats those outputs as pseudo labels and tunes the
ranking weights so higher LLM-judged relation satisfaction moves toward the top.

No API calls happen here. The only inputs are scored candidate rows.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .rankers import RankingWeights


WEIGHT_NAMES = ("semantic", "graph", "relation", "citation", "novelty", "diversity")
FEATURE_COLUMNS = {
    "semantic": "semantic_score",
    "graph": "graph_score",
    "relation": "relation_score",
    "citation": "citation_score",
    "novelty": "novelty_score",
    "diversity": "diversity_score",
}
DEFAULT_WEIGHT_BOUNDS = {
    "semantic": (0.05, 0.60),
    "graph": (0.05, 0.45),
    "relation": (0.10, 0.55),
    "citation": (0.00, 0.25),
    "novelty": (0.00, 0.20),
    "diversity": (0.00, 0.20),
}


@dataclass
class SelfTrainingConfig:
    enabled: bool = False
    iterations: int = 2
    search_trials: int = 80
    target_k: int = 10
    reward_threshold: float = 0.55
    min_improvement: float = 0.001
    regularization: float = 0.02
    random_state: int = 172
    weight_bounds: Dict[str, Tuple[float, float]] = field(default_factory=lambda: dict(DEFAULT_WEIGHT_BOUNDS))

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, object]]) -> "SelfTrainingConfig":
        data = data or {}
        known = {k: data[k] for k in data if k in cls.__dataclass_fields__ and k != "weight_bounds"}
        cfg = cls(**known)
        raw_bounds = data.get("weight_bounds")
        if isinstance(raw_bounds, dict):
            bounds: Dict[str, Tuple[float, float]] = dict(DEFAULT_WEIGHT_BOUNDS)
            for name, value in raw_bounds.items():
                if name not in WEIGHT_NAMES:
                    continue
                try:
                    lo, hi = value  # type: ignore[misc]
                    bounds[name] = (float(lo), float(hi))
                except (TypeError, ValueError):
                    continue
            cfg.weight_bounds = bounds
        cfg.iterations = max(1, int(cfg.iterations))
        cfg.search_trials = max(1, int(cfg.search_trials))
        cfg.target_k = max(1, int(cfg.target_k))
        cfg.reward_threshold = float(np.clip(cfg.reward_threshold, 0.0, 1.0))
        cfg.min_improvement = max(0.0, float(cfg.min_improvement))
        cfg.regularization = max(0.0, float(cfg.regularization))
        return cfg


def weights_to_dict(weights: RankingWeights) -> Dict[str, float]:
    return {name: float(asdict(weights).get(name, 0.0)) for name in WEIGHT_NAMES}


def normalize_weights(weights: RankingWeights, bounds: Optional[Dict[str, Tuple[float, float]]] = None) -> RankingWeights:
    return _dict_to_weights(weights_to_dict(weights), bounds or DEFAULT_WEIGHT_BOUNDS)


def merge_relation_score_frames(existing: Optional[pd.DataFrame], new_scores: Optional[pd.DataFrame]) -> pd.DataFrame:
    """Merge relation-score tables and keep the newest score for each pair."""
    keys = ["target_paper_id", "candidate_paper_id"]
    frames = [df for df in [existing, new_scores] if df is not None and not df.empty]
    if not frames:
        return pd.DataFrame(columns=[
            "target_paper_id", "candidate_paper_id", "predicted_relation",
            "relation_confidence", "relation_satisfaction", "relation_reason",
        ])
    out = pd.concat(frames, ignore_index=True)
    return out.drop_duplicates(subset=keys, keep="last").reset_index(drop=True)


def add_judge_reward(ranked: pd.DataFrame, desired_relation: str) -> pd.DataFrame:
    """Add continuous pseudo-label reward columns derived from LLM judge output."""
    df = ranked.copy()
    if df.empty:
        df["judge_reward"] = []
        df["pseudo_relevance_grade"] = []
        return df

    confidence = _numeric_column(df, "relation_confidence", 0.0).fillna(0.0).clip(0.0, 1.0)
    satisfaction = _numeric_column(df, "relation_satisfaction", np.nan)
    relation_score = _numeric_column(df, "relation_score", np.nan)
    label_match = (
        df.get("predicted_relation", pd.Series([""] * len(df), index=df.index))
        .fillna("")
        .astype(str)
        .str.lower()
        .eq(desired_relation.lower())
        .astype(float)
    )

    fallback = relation_score.where(relation_score.notna(), label_match * confidence)
    satisfaction = satisfaction.where(satisfaction.notna(), fallback).fillna(0.0).clip(0.0, 1.0)

    # Confidence gates noisy pseudo labels without erasing useful low-confidence
    # weak negatives. A confident 0 stays 0; a low-confidence positive is softened.
    reward = satisfaction * (0.50 + 0.50 * confidence)
    df["judge_reward"] = reward.clip(0.0, 1.0)
    df["pseudo_relevance_grade"] = df["judge_reward"].map(reward_to_grade)
    return df


def reward_to_grade(value: object) -> int:
    try:
        score = float(value)
    except (TypeError, ValueError):
        score = 0.0
    if score >= 0.80:
        return 3
    if score >= 0.55:
        return 2
    if score >= 0.30:
        return 1
    return 0


def _numeric_column(df: pd.DataFrame, name: str, default: float) -> pd.Series:
    if name in df.columns:
        values = df[name]
    else:
        values = pd.Series([default] * len(df), index=df.index)
    return pd.to_numeric(values, errors="coerce")


def rank_with_weights(df: pd.DataFrame, weights: RankingWeights) -> pd.DataFrame:
    """Recompute final_score/rank from already-computed feature columns."""
    out = df.copy()
    for name, col in FEATURE_COLUMNS.items():
        if col not in out.columns:
            out[col] = 0.0
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)

    w = weights_to_dict(weights)
    out["final_score"] = sum(float(w[name]) * out[FEATURE_COLUMNS[name]] for name in WEIGHT_NAMES)
    out["rank"] = out.groupby("target_paper_id")["final_score"].rank(method="first", ascending=False).astype(int)
    return out.sort_values(["target_paper_id", "rank"]).reset_index(drop=True)


def evaluate_pseudo_ranking(
    ranked: pd.DataFrame,
    k_values: Iterable[int] = (10,),
    reward_threshold: float = 0.55,
) -> pd.DataFrame:
    """Evaluate ranking quality against LLM judge pseudo labels."""
    if ranked.empty:
        return pd.DataFrame()
    if "judge_reward" not in ranked.columns:
        raise ValueError("evaluate_pseudo_ranking expects a judge_reward column.")

    rows = []
    for k in k_values:
        per_target = []
        for target_id, group in ranked.groupby("target_paper_id"):
            group = group.sort_values("rank")
            rewards = pd.to_numeric(group["judge_reward"], errors="coerce").fillna(0.0).clip(0.0, 1.0)
            top = rewards.head(int(k)).to_numpy(dtype=float)
            binary_top = top >= reward_threshold
            all_binary = rewards.to_numpy(dtype=float) >= reward_threshold
            total_relevant = int(all_binary.sum())
            per_target.append({
                "target_paper_id": target_id,
                "precision": float(binary_top.mean()) if len(binary_top) else 0.0,
                "recall": float(binary_top.sum() / total_relevant) if total_relevant else 0.0,
                "mrr": _mrr(binary_top),
                "ndcg": _ndcg(top, int(k), rewards.to_numpy(dtype=float)),
                "avg_reward": float(top.mean()) if len(top) else 0.0,
            })
        per = pd.DataFrame(per_target)
        if per.empty:
            continue
        rows.append({
            "k": int(k),
            "PseudoPrecision@k": per["precision"].mean(),
            "PseudoRecall@k": per["recall"].mean(),
            "PseudoMRR@k": per["mrr"].mean(),
            "PseudoNDCG@k": per["ndcg"].mean(),
            "AvgJudgeReward@k": per["avg_reward"].mean(),
        })
    return pd.DataFrame(rows)


def tune_weights_from_judge(
    judged_candidates: pd.DataFrame,
    current_weights: RankingWeights,
    config: SelfTrainingConfig,
    desired_relation: Optional[str] = None,
) -> tuple[RankingWeights, pd.DataFrame]:
    """Search for weights that improve LLM-pseudo-label ranking metrics."""
    if judged_candidates.empty:
        return current_weights, pd.DataFrame()

    prepared = add_judge_reward(judged_candidates, desired_relation=desired_relation or _desired_relation(judged_candidates))
    current_weights = normalize_weights(current_weights, config.weight_bounds)
    rng = np.random.default_rng(config.random_state)

    candidates = _candidate_weight_dicts(current_weights, config, rng)
    current_metric = _objective(rank_with_weights(prepared, current_weights), config)
    rows = [{
        "candidate": -1,
        "raw_objective": current_metric,
        "objective": current_metric,
        "current_objective": current_metric,
        **{f"weight_{name}": weights_to_dict(current_weights)[name] for name in WEIGHT_NAMES},
    }]
    best_weights = current_weights
    best_objective = current_metric

    for i, mapping in enumerate(candidates):
        weights = _dict_to_weights(mapping, config.weight_bounds)
        ranked = rank_with_weights(prepared, weights)
        raw_objective = _objective(ranked, config)
        objective = raw_objective - config.regularization * _weight_distance(current_weights, weights)
        row = {
            "candidate": i,
            "raw_objective": raw_objective,
            "objective": objective,
            "current_objective": current_metric,
        }
        row.update({f"weight_{name}": weights_to_dict(weights)[name] for name in WEIGHT_NAMES})
        rows.append(row)
        if objective > best_objective:
            best_objective = objective
            best_weights = weights

    summary = pd.DataFrame(rows).sort_values("objective", ascending=False).reset_index(drop=True)
    return best_weights, summary


def build_pseudo_qrels(ranked: pd.DataFrame, desired_relation: str) -> pd.DataFrame:
    """Create an evaluation table based entirely on LLM judge pseudo labels."""
    df = add_judge_reward(ranked, desired_relation)
    rows = []
    for _, row in df.iterrows():
        target_id = row.get("target_paper_id", "")
        candidate_id = row.get("candidate_paper_id", "")
        desired = str(row.get("desired_relation", desired_relation) or desired_relation).lower()
        rows.append({
            "query_id": "q_" + _hash_id(target_id, desired, length=10),
            "target_paper_id": target_id,
            "desired_relation": desired,
            "candidate_paper_id": candidate_id,
            "pair_id": "p_" + _hash_id(target_id, desired, candidate_id, length=12),
            "pseudo_relation": str(row.get("predicted_relation", "") or "").lower(),
            "pseudo_relevance_grade": int(row.get("pseudo_relevance_grade", 0) or 0),
            "judge_reward": float(row.get("judge_reward", 0.0) or 0.0),
            "relation_satisfaction": row.get("relation_satisfaction", ""),
            "relation_confidence": row.get("relation_confidence", ""),
            "source": "llm_judge_pseudo_label",
        })
    return pd.DataFrame(rows)


def _desired_relation(df: pd.DataFrame) -> str:
    if "desired_relation" not in df.columns:
        return "extension"
    values = df["desired_relation"].dropna().astype(str).str.strip()
    values = values[values.ne("")]
    return values.iloc[0].lower() if not values.empty else "extension"


def _objective(ranked: pd.DataFrame, config: SelfTrainingConfig) -> float:
    metrics = evaluate_pseudo_ranking(
        ranked,
        k_values=[config.target_k],
        reward_threshold=config.reward_threshold,
    )
    if metrics.empty:
        return 0.0
    row = metrics.iloc[0]
    return float(
        0.70 * row["PseudoNDCG@k"]
        + 0.20 * row["PseudoMRR@k"]
        + 0.10 * row["AvgJudgeReward@k"]
    )


def _mrr(binary_relevant: Sequence[bool]) -> float:
    for i, rel in enumerate(binary_relevant, start=1):
        if bool(rel):
            return 1.0 / i
    return 0.0


def _ndcg(top_rewards: np.ndarray, k: int, all_rewards: np.ndarray) -> float:
    gains = np.asarray(top_rewards[:k], dtype=float)
    if gains.size == 0:
        return 0.0
    discounts = 1.0 / np.log2(np.arange(2, len(gains) + 2))
    dcg = float(np.sum(((2.0 ** gains) - 1.0) * discounts))
    ideal = np.sort(np.asarray(all_rewards, dtype=float))[::-1][:k]
    ideal_discounts = 1.0 / np.log2(np.arange(2, len(ideal) + 2))
    idcg = float(np.sum(((2.0 ** ideal) - 1.0) * ideal_discounts))
    return dcg / idcg if idcg > 0 else 0.0


def _candidate_weight_dicts(
    current_weights: RankingWeights,
    config: SelfTrainingConfig,
    rng: np.random.Generator,
) -> list[Dict[str, float]]:
    current = _project_weights(weights_to_dict(current_weights), config.weight_bounds)
    out = [current]

    for name in WEIGHT_NAMES:
        for multiplier in (0.70, 1.30):
            raw = dict(current)
            raw[name] = raw[name] * multiplier + (0.02 if multiplier > 1 else 0.0)
            out.append(_project_weights(raw, config.weight_bounds))

    base_vec = np.array([max(current[name], 1e-6) for name in WEIGHT_NAMES], dtype=float)
    alpha = np.maximum(base_vec * 45.0, 0.25)
    for _ in range(config.search_trials):
        if rng.random() < 0.80:
            vec = rng.dirichlet(alpha)
        else:
            vec = rng.dirichlet(np.ones(len(WEIGHT_NAMES)))
        out.append(_project_weights(dict(zip(WEIGHT_NAMES, vec)), config.weight_bounds))

    seen = set()
    unique = []
    for mapping in out:
        key = tuple(round(mapping[name], 4) for name in WEIGHT_NAMES)
        if key in seen:
            continue
        seen.add(key)
        unique.append(mapping)
    return unique


def _dict_to_weights(raw: Dict[str, float], bounds: Dict[str, Tuple[float, float]]) -> RankingWeights:
    return RankingWeights.from_dict(_project_weights(raw, bounds))


def _project_weights(raw: Dict[str, float], bounds: Dict[str, Tuple[float, float]]) -> Dict[str, float]:
    lows = {name: float(bounds.get(name, (0.0, 1.0))[0]) for name in WEIGHT_NAMES}
    highs = {name: float(bounds.get(name, (0.0, 1.0))[1]) for name in WEIGHT_NAMES}
    low_sum = sum(lows.values())
    high_sum = sum(highs.values())

    if low_sum > 1.0 or high_sum < 1.0:
        values = {name: max(0.0, float(raw.get(name, 0.0))) for name in WEIGHT_NAMES}
        total = sum(values.values()) or 1.0
        return {name: values[name] / total for name in WEIGHT_NAMES}

    weights = dict(lows)
    remaining = 1.0 - low_sum
    caps = {name: max(0.0, highs[name] - lows[name]) for name in WEIGHT_NAMES}
    prefs = {name: max(1e-9, float(raw.get(name, 0.0))) for name in WEIGHT_NAMES}
    free = set(WEIGHT_NAMES)

    while free and remaining > 1e-12:
        pref_sum = sum(prefs[name] for name in free)
        if pref_sum <= 0:
            share = {name: remaining / len(free) for name in free}
        else:
            share = {name: remaining * prefs[name] / pref_sum for name in free}

        capped = []
        used = 0.0
        for name in list(free):
            add = min(caps[name], share[name])
            weights[name] += add
            caps[name] -= add
            used += add
            if caps[name] <= 1e-12:
                capped.append(name)

        remaining -= used
        if not capped and used > 0:
            break
        for name in capped:
            free.discard(name)

    total = sum(weights.values()) or 1.0
    return {name: weights[name] / total for name in WEIGHT_NAMES}


def _weight_distance(a: RankingWeights, b: RankingWeights) -> float:
    av = weights_to_dict(a)
    bv = weights_to_dict(b)
    return float(np.sqrt(np.mean([(av[name] - bv[name]) ** 2 for name in WEIGHT_NAMES])))


def _hash_id(*parts: object, length: int = 12) -> str:
    raw = "|".join("" if p is None or pd.isna(p) else str(p) for p in parts)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:length]
