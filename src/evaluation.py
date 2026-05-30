from __future__ import annotations

from typing import Iterable, List, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_recall_fscore_support


def _is_relevant(label: object, desired_relation: str | None = None, positive_labels: Sequence[str] | None = None) -> bool:
    if label is None or pd.isna(label):
        return False
    s = str(label).strip().lower()
    if desired_relation:
        return s == desired_relation.lower()
    if positive_labels is None:
        positive_labels = ["critique", "extension", "application"]
    return s in {x.lower() for x in positive_labels}


def precision_at_k(labels: Sequence[object], k: int, desired_relation: str | None = None, positive_labels: Sequence[str] | None = None) -> float:
    top = list(labels)[:k]
    if not top:
        return 0.0
    return float(np.mean([_is_relevant(x, desired_relation, positive_labels) for x in top]))


def recall_at_k(labels: Sequence[object], k: int, total_relevant: int, desired_relation: str | None = None, positive_labels: Sequence[str] | None = None) -> float:
    if total_relevant <= 0:
        return 0.0
    top = list(labels)[:k]
    hits = sum(_is_relevant(x, desired_relation, positive_labels) for x in top)
    return float(hits / total_relevant)


def hit_rate_at_k(labels: Sequence[object], k: int, desired_relation: str | None = None, positive_labels: Sequence[str] | None = None) -> float:
    top = list(labels)[:k]
    return float(any(_is_relevant(x, desired_relation, positive_labels) for x in top))


def mrr_at_k(labels: Sequence[object], k: int, desired_relation: str | None = None, positive_labels: Sequence[str] | None = None) -> float:
    for i, label in enumerate(list(labels)[:k], start=1):
        if _is_relevant(label, desired_relation, positive_labels):
            return float(1.0 / i)
    return 0.0


def ndcg_at_k(labels: Sequence[object], k: int, desired_relation: str | None = None, positive_labels: Sequence[str] | None = None) -> float:
    rel = np.array([1.0 if _is_relevant(x, desired_relation, positive_labels) else 0.0 for x in list(labels)[:k]])
    if len(rel) == 0:
        return 0.0
    discounts = 1.0 / np.log2(np.arange(2, len(rel) + 2))
    dcg = float(np.sum(rel * discounts))
    ideal = np.sort(rel)[::-1]
    idcg = float(np.sum(ideal * discounts))
    return dcg / idcg if idcg > 0 else 0.0


def path_support_rate(ranked: pd.DataFrame, k: int) -> float:
    if ranked.empty:
        return 0.0
    vals = []
    for _, group in ranked.groupby("target_paper_id"):
        top = group.sort_values("rank").head(k)
        vals.extend((top["citation_path"].fillna("").astype(str).str.len() > 0).tolist())
    return float(np.mean(vals)) if vals else 0.0


def evidence_support_precision(ranked: pd.DataFrame, k: int) -> float:
    """Approximate evidence support from available fields.

    A recommendation is counted as evidence-supported if its relation reason is
    non-empty and it has a title/abstract. For the report, manually verify a
    subset and replace/augment this with human labels if possible.
    """
    if ranked.empty:
        return 0.0
    vals = []
    for _, group in ranked.groupby("target_paper_id"):
        top = group.sort_values("rank").head(k)
        has_reason = top.get("relation_reason", pd.Series([""] * len(top))).fillna("").astype(str).str.len() > 0
        has_text = top["candidate_abstract"].fillna("").astype(str).str.len() > 0
        vals.extend((has_reason & has_text).tolist())
    return float(np.mean(vals)) if vals else 0.0


def evaluate_ranking(
    ranked: pd.DataFrame,
    k_values: Iterable[int] = (5, 10, 20),
    desired_relation: str | None = None,
    positive_labels: Sequence[str] | None = None,
) -> pd.DataFrame:
    if "label" not in ranked.columns:
        raise ValueError("Ranked dataframe must include a `label` column for evaluation.")

    rows = []
    for k in k_values:
        per_target = []
        for target_id, group in ranked.groupby("target_paper_id"):
            group = group.sort_values("rank")
            labels = group["label"].tolist()
            total_rel = sum(_is_relevant(x, desired_relation, positive_labels) for x in labels)
            per_target.append({
                "target_paper_id": target_id,
                "precision": precision_at_k(labels, k, desired_relation, positive_labels),
                "recall": recall_at_k(labels, k, total_rel, desired_relation, positive_labels),
                "hit_rate": hit_rate_at_k(labels, k, desired_relation, positive_labels),
                "mrr": mrr_at_k(labels, k, desired_relation, positive_labels),
                "ndcg": ndcg_at_k(labels, k, desired_relation, positive_labels),
            })
        per = pd.DataFrame(per_target)
        if per.empty:
            continue
        rows.append({
            "k": k,
            "Precision@k": per["precision"].mean(),
            "Recall@k": per["recall"].mean(),
            "HitRate@k": per["hit_rate"].mean(),
            "MRR@k": per["mrr"].mean(),
            "nDCG@k": per["ndcg"].mean(),
            "PathSupportRate@k": path_support_rate(ranked, k),
            "EvidenceSupportPrecision@k": evidence_support_precision(ranked, k),
        })
    return pd.DataFrame(rows)


def evaluate_relation_classification(df: pd.DataFrame) -> pd.DataFrame:
    """Evaluate predicted relation labels if gold labels are available."""
    if "label" not in df.columns or "predicted_relation" not in df.columns:
        return pd.DataFrame()
    valid = df[df["label"].fillna("").astype(str).str.len() > 0].copy()
    valid = valid[valid["label"].astype(str).str.lower() != "unlabeled"]
    if valid.empty:
        return pd.DataFrame()
    y_true = valid["label"].astype(str).str.lower()
    y_pred = valid["predicted_relation"].fillna("unrelated").astype(str).str.lower()
    labels = sorted(set(y_true) | set(y_pred))
    macro_f1 = f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    rows = [{"label": "macro_avg", "precision": None, "recall": None, "f1": macro_f1, "support": len(valid)}]
    rows.extend({"label": lab, "precision": p, "recall": r, "f1": f, "support": s} for lab, p, r, f, s in zip(labels, precision, recall, f1, support))
    return pd.DataFrame(rows)
