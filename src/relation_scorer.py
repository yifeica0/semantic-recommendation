"""Relation scoring: decide how a candidate paper relates to a target paper.

The pipeline requires the LLM judge. It reads the target/candidate
titles+abstracts and the citation-path evidence and returns a label, a
continuous *satisfaction* score for the desired relation, a confidence, and a
short citation-grounded reason.

Output columns (per target/candidate pair):
    target_paper_id, candidate_paper_id, predicted_relation,
    relation_confidence, relation_satisfaction, relation_reason
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import List, Optional, Sequence

import pandas as pd

from .llm_client import LLMClient


RELATION_LABELS = ["critique", "extension", "application", "background", "unrelated"]

RELATION_DEFINITIONS = {
    "critique": "questions, challenges, identifies limitations/errors/biases of, or fails to reproduce the target",
    "extension": "builds upon, improves, generalizes, or proposes a new variant of the target's method or ideas",
    "application": "uses or applies the target's method, model, dataset, or tool to a concrete task or domain",
    "background": "is foundational or contextual to the target (survey, prior art, definitions); no direct improvement or critique",
    "unrelated": "has no meaningful scientific relationship to the target",
}


@dataclass
class RelationPrediction:
    label: str
    confidence: float
    reason: str
    satisfaction: Optional[float] = None  # 0..1 strength of the *desired* relation


# --------------------------------------------------------------------------- #
# LLM prompt + parsing                                                        #
# --------------------------------------------------------------------------- #
def _truncate(text: object, limit: int = 1600) -> str:
    s = "" if text is None else str(text)
    s = s.strip()
    return s if len(s) <= limit else s[:limit].rsplit(" ", 1)[0] + " ..."


def build_relation_messages(row: pd.Series, desired_relation: str) -> list[dict]:
    """Build the constrained relation-scoring chat messages for one pair."""
    defs = "\n".join(f"- {k}: {v}" for k, v in RELATION_DEFINITIONS.items())
    edge_type = row.get("edge_type", "")
    hop = row.get("hop", "")
    path = row.get("citation_path", "")

    system = (
        "You are a precise scientific citation-relationship classifier. Decide how a CANDIDATE paper "
        "relates to a TARGET paper, using ONLY the provided titles, abstracts, and citation-path "
        "evidence. Be conservative: if the evidence is weak or generic, prefer 'background' or "
        "'unrelated'. Judge the relationship from the candidate's perspective toward the target."
    )
    user = f"""Relation labels (choose exactly one):
{defs}

DESIRED relation for this query: "{desired_relation}"

TARGET paper
  title: {_truncate(row.get('target_title', ''), 400)}
  abstract: {_truncate(row.get('target_abstract', ''))}

CANDIDATE paper
  title: {_truncate(row.get('candidate_title', ''), 400)}
  abstract: {_truncate(row.get('candidate_abstract', ''))}

Citation-graph evidence
  edge_type: {edge_type}
  hops_between: {hop}
  citation_path: {path}

Tasks:
1. Pick the single best label describing how the CANDIDATE relates to the TARGET.
2. Rate, from 0.0 to 1.0, how strongly the CANDIDATE "{desired_relation}" the TARGET specifically.
3. Give your confidence 0.0-1.0 and a <=30 word reason citing concrete evidence from the abstracts or path.

Respond with ONE JSON object and nothing after it:
{{"label": "critique|extension|application|background|unrelated", "desired_relation_satisfied": 0.0, "confidence": 0.0, "reason": "..."}}"""

    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def build_prompt(row: pd.Series) -> str:
    """Render one relation-scoring prompt for inspection/debugging."""
    msgs = build_relation_messages(row, desired_relation="extension")
    return msgs[0]["content"] + "\n\n" + msgs[1]["content"]


def _extract_json(text: str) -> Optional[dict]:
    """Pull the last balanced JSON object out of a model response."""
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Scan for balanced { ... } blocks; keep the last parseable one.
    best = None
    depth = 0
    start = -1
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start >= 0:
                    chunk = text[start:i + 1]
                    try:
                        best = json.loads(chunk)
                    except json.JSONDecodeError:
                        pass
    return best


def parse_relation_response(text: str) -> RelationPrediction:
    """Parse a JSON relation judgement from an LLM response."""
    data = _extract_json(text)
    if data is None:
        return RelationPrediction("unrelated", 0.0, "Could not parse model response.", 0.0)

    label = str(data.get("label", "unrelated")).strip().lower()
    if label not in RELATION_LABELS:
        label = "unrelated"

    def _f(key: str) -> Optional[float]:
        try:
            return max(0.0, min(1.0, float(data.get(key))))
        except (TypeError, ValueError):
            return None

    confidence = _f("confidence")
    satisfaction = _f("desired_relation_satisfied")
    reason = str(data.get("reason", "")).strip()
    return RelationPrediction(label, confidence if confidence is not None else 0.0, reason, satisfaction)


# --------------------------------------------------------------------------- #
# Scoring entry points                                                        #
# --------------------------------------------------------------------------- #
def _rows_to_frame(rows: list[dict]) -> pd.DataFrame:
    cols = [
        "target_paper_id", "candidate_paper_id", "predicted_relation",
        "relation_confidence", "relation_satisfaction", "relation_reason",
    ]
    df = pd.DataFrame(rows)
    for c in cols:
        if c not in df.columns:
            df[c] = None
    return df[cols]


def score_pairs_llm(
    pairs: pd.DataFrame,
    client: LLMClient,
    desired_relation: str = "extension",
    model: Optional[str] = None,
) -> pd.DataFrame:
    """Score every pair with the real LLM (concurrent + cached)."""
    if pairs.empty:
        return _rows_to_frame([])

    model = model or client.config.scorer_model
    rows_list = [pd.Series(r._asdict()) for r in pairs.itertuples(index=False)]
    batch = [build_relation_messages(s, desired_relation) for s in rows_list]
    responses = client.complete_many(batch, model=model, desc=f"LLM relation [{desired_relation}]")

    rows = []
    for s, text in zip(rows_list, responses):
        pred = parse_relation_response(text)
        rows.append({
            "target_paper_id": s.get("target_paper_id"),
            "candidate_paper_id": s.get("candidate_paper_id"),
            "predicted_relation": pred.label,
            "relation_confidence": pred.confidence,
            "relation_satisfaction": pred.satisfaction,
            "relation_reason": pred.reason,
        })
    return _rows_to_frame(rows)


def score_pairs(
    pairs: pd.DataFrame,
    desired_relation: str = "extension",
    client: Optional[LLMClient] = None,
    model: Optional[str] = None,
) -> pd.DataFrame:
    """Score every pair with the real LLM."""
    if client is None or not client.available:
        raise RuntimeError("DeepSeek-backed relation scoring requires a configured API key.")
    return score_pairs_llm(pairs, client, desired_relation=desired_relation, model=model)


def merge_relation_scores(
    pairs: pd.DataFrame,
    relation_scores: Optional[pd.DataFrame],
    desired_relation: str,
) -> pd.DataFrame:
    """Attach relation columns and derive a continuous ``relation_score`` in [0,1].

    Prefers the LLM's continuous ``relation_satisfaction`` for the desired
    relation; falls back to (label == desired) * confidence when satisfaction is
    unavailable (e.g. legacy heuristic output).
    """
    df = pairs.copy()
    if relation_scores is not None and not relation_scores.empty:
        keys = ["target_paper_id", "candidate_paper_id"]
        extra = [c for c in relation_scores.columns if c not in keys and c not in df.columns]
        df = df.merge(relation_scores[keys + extra], on=keys, how="left")

    if "predicted_relation" not in df.columns:
        df["predicted_relation"] = ""
    if "relation_confidence" not in df.columns:
        df["relation_confidence"] = 0.0
    if "relation_satisfaction" not in df.columns:
        df["relation_satisfaction"] = pd.NA

    df["relation_confidence"] = pd.to_numeric(df["relation_confidence"], errors="coerce").fillna(0.0)
    sat = pd.to_numeric(df["relation_satisfaction"], errors="coerce")
    label_match = (df["predicted_relation"].astype(str).str.lower() == desired_relation.lower()).astype(float)
    fallback = label_match * df["relation_confidence"]
    df["relation_score"] = sat.where(sat.notna(), fallback).astype(float)
    return df
