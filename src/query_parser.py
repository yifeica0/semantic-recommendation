"""Query understanding: natural language -> structured, resolved query.

Turns a request like::

    "find recent papers that extend ChimeraX for cryo-EM"

into a structured object::

    ParsedQuery(target_paper_id=<resolved id>, target_title="UCSF ChimeraX ...",
                desired_relation="extension", filters={"year_min": 2022, ...},
                top_k=20, match_score=0.71, alternatives=[...])

The LLM extracts {target_reference, desired_relation, filters, top_k}. We then
resolve ``target_reference`` to an actual ``paperId`` in the corpus: a 40-hex
Semantic Scholar id is used directly; otherwise we TF-IDF match the reference
against paper titles. The pipeline requires the LLM parser and does not use a
deterministic fallback.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

from .llm_client import LLMClient
from .relation_scorer import _extract_json

# Map free-text intent to the constrained relation vocabulary.
RELATION_SYNONYMS = {
    "critique": "critique", "critiques": "critique", "criticize": "critique",
    "criticizes": "critique", "challenge": "critique", "challenges": "critique",
    "limitation": "critique", "limitations": "critique", "refute": "critique",
    "rebut": "critique", "dispute": "critique", "contradict": "critique",
    "extend": "extension", "extends": "extension", "extension": "extension",
    "improve": "extension", "improves": "extension", "build": "extension",
    "builds": "extension", "buildson": "extension", "generalize": "extension",
    "enhance": "extension", "advance": "extension", "successor": "extension",
    "follow-up": "extension", "followup": "extension",
    "apply": "application", "applies": "application", "application": "application",
    "applications": "application", "use": "application", "uses": "application",
    "using": "application", "deploy": "application", "case study": "application",
    "background": "background", "survey": "background", "foundational": "background",
    "review": "background", "overview": "background", "related": "background",
}
VALID_RELATIONS = ["critique", "extension", "application", "background"]
_HEX_ID = re.compile(r"^[0-9a-f]{40}$")


@dataclass
class ParsedQuery:
    raw_query: str
    target_reference: str
    desired_relation: str
    target_paper_id: Optional[str] = None
    target_title: Optional[str] = None
    match_score: float = 0.0
    top_k: int = 20
    filters: Dict[str, Any] = field(default_factory=dict)
    alternatives: List[Dict[str, Any]] = field(default_factory=list)
    parser_backend: str = "heuristic"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------------------- #
# Step 1: extract structured fields from the query                            #
# --------------------------------------------------------------------------- #
def _normalize_relation(value: object, default: str = "extension") -> str:
    s = re.sub(r"[^a-z\- ]", "", str(value or "").lower()).strip()
    if s in RELATION_SYNONYMS:
        return RELATION_SYNONYMS[s]
    for token in s.split():
        if token in RELATION_SYNONYMS:
            return RELATION_SYNONYMS[token]
    return default if default in VALID_RELATIONS else "extension"


def parse_query_llm(query: str, client: LLMClient) -> Dict[str, Any]:
    system = (
        "You convert a user's scientific-paper discovery request into a structured JSON query. "
        "Identify the target paper the user is asking ABOUT, and the relationship the user wants the "
        "recommended papers to have toward that target."
    )
    user = f"""User request:
{query}

Allowed desired_relation values: critique, extension, application, background.
(extension = builds on/improves; critique = challenges/limitations; application = uses/applies it;
 background = foundational/survey.)

Return ONE JSON object and nothing after it:
{{"target_reference": "the paper title or id the user is asking about",
  "desired_relation": "critique|extension|application|background",
  "filters": {{"year_min": null, "year_max": null, "venue": null, "fields_of_study": []}},
  "top_k": 20}}"""
    text = client.complete(
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        model=client.config.parser_model,
    )
    data = _extract_json(text) or {}
    return {
        "target_reference": str(data.get("target_reference", "")).strip(),
        "desired_relation": _normalize_relation(data.get("desired_relation"), default="extension"),
        "filters": data.get("filters") or {},
        "top_k": int(data.get("top_k") or 20),
    }


# --------------------------------------------------------------------------- #
# Step 2: resolve the target reference to a real paperId                       #
# --------------------------------------------------------------------------- #
def resolve_target(
    reference: str,
    papers: pd.DataFrame,
    top_n: int = 5,
) -> Dict[str, Any]:
    """Resolve a free-text reference (or id) to a paper in the corpus."""
    ref = (reference or "").strip()
    ids = set(papers["paperId"].astype(str))

    if _HEX_ID.match(ref.lower()) and ref.lower() in ids:
        row = papers[papers["paperId"].astype(str) == ref.lower()].iloc[0]
        return {"paper_id": ref.lower(), "title": row.get("title", ""), "score": 1.0, "alternatives": []}

    if not ref:
        return {"paper_id": None, "title": None, "score": 0.0, "alternatives": []}

    titles = papers["title"].fillna("").astype(str)
    vec = TfidfVectorizer(lowercase=True, stop_words="english", ngram_range=(1, 2), min_df=1)
    matrix = vec.fit_transform(titles.tolist())
    q = vec.transform([ref])
    sims = linear_kernel(q, matrix).ravel()
    if sims.size == 0:
        return {"paper_id": None, "title": None, "score": 0.0, "alternatives": []}

    order = np.argsort(sims)[::-1][:top_n]
    best = int(order[0])
    alts = [
        {"paper_id": str(papers.iloc[int(i)]["paperId"]),
         "title": str(papers.iloc[int(i)]["title"]),
         "score": float(sims[int(i)])}
        for i in order
    ]
    return {
        "paper_id": str(papers.iloc[best]["paperId"]),
        "title": str(papers.iloc[best]["title"]),
        "score": float(sims[best]),
        "alternatives": alts[1:],
    }


# --------------------------------------------------------------------------- #
# Orchestrator                                                                #
# --------------------------------------------------------------------------- #
def parse_query(
    query: str,
    papers: Optional[pd.DataFrame] = None,
    client: Optional[LLMClient] = None,
) -> ParsedQuery:
    """Parse a natural-language query and (if papers given) resolve the target."""
    if client is None or not client.available:
        raise RuntimeError("DeepSeek-backed query parsing requires a configured API key.")

    fields = parse_query_llm(query, client)
    backend = "llm"

    parsed = ParsedQuery(
        raw_query=query,
        target_reference=fields["target_reference"],
        desired_relation=fields["desired_relation"],
        top_k=fields.get("top_k", 20),
        filters=fields.get("filters") or {},
        parser_backend=backend,
    )

    if papers is not None and not papers.empty:
        res = resolve_target(parsed.target_reference, papers)
        parsed.target_paper_id = res["paper_id"]
        parsed.target_title = res["title"]
        parsed.match_score = res["score"]
        parsed.alternatives = res["alternatives"]
    return parsed
