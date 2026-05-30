from __future__ import annotations

from typing import Iterable, Optional

import numpy as np
import pandas as pd

from .graph import CitationGraph


def select_target_papers(
    papers: pd.DataFrame,
    n_targets: int = 50,
    random_state: int = 172,
) -> list[str]:
    """Select target papers with a mix of high-citation, recent, and random papers."""
    rng = np.random.default_rng(random_state)
    df = papers.copy()
    df["citationCount"] = pd.to_numeric(df.get("citationCount", 0), errors="coerce").fillna(0)
    df["year"] = pd.to_numeric(df.get("year", np.nan), errors="coerce")

    selected: list[str] = []

    high_n = max(1, int(n_targets * 0.4))
    recent_n = max(1, int(n_targets * 0.3))
    random_n = n_targets - high_n - recent_n

    high = df.sort_values("citationCount", ascending=False).head(high_n)["paperId"].tolist()
    selected.extend(high)

    recent_pool = df[df["year"] >= 2020]
    recent_pool = recent_pool[~recent_pool["paperId"].isin(selected)]
    if len(recent_pool) > 0:
        recent = recent_pool.sample(n=min(recent_n, len(recent_pool)), random_state=random_state)["paperId"].tolist()
        selected.extend(recent)

    remaining = df[~df["paperId"].isin(selected)]
    if random_n > 0 and len(remaining) > 0:
        random = remaining.sample(n=min(random_n, len(remaining)), random_state=random_state + 1)["paperId"].tolist()
        selected.extend(random)

    if len(selected) < n_targets:
        remaining = df[~df["paperId"].isin(selected)]
        extra = remaining.head(n_targets - len(selected))["paperId"].tolist()
        selected.extend(extra)

    return selected[:n_targets]


def build_candidate_pairs(
    papers: pd.DataFrame,
    edges: pd.DataFrame,
    target_ids: Optional[Iterable[str]] = None,
    n_targets: int = 50,
    candidates_per_target: int = 200,
    include_two_hop: bool = True,
    max_two_hop_per_target: int = 100,
    negatives_per_target: int = 50,
    random_state: int = 172,
) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    paper_ids = set(papers["paperId"])
    graph = CitationGraph(edges)
    paper_meta = papers.set_index("paperId").to_dict(orient="index")

    if target_ids is None:
        target_ids = select_target_papers(papers, n_targets=n_targets, random_state=random_state)
    else:
        target_ids = [pid for pid in target_ids if pid in paper_ids]

    rows = []
    all_ids = np.array(list(paper_ids))

    for target_id in target_ids:
        candidate_results = []
        candidate_results.extend(graph.one_hop_neighbors(target_id))
        if include_two_hop:
            two_hop = [r for r in graph.bfs_neighbors(target_id, max_hop=2, limit=max_two_hop_per_target * 3) if r.hop == 2]
            candidate_results.extend(two_hop[:max_two_hop_per_target])

        # Remove candidates missing metadata and deduplicate.
        seen = {target_id}
        filtered = []
        for r in candidate_results:
            if r.candidate_paper_id in paper_ids and r.candidate_paper_id not in seen:
                seen.add(r.candidate_paper_id)
                filtered.append(r)

        if len(filtered) > candidates_per_target:
            filtered = list(rng.choice(filtered, size=candidates_per_target, replace=False))

        for r in filtered:
            t = paper_meta[target_id]
            c = paper_meta[r.candidate_paper_id]
            rows.append(_pair_row(target_id, r.candidate_paper_id, t, c, r.edge_type, r.hop, r.citation_path, label=""))

        # Negative/unrelated candidates: no immediate graph relation.
        existing = {r.candidate_paper_id for r in filtered} | {target_id}
        neg_pool = [pid for pid in all_ids if pid not in existing]
        if neg_pool and negatives_per_target > 0:
            neg_ids = rng.choice(neg_pool, size=min(negatives_per_target, len(neg_pool)), replace=False)
            for neg_id in neg_ids:
                t = paper_meta[target_id]
                c = paper_meta[neg_id]
                rows.append(_pair_row(target_id, neg_id, t, c, "negative", 999, "", label="unrelated"))

    pairs = pd.DataFrame(rows)
    if not pairs.empty:
        pairs = pairs.drop_duplicates(subset=["target_paper_id", "candidate_paper_id"]).reset_index(drop=True)
    return pairs


def _pair_row(
    target_id: str,
    candidate_id: str,
    target_meta: dict,
    candidate_meta: dict,
    edge_type: str,
    hop: int,
    citation_path: str,
    label: str = "",
) -> dict:
    return {
        "target_paper_id": target_id,
        "target_title": target_meta.get("title", ""),
        "target_abstract": target_meta.get("abstract", ""),
        "candidate_paper_id": candidate_id,
        "candidate_title": candidate_meta.get("title", ""),
        "candidate_abstract": candidate_meta.get("abstract", ""),
        "candidate_year": candidate_meta.get("year", None),
        "candidate_citationCount": candidate_meta.get("citationCount", 0),
        "candidate_referenceCount": candidate_meta.get("referenceCount", 0),
        "edge_type": edge_type,
        "hop": hop,
        "citation_path": citation_path,
        "label": label,
    }
