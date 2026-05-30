from __future__ import annotations

import pandas as pd


def dataset_statistics(papers: pd.DataFrame, edges: pd.DataFrame, pairs: pd.DataFrame | None = None) -> pd.DataFrame:
    stats = {
        "num_papers": len(papers),
        "num_edges": len(edges),
        "num_candidate_pairs": len(pairs) if pairs is not None else None,
        "num_targets": pairs["target_paper_id"].nunique() if pairs is not None and not pairs.empty else None,
        "abstract_coverage": papers["abstract"].notna().mean() if "abstract" in papers else None,
        "low_connectivity_rate": papers["low_connectivity"].mean() if "low_connectivity" in papers else None,
        "avg_citation_count": pd.to_numeric(papers.get("citationCount", 0), errors="coerce").fillna(0).mean(),
        "avg_reference_count": pd.to_numeric(papers.get("referenceCount", 0), errors="coerce").fillna(0).mean(),
    }
    return pd.DataFrame([stats])


def top_recommendation_examples(ranked: pd.DataFrame, k: int = 5) -> pd.DataFrame:
    cols = [
        "target_title", "candidate_title", "rank", "final_score", "semantic_score", "graph_score",
        "relation_score", "predicted_relation", "relation_reason", "citation_path"
    ]
    available = [c for c in cols if c in ranked.columns]
    return ranked.sort_values(["target_paper_id", "rank"]).groupby("target_paper_id").head(k)[available]
