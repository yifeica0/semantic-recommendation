from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd

from .io_utils import normalize_paper_id


REQUIRED_PAPER_COLUMNS = ["paperId", "title", "abstract"]


def validate_papers_schema(papers: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_PAPER_COLUMNS if c not in papers.columns]
    if missing:
        raise ValueError(f"Missing required paper columns: {missing}")


def clean_papers(
    papers: pd.DataFrame,
    min_abstract_chars: int = 50,
    low_connectivity_citation_threshold: int = 5,
    low_connectivity_reference_threshold: int = 5,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Clean paper metadata.

    Returns:
        cleaned_papers, cleaning_report
    """
    validate_papers_schema(papers)
    df = papers.copy()
    before = len(df)

    df["paperId"] = df["paperId"].map(normalize_paper_id)
    df["title"] = df["title"].fillna("").astype(str).str.strip()
    df["abstract"] = df["abstract"].fillna("").astype(str).str.strip()

    if "citationCount" not in df.columns:
        df["citationCount"] = 0
    if "referenceCount" not in df.columns:
        df["referenceCount"] = 0
    if "year" not in df.columns:
        df["year"] = np.nan

    df["citationCount"] = pd.to_numeric(df["citationCount"], errors="coerce").fillna(0).astype(int)
    df["referenceCount"] = pd.to_numeric(df["referenceCount"], errors="coerce").fillna(0).astype(int)
    df["year"] = pd.to_numeric(df["year"], errors="coerce")

    report_rows = []

    def add_report(step: str, n_before: int, n_after: int) -> None:
        report_rows.append({"step": step, "before": n_before, "after": n_after, "removed": n_before - n_after})

    n = len(df)
    df = df[df["paperId"].notna()].copy()
    add_report("remove_missing_paper_id", n, len(df))

    n = len(df)
    df = df[df["title"].str.len() > 0].copy()
    add_report("remove_empty_title", n, len(df))

    n = len(df)
    df = df[df["abstract"].str.len() >= min_abstract_chars].copy()
    add_report("remove_missing_or_short_abstract", n, len(df))

    n = len(df)
    df = df.drop_duplicates(subset=["paperId"], keep="first").copy()
    add_report("deduplicate_paper_id", n, len(df))

    df["low_connectivity"] = (
        (df["citationCount"] < low_connectivity_citation_threshold)
        | (df["referenceCount"] < low_connectivity_reference_threshold)
    )

    df["text_for_retrieval"] = (df["title"].fillna("") + ". " + df["abstract"].fillna("")).str.strip()

    report_rows.insert(0, {"step": "initial", "before": before, "after": before, "removed": 0})
    report = pd.DataFrame(report_rows)
    return df.reset_index(drop=True), report


def clean_edges(edges: pd.DataFrame, valid_paper_ids: set[str] | None = None) -> pd.DataFrame:
    required = ["source_paper_id", "target_paper_id"]
    missing = [c for c in required if c not in edges.columns]
    if missing:
        raise ValueError(f"Missing required edge columns: {missing}")

    df = edges.copy()
    df["source_paper_id"] = df["source_paper_id"].map(normalize_paper_id)
    df["target_paper_id"] = df["target_paper_id"].map(normalize_paper_id)

    if "edge_type" not in df.columns:
        df["edge_type"] = "citation"
    if "hop" not in df.columns:
        df["hop"] = 1
    if "citation_path" not in df.columns:
        df["citation_path"] = df["source_paper_id"].astype(str) + "|" + df["target_paper_id"].astype(str)

    df["edge_type"] = df["edge_type"].fillna("unknown").astype(str)
    df["hop"] = pd.to_numeric(df["hop"], errors="coerce").fillna(1).astype(int)

    df = df.dropna(subset=["source_paper_id", "target_paper_id"])
    df = df[df["source_paper_id"] != df["target_paper_id"]].copy()

    if valid_paper_ids is not None:
        df = df[df["source_paper_id"].isin(valid_paper_ids) & df["target_paper_id"].isin(valid_paper_ids)].copy()

    df = df.drop_duplicates(subset=["source_paper_id", "target_paper_id", "edge_type"], keep="first")
    return df.reset_index(drop=True)
