"""Evaluate ranking quality using citation-graph edges as weak labels.

This is a no-human-label sanity check. A ranked candidate is relevant when it is
a direct one-hop citation/reference neighbor of the target; two-hop and random
negative candidates are treated as not relevant.

Usage:
    python scripts/evaluate_citation_reconstruction.py \
        --ranked outputs/ranked_results.csv \
        --out outputs/citation_reconstruction_metrics.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.evaluation import evaluate_ranking
from src.io_utils import read_table, write_table

RELEVANT = "relevant"
IRRELEVANT = "unrelated"


def label_from_graph(ranked: pd.DataFrame) -> pd.DataFrame:
    df = ranked.copy()
    hop = pd.to_numeric(df.get("hop"), errors="coerce")
    edge = df.get("edge_type", pd.Series([""] * len(df))).fillna("").astype(str).str.lower()

    is_direct = (hop == 1) & ~edge.str.contains("two_hop") & (edge != "negative")
    df["label"] = pd.Series([RELEVANT if value else IRRELEVANT for value in is_direct], index=df.index)
    return df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "config.yaml"))
    parser.add_argument("--ranked", default=None, help="ranked results CSV (default: paths.ranked_results)")
    parser.add_argument("--out", default=str(ROOT / "outputs" / "citation_reconstruction_metrics.csv"))
    args = parser.parse_args()
    cfg = load_config(args.config)

    ranked_path = args.ranked or cfg.path("paths.ranked_results")
    ranked = read_table(ranked_path)
    labelled = label_from_graph(ranked)

    n_pos = int((labelled["label"] == RELEVANT).sum())
    n_targets = labelled["target_paper_id"].nunique()
    print(f"Loaded {len(labelled)} ranked rows across {n_targets} target(s); "
          f"{n_pos} direct-citation positives.")

    metrics = evaluate_ranking(
        labelled,
        k_values=cfg.get("evaluation.k_values", [5, 10, 20]),
        desired_relation=None,
        positive_labels=[RELEVANT],
    )
    write_table(metrics, args.out)
    print(metrics.to_string(index=False))
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
