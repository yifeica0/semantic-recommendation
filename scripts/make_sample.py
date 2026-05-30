"""Build a tiny connected sample of the dataset for fast testing/demos.

Picks a handful of citation anchors, keeps their edges and the papers they
touch, and writes data/sample_papers.jsonl + data/sample_edges.csv. Use with
config_test.yaml.

  python scripts/make_sample.py --anchors 25 --max-edges 4000
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-papers", default=str(ROOT / "data" / "raw_papers.jsonl"))
    ap.add_argument("--raw-edges", default=str(ROOT / "data" / "raw_edges.csv"))
    ap.add_argument("--out-papers", default=str(ROOT / "data" / "sample_papers.jsonl"))
    ap.add_argument("--out-edges", default=str(ROOT / "data" / "sample_edges.csv"))
    ap.add_argument("--anchors", type=int, default=25)
    ap.add_argument("--max-edges", type=int, default=4000)
    args = ap.parse_args()

    # 1. Collect edges for the first N anchors.
    anchors: list[str] = []
    anchor_set: set[str] = set()
    kept_rows: list[dict] = []
    needed_ids: set[str] = set()
    fieldnames = None
    with open(args.raw_edges, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for row in reader:
            anchor = row.get("anchor_paper_id") or row.get("target_paper_id")
            if anchor not in anchor_set:
                if len(anchor_set) >= args.anchors:
                    if anchor not in anchor_set:
                        continue
                anchor_set.add(anchor)
                anchors.append(anchor)
            kept_rows.append(row)
            needed_ids.add(row["source_paper_id"])
            needed_ids.add(row["target_paper_id"])
            if len(kept_rows) >= args.max_edges:
                break

    # 2. Stream papers, keep those referenced by the kept edges.
    found = 0
    with open(args.raw_papers, encoding="utf-8") as fin, open(args.out_papers, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if obj.get("paperId") in needed_ids:
                fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
                found += 1

    # 3. Write the kept edges.
    with open(args.out_edges, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(kept_rows)

    print(f"anchors={len(anchors)} edges={len(kept_rows)} referenced_ids={len(needed_ids)} papers_found={found}")
    print(f"wrote {args.out_papers} and {args.out_edges}")


if __name__ == "__main__":
    main()
