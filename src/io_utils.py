from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd


def ensure_parent(path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def read_jsonl(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        raise FileNotFoundError(f"JSONL file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on line {line_no} in {path}: {e}") from e
    return pd.DataFrame(rows)


def write_jsonl(df: pd.DataFrame, path: str | Path) -> None:
    ensure_parent(path)
    path = Path(path)
    with path.open("w", encoding="utf-8") as f:
        for record in df.to_dict(orient="records"):
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Table file not found: {path}")
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if suffix == ".jsonl":
        return read_jsonl(path)
    raise ValueError(f"Unsupported file type: {path}")


def write_table(df: pd.DataFrame, path: str | Path, index: bool = False) -> None:
    ensure_parent(path)
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        df.to_csv(path, index=index)
    elif suffix in {".parquet", ".pq"}:
        df.to_parquet(path, index=index)
    elif suffix == ".jsonl":
        write_jsonl(df, path)
    else:
        raise ValueError(f"Unsupported file type: {path}")


def normalize_paper_id(value: Any) -> Optional[str]:
    if value is None or pd.isna(value):
        return None
    s = str(value).strip()
    return s if s else None


def parse_path(value: Any) -> list[str]:
    if value is None or pd.isna(value):
        return []
    if isinstance(value, list):
        return [str(x) for x in value]
    s = str(value).strip()
    if not s:
        return []
    if "|" in s:
        return [x for x in s.split("|") if x]
    if "," in s:
        return [x.strip() for x in s.split(",") if x.strip()]
    return [s]


def stringify_path(path: Iterable[Any]) -> str:
    return "|".join(str(x) for x in path if x is not None)
