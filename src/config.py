from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import yaml


@dataclass
class ProjectConfig:
    raw: Dict[str, Any]
    base_dir: Path

    def path(self, dotted_key: str) -> Path:
        cur: Any = self.raw
        for part in dotted_key.split("."):
            cur = cur[part]
        return (self.base_dir / cur).resolve()

    def get(self, dotted_key: str, default: Any = None) -> Any:
        cur: Any = self.raw
        for part in dotted_key.split("."):
            if not isinstance(cur, dict) or part not in cur:
                return default
            cur = cur[part]
        return cur


def load_config(path: str | Path) -> ProjectConfig:
    path = Path(path).resolve()
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return ProjectConfig(raw=raw, base_dir=path.parent)
