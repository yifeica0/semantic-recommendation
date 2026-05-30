"""Citation-grounded explanations for recommendations.

Each top recommendation gets a short, evidence-grounded explanation built from
fields we already have (no extra LLM call): the citation-graph connection
(edge type / hop / rendered citation path) plus the LLM relation judgement
(label, satisfaction, reason) and the semantic-similarity signal.
"""

from __future__ import annotations

from typing import Dict, Optional

import pandas as pd

from .io_utils import parse_path


def _title_map(papers: pd.DataFrame) -> Dict[str, str]:
    if papers is None or papers.empty or "paperId" not in papers.columns:
        return {}
    return {str(pid): str(title) for pid, title in zip(papers["paperId"], papers.get("title", ""))}


def _short(title: str, n: int = 60) -> str:
    title = (title or "").strip()
    return title if len(title) <= n else title[: n - 1].rstrip() + "…"


def _render_path(path: str, titles: Dict[str, str]) -> str:
    ids = parse_path(path)
    if len(ids) < 2:
        return ""
    rendered = [_short(titles.get(pid, pid[:8] if pid else "?"), 40) for pid in ids]
    return " → ".join(rendered)


def _graph_clause(row: pd.Series, titles: Dict[str, str]) -> str:
    edge_type = str(row.get("edge_type", "") or "")
    try:
        hop = int(row.get("hop", 0) or 0)
    except (TypeError, ValueError):
        hop = 0
    path_str = _render_path(row.get("citation_path", ""), titles)

    if edge_type.startswith("reverse_"):
        base = "Cites the target paper (1 hop)."
    elif hop == 1 and "reference" in edge_type:
        base = "Appears in the target's reference list (1 hop)."
    elif hop == 1 and "citation" in edge_type:
        base = "Directly linked to the target by a citation edge (1 hop)."
    elif hop == 2 or edge_type == "two_hop":
        base = "Connected to the target through a 2-hop citation path."
    elif edge_type == "negative" or hop >= 999:
        base = "No direct citation-graph link; surfaced by semantic similarity."
    elif hop >= 1:
        base = f"Connected in the citation graph ({hop} hops)."
    else:
        base = "Related in the citation graph."

    if path_str:
        base += f" Path: {path_str}."
    return base


def _relation_clause(row: pd.Series, desired_relation: str) -> str:
    label = str(row.get("predicted_relation", "") or "").strip() or "unrelated"
    reason = str(row.get("relation_reason", "") or "").strip()
    sat = row.get("relation_satisfaction", None)
    conf = row.get("relation_confidence", None)

    bits = []
    verb = {
        "critique": "Critiques", "extension": "Extends",
        "application": "Applies", "background": "Background for", "unrelated": "Unrelated to",
    }.get(label, label.capitalize())
    head = f"{verb} the target"
    try:
        if sat is not None and pd.notna(sat):
            head += f" (desired-relation match {float(sat):.2f})"
        elif conf is not None and pd.notna(conf):
            head += f" (confidence {float(conf):.2f})"
    except (TypeError, ValueError):
        pass
    bits.append(head + ".")
    if reason:
        bits.append(reason if reason.endswith(".") else reason + ".")
    if label != desired_relation.lower():
        bits.append(f"(Query asked for '{desired_relation}'.)")
    return " ".join(bits)


def build_explanations(
    ranked: pd.DataFrame,
    papers: Optional[pd.DataFrame] = None,
    desired_relation: str = "extension",
) -> pd.DataFrame:
    """Add an ``explanation`` column grounded in the citation path + relation."""
    if ranked.empty:
        ranked = ranked.copy()
        ranked["explanation"] = []
        return ranked
    titles = _title_map(papers) if papers is not None else {}
    df = ranked.copy()

    def _explain(row: pd.Series) -> str:
        graph = _graph_clause(row, titles)
        relation = _relation_clause(row, desired_relation)
        sem = row.get("semantic_score", None)
        tail = ""
        try:
            if sem is not None and pd.notna(sem):
                tail = f" [semantic sim {float(sem):.2f}]"
        except (TypeError, ValueError):
            tail = ""
        return f"{relation} {graph}{tail}".strip()

    df["explanation"] = df.apply(_explain, axis=1)
    return df


def recommendations_markdown(ranked: pd.DataFrame, top_k: int = 10) -> str:
    """Render a human-readable recommendations report grouped by target."""
    if ranked.empty:
        return "# Recommendations\n\n(No recommendations.)\n"
    lines = ["# Relation-aware recommendations\n"]
    for target_id, group in ranked.groupby("target_paper_id"):
        group = group.sort_values("rank").head(top_k)
        target_title = str(group.iloc[0].get("target_title", "")) or target_id
        rel = str(group.iloc[0].get("desired_relation", "")) if "desired_relation" in group.columns else ""
        header = f"\n## Target: {target_title}"
        if rel:
            header += f"  —  desired relation: **{rel}**"
        lines.append(header)
        lines.append(f"`{target_id}`\n")
        for _, r in group.iterrows():
            score = float(r.get("final_score", 0.0) or 0.0)
            cand = str(r.get("candidate_title", "")) or str(r.get("candidate_paper_id", ""))
            lines.append(f"{int(r.get('rank', 0))}. **{cand}**  (score {score:.3f})")
            expl = str(r.get("explanation", "")).strip()
            if expl:
                lines.append(f"   - {expl}")
    return "\n".join(lines) + "\n"
