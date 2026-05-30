from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Set, Tuple

import networkx as nx
import pandas as pd

from .io_utils import stringify_path


@dataclass
class NeighborResult:
    target_paper_id: str
    candidate_paper_id: str
    edge_type: str
    hop: int
    citation_path: str


class CitationGraph:
    """Directed citation/reference graph wrapper.

    Edges are stored exactly as provided in `edges.csv`. For candidate discovery,
    we also allow undirected traversal because shared citation neighborhoods are
    useful even when edge direction differs.
    """

    def __init__(self, edges: pd.DataFrame):
        self.edges = edges.copy()
        self.directed = nx.DiGraph()
        self.undirected = nx.Graph()

        for row in self.edges.itertuples(index=False):
            src = getattr(row, "source_paper_id")
            dst = getattr(row, "target_paper_id")
            edge_type = getattr(row, "edge_type", "unknown")
            hop = int(getattr(row, "hop", 1))
            self.directed.add_edge(src, dst, edge_type=edge_type, hop=hop)
            self.undirected.add_edge(src, dst, edge_type=edge_type, hop=hop)

        self._edge_lookup: Dict[Tuple[str, str], List[str]] = defaultdict(list)
        for row in self.edges.itertuples(index=False):
            src = getattr(row, "source_paper_id")
            dst = getattr(row, "target_paper_id")
            edge_type = getattr(row, "edge_type", "unknown")
            self._edge_lookup[(src, dst)].append(edge_type)
            self._edge_lookup[(dst, src)].append(edge_type)

    @property
    def nodes(self) -> Set[str]:
        return set(self.undirected.nodes())

    def one_hop_neighbors(self, paper_id: str) -> list[NeighborResult]:
        if paper_id not in self.undirected:
            return []

        results: list[NeighborResult] = []
        # Use raw edges where source is paper_id.
        direct = self.edges[self.edges["source_paper_id"] == paper_id]
        for row in direct.itertuples(index=False):
            results.append(
                NeighborResult(
                    target_paper_id=paper_id,
                    candidate_paper_id=getattr(row, "target_paper_id"),
                    edge_type=getattr(row, "edge_type", "unknown"),
                    hop=int(getattr(row, "hop", 1)),
                    citation_path=getattr(row, "citation_path", stringify_path([paper_id, getattr(row, "target_paper_id")]))
                )
            )

        # Also include edges where paper_id appears as target, because candidates can cite the anchor.
        reverse = self.edges[self.edges["target_paper_id"] == paper_id]
        for row in reverse.itertuples(index=False):
            src = getattr(row, "source_paper_id")
            results.append(
                NeighborResult(
                    target_paper_id=paper_id,
                    candidate_paper_id=src,
                    edge_type=f"reverse_{getattr(row, 'edge_type', 'unknown')}",
                    hop=1,
                    citation_path=stringify_path([paper_id, src])
                )
            )

        # Deduplicate candidate IDs but preserve first result.
        seen: set[str] = set()
        unique: list[NeighborResult] = []
        for r in results:
            if r.candidate_paper_id not in seen and r.candidate_paper_id != paper_id:
                seen.add(r.candidate_paper_id)
                unique.append(r)
        return unique

    def bfs_neighbors(self, paper_id: str, max_hop: int = 2, limit: Optional[int] = None) -> list[NeighborResult]:
        if paper_id not in self.undirected:
            return []

        visited = {paper_id}
        q = deque([(paper_id, [paper_id], 0)])
        results: list[NeighborResult] = []

        while q:
            node, path, depth = q.popleft()
            if depth >= max_hop:
                continue
            for nbr in self.undirected.neighbors(node):
                if nbr in visited:
                    continue
                visited.add(nbr)
                new_path = path + [nbr]
                hop = depth + 1
                edge_types = self._edge_lookup.get((node, nbr), ["unknown"])
                results.append(
                    NeighborResult(
                        target_paper_id=paper_id,
                        candidate_paper_id=nbr,
                        edge_type=edge_types[0] if hop == 1 else "two_hop",
                        hop=hop,
                        citation_path=stringify_path(new_path),
                    )
                )
                if limit is not None and len(results) >= limit:
                    return results
                q.append((nbr, new_path, hop))
        return results

    def shortest_hop(self, source: str, target: str, cutoff: int = 3) -> Optional[int]:
        if source not in self.undirected or target not in self.undirected:
            return None
        try:
            return nx.shortest_path_length(self.undirected, source, target, cutoff=cutoff)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None


def build_graph_stats(papers: pd.DataFrame, edges: pd.DataFrame) -> pd.DataFrame:
    g = CitationGraph(edges)
    years = pd.to_numeric(papers.get("year", pd.Series(dtype=float)), errors="coerce")
    stats = {
        "num_papers": len(papers),
        "num_edges": len(edges),
        "num_graph_nodes": len(g.nodes),
        "avg_degree_undirected": (sum(dict(g.undirected.degree()).values()) / max(1, len(g.nodes))),
        "num_components": nx.number_connected_components(g.undirected) if len(g.nodes) else 0,
        "year_min": int(years.min()) if years.notna().any() else None,
        "year_max": int(years.max()) if years.notna().any() else None,
        "abstract_coverage": float(papers["abstract"].notna().mean()) if "abstract" in papers else None,
        "low_connectivity_rate": float(papers["low_connectivity"].mean()) if "low_connectivity" in papers else None,
    }
    return pd.DataFrame([stats])
