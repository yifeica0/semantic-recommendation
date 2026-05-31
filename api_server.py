from __future__ import annotations

import argparse
import json
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from src.clean import clean_edges, clean_papers
from src.config import load_config
from src.explain import build_explanations
from src.io_utils import read_jsonl, read_table
from src.llm_client import LLMClient, LLMConfig
from src.pair_builder import build_candidate_pairs
from src.query_parser import parse_query
from src.rankers import RankingWeights, finalize_ranking, retrieve_candidates, rank_candidate_pairs
from src.relation_scorer import score_pairs
from src.self_training import normalize_weights


class RecommendationApp:
    def __init__(self, config_path: str | Path):
        self.cfg = load_config(config_path)
        self.base_dir = self.cfg.base_dir
        self.frontend_path = self.base_dir / "frontend.html"
        self.client = LLMClient(LLMConfig.from_dict(self.cfg.get("llm", {})), base_dir=self.base_dir)
        self._load_lock = threading.Lock()
        self._loaded = False
        self._loading = False
        self._load_error: str | None = None
        self.papers = None
        self.edges = None
        self.cleaning_report = None

    def start_background_load(self) -> None:
        if self._loaded or self._loading:
            return
        self._loading = True

        def _worker() -> None:
            try:
                self.ensure_loaded()
            finally:
                self._loading = False

        threading.Thread(target=_worker, name="recommendation-load", daemon=True).start()

    def ensure_loaded(self) -> None:
        if self._loaded:
            return
        with self._load_lock:
            if self._loaded:
                return
            raw_papers = read_jsonl(self.cfg.path("paths.raw_papers"))
            raw_edges = read_table(self.cfg.path("paths.raw_edges"))
            self.papers, self.cleaning_report = clean_papers(
                raw_papers,
                min_abstract_chars=self.cfg.get("cleaning.min_abstract_chars", 50),
                low_connectivity_citation_threshold=self.cfg.get("cleaning.low_connectivity_citation_threshold", 5),
                low_connectivity_reference_threshold=self.cfg.get("cleaning.low_connectivity_reference_threshold", 5),
            )
            self.edges = clean_edges(raw_edges, valid_paper_ids=set(self.papers["paperId"]))
            self._loaded = True
            self._load_error = None

    def recommend(self, query: str, top_k: int = 10, interactive: bool = True, fast: bool = True) -> dict[str, Any]:
        # If loading is ongoing, don't block here — return an explicit loading status so the UI can retry.
        if not self._loaded:
            if self._loading:
                return {"query": query, "parsed": None, "results": [], "error": "loading"}
            # not loaded and not loading -> load now (blocking)
            self.ensure_loaded()
        if self.papers is None or self.edges is None:
            return {"query": query, "parsed": None, "results": [], "error": self._load_error or "data not loaded"}

        parsed = parse_query(query, self.papers, self.client)
        desired_relation = parsed.desired_relation
        target_ids = [parsed.target_paper_id] if parsed.target_paper_id else None

        # Allow interactive (frontend) mode to be faster by capping rerank size/targets.
        rerank_top_k_cfg = int(self.cfg.get("ranking.rerank_top_k", 30))
        rerank_top_k = rerank_top_k_cfg
        if interactive:
            rerank_top_k = min(rerank_top_k_cfg, 6)

        pairs = build_candidate_pairs(
            self.papers,
            self.edges,
            target_ids=target_ids,
            n_targets=self.cfg.get("pair_building.n_targets", 50),
            candidates_per_target=self.cfg.get("pair_building.candidates_per_target", 200),
            include_two_hop=self.cfg.get("pair_building.include_two_hop", True),
            max_two_hop_per_target=self.cfg.get("pair_building.max_two_hop_per_target", 100),
            negatives_per_target=self.cfg.get("pair_building.negatives_per_target", 50),
            random_state=self.cfg.get("pair_building.random_state", 172),
        )

        if pairs.empty:
            return {"query": query, "parsed": parsed.to_dict(), "results": []}
        # Optionally allow loading learned weights from a previous self-training run
        learned_path = self.cfg.get("ranking.learned_weights_path")
        if learned_path:
            try:
                lp = Path(learned_path)
                if not lp.is_absolute():
                    lp = self.base_dir / learned_path
                if lp.exists():
                    learned = json.loads(lp.read_text(encoding="utf-8"))
                    weights_dict = dict(self.cfg.get("ranking.weights", {}))
                    weights_dict.update(learned)
                    weights = normalize_weights(RankingWeights.from_dict(weights_dict))
                else:
                    weights = normalize_weights(RankingWeights.from_dict(self.cfg.get("ranking.weights", {})))
            except Exception:
                weights = normalize_weights(RankingWeights.from_dict(self.cfg.get("ranking.weights", {})))
        else:
            weights = normalize_weights(RankingWeights.from_dict(self.cfg.get("ranking.weights", {})))
        # Fast path: return baseline ranking (no LLM scoring) quickly for interactive UI.
        if fast:
            ranked = rank_candidate_pairs(pairs, self.papers, relation_scores=None, desired_relation=desired_relation, weights=weights, mmr_lambda=self.cfg.get("ranking.mmr_lambda", 0.75), top_k=top_k)
        else:
            survivors, text_model = retrieve_candidates(
                pairs,
                self.papers,
                weights=weights,
                top_k=rerank_top_k,
            )
            # If interactive, cap the number of targets we send to the LLM to keep latency low.
            if interactive:
                max_targets_cfg = self.cfg.get("ranking.max_rerank_targets")
                max_targets = int(max_targets_cfg) if max_targets_cfg is not None else 3
                unique_targets = list(dict.fromkeys(survivors["target_paper_id"].tolist()))
                limited_targets = set(unique_targets[:max_targets])
                survivors = survivors[survivors["target_paper_id"].isin(limited_targets)].copy()
            relation_scores = score_pairs(survivors, desired_relation=desired_relation, client=self.client)
            ranked = finalize_ranking(
                survivors,
                relation_scores=relation_scores,
                desired_relation=desired_relation,
                weights=weights,
                mmr_lambda=self.cfg.get("ranking.mmr_lambda", 0.75),
                text_model=text_model,
            )
        ranked = build_explanations(ranked, self.papers, desired_relation=desired_relation)

        if parsed.target_paper_id:
            ranked = ranked[ranked["target_paper_id"] == parsed.target_paper_id].copy()

        ranked = ranked.sort_values(["target_paper_id", "rank"]).head(int(top_k)).reset_index(drop=True)
        results = []
        for row in ranked.to_dict(orient="records"):
            results.append(
                {
                    "target_paper_id": row.get("target_paper_id"),
                    "target_title": row.get("target_title"),
                    "candidate_paper_id": row.get("candidate_paper_id"),
                    "title": row.get("candidate_title"),
                    "abstract": row.get("candidate_abstract", ""),
                    "score": float(row.get("final_score", 0.0) or 0.0),
                    "rank": int(row.get("rank", 0) or 0),
                    "desired_relation": row.get("desired_relation", desired_relation),
                    "predicted_relation": row.get("predicted_relation", ""),
                    "relation_score": float(row.get("relation_score", 0.0) or 0.0),
                    "explanation": row.get("explanation", ""),
                }
            )

        return {"query": query, "parsed": parsed.to_dict(), "results": results}


class RecommendationHandler(BaseHTTPRequestHandler):
    app: RecommendationApp

    def _send_json(self, payload: dict[str, Any], status: int = HTTPStatus.OK) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_text(self, text: str, content_type: str = "text/plain; charset=utf-8", status: int = HTTPStatus.OK) -> None:
        data = text.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path == "/api/health":
            self._send_json({"ok": True, "llm": self.app.client.available, "loaded": self.app._loaded, "loading": self.app._loading})
            return
        if parsed.path in {"/", "/frontend.html"}:
            if not self.app.frontend_path.exists():
                self._send_text("frontend.html not found", status=HTTPStatus.NOT_FOUND)
                return
            self._send_text(self.app.frontend_path.read_text(encoding="utf-8"), content_type="text/html; charset=utf-8")
            return
        self._send_text("Not found", status=HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path != "/api/search":
            self._send_text("Not found", status=HTTPStatus.NOT_FOUND)
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length) if length > 0 else b"{}"
            body = json.loads(raw.decode("utf-8"))
            query = str(body.get("query", "")).strip()
            top_k = int(body.get("top_k", 10) or 10)
            if not query:
                self._send_json({"query": "", "parsed": None, "results": []})
                return
            payload = self.app.recommend(query, top_k=top_k)
            self._send_json(payload)
        except Exception as exc:  # pragma: no cover - runtime guard for UI calls
            self._send_json({"error": str(exc)}, status=HTTPStatus.INTERNAL_SERVER_ERROR)


def main() -> None:
    parser = argparse.ArgumentParser(description="Local recommendation API for the frontend.")
    parser.add_argument("--config", default="config.yaml", help="Path to the pipeline config file.")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind the local server to.")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind the local server to.")
    args = parser.parse_args()

    app = RecommendationApp(args.config)
    app.start_background_load()
    RecommendationHandler.app = app
    server = ThreadingHTTPServer((args.host, args.port), RecommendationHandler)
    print(f"Serving frontend on http://{args.host}:{args.port}")
    print(f"API health check: http://{args.host}:{args.port}/api/health")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("Shutting down.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()