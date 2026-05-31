"""End-to-end relation-aware citation recommender pipeline."""

from __future__ import annotations

import argparse
import json
from typing import Optional

import pandas as pd

from src.clean import clean_edges, clean_papers
from src.config import ProjectConfig, load_config
from src.evaluation import evaluate_ranking, evaluate_relation_classification
from src.explain import build_explanations, recommendations_markdown
from src.graph import build_graph_stats
from src.io_utils import read_jsonl, read_table, write_jsonl, write_table
from src.llm_client import LLMClient, LLMConfig
from src.pair_builder import build_candidate_pairs
from src.query_parser import parse_query
from src.rankers import RankingWeights, finalize_ranking, retrieve_candidates
from src.relation_scorer import score_pairs
from src.report_tables import dataset_statistics, top_recommendation_examples
from src.self_training import (
    SelfTrainingConfig,
    add_judge_reward,
    build_pseudo_qrels,
    evaluate_pseudo_ranking,
    merge_relation_score_frames,
    normalize_weights,
    rank_with_weights,
    tune_weights_from_judge,
    weights_to_dict,
)


def build_client(cfg: ProjectConfig) -> LLMClient:
    return LLMClient(LLMConfig.from_dict(cfg.get("llm", {})), base_dir=cfg.base_dir)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--query", default=None, help="Natural-language request; parsed into target + relation.")
    parser.add_argument("--desired-relation", default=None, help="Override the desired relation.")
    parser.add_argument("--mode", choices=["llm"], default="llm",
                        help="DeepSeek-backed mode only.")
    parser.add_argument("--max-targets", type=int, default=None, help="Cap #targets sent to the judge.")
    parser.add_argument("--reuse-clean", action="store_true",
                        help="Reuse already-cleaned papers/edges from disk instead of re-cleaning the raw corpus "
                             "(query-independent; big speedup when testing many queries). Falls back to cleaning "
                             "if the cached files are missing or empty.")
    parser.add_argument("--fast-features", action="store_true",
                        help="Fit TF-IDF only on the candidate pairs instead of the full corpus. "
                             "Much faster for testing a single query; IDF is estimated from the candidate set.")
    parser.add_argument("--use-learned-weights", action="store_true",
                        help="Load ranking weights from outputs/learned_weights.json instead of config "
                             "(reuse a previous self-training run; pair with --no-self-train to skip retuning).")
    parser.add_argument("--self-train", action="store_true", help="Enable zero-manual-label iterative weight tuning.")
    parser.add_argument("--no-self-train", action="store_true", help="Disable self-training even if config enables it.")
    parser.add_argument("--iterations", type=int, default=None, help="Override self_training.iterations.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    client = build_client(cfg)
    print(f"[llm] {client.status_message()}")

    cleaned, cleaned_edges = _clean_inputs(cfg, reuse=args.reuse_clean)
    query_client = None if args.mode == "heuristic" else client
    desired_relation, target_ids = _parse_query(args, cfg, cleaned, query_client)
    pairs = _build_pairs(cfg, cleaned, cleaned_edges, target_ids)
    if pairs.empty:
        print("      No candidate pairs (target may have no graph neighbors). Stopping.")
        return

    weights = _load_weights(args, cfg)
    self_training = _self_training_config(args, cfg)

    if self_training.enabled:
        print("[4/7] Self-training: iterative LLM-judge tuning...")
        ranked, relation_scores, weights = _run_self_training(
            pairs=pairs,
            papers=cleaned,
            cfg=cfg,
            args=args,
            client=client,
            desired_relation=desired_relation,
            weights=weights,
            self_training=self_training,
        )
    else:
        print("[4/7] Stage 1: retrieval prefilter...")
        write_table(pd.DataFrame([{"enabled": 0, "reason": "self_training_disabled"}]),
                    cfg.base_dir / "outputs" / "self_training_metrics.csv")
        survivors, text_model = _retrieve_survivors(pairs, cleaned, cfg, args, weights)
        print(f"      {len(survivors)} survivors across {survivors['target_paper_id'].nunique()} targets")

        print(f"[5/7] Stage 2: relation judging (mode={args.mode}, relation='{desired_relation}')...")
        relation_scores = _score_survivors(survivors, desired_relation, client)
        write_table(relation_scores, cfg.path("paths.relation_scores"))

        print("[6/7] Final ranking + explanations...")
        ranked = finalize_ranking(
            survivors,
            relation_scores=relation_scores,
            desired_relation=desired_relation,
            weights=weights,
            mmr_lambda=cfg.get("ranking.mmr_lambda", 0.75),
            text_model=text_model,
        )

    ranked["desired_relation"] = desired_relation
    ranked = build_explanations(ranked, cleaned, desired_relation=desired_relation)
    write_table(ranked, cfg.path("paths.ranked_results"))
    (cfg.base_dir / "outputs" / "recommendations.md").write_text(
        recommendations_markdown(ranked, top_k=cfg.get("ranking.top_k", 20)), encoding="utf-8"
    )
    _write_learned_weights(cfg, weights)

    print("[7/7] Evaluation + report tables...")
    _write_evaluation_outputs(cfg, cleaned, cleaned_edges, pairs, ranked, desired_relation)
    print("Done. See outputs/recommendations.md for ranked, citation-grounded recommendations.")


def _nonempty(path) -> bool:
    return path.exists() and path.stat().st_size > 0


def _clean_inputs(cfg: ProjectConfig, reuse: bool = False):
    cleaned_path = cfg.path("paths.cleaned_papers")
    edges_path = cfg.path("paths.graph_edges")
    if reuse:
        if _nonempty(cleaned_path) and _nonempty(edges_path):
            print(f"[1/7] Reusing cleaned papers/edges from disk (skipping clean).")
            cleaned = read_jsonl(cleaned_path)
            cleaned_edges = read_table(edges_path)
            print(f"      papers={len(cleaned)} edges={len(cleaned_edges)}")
            return cleaned, cleaned_edges
        print("[1/7] --reuse-clean set but cached files missing/empty; cleaning from raw.")

    print("[1/7] Cleaning papers and edges...")
    papers = read_jsonl(cfg.path("paths.raw_papers"))
    edges = read_table(cfg.path("paths.raw_edges"))
    cleaned, cleaning_report = clean_papers(
        papers,
        min_abstract_chars=cfg.get("cleaning.min_abstract_chars", 50),
        low_connectivity_citation_threshold=cfg.get("cleaning.low_connectivity_citation_threshold", 5),
        low_connectivity_reference_threshold=cfg.get("cleaning.low_connectivity_reference_threshold", 5),
    )
    cleaned_edges = clean_edges(edges, valid_paper_ids=set(cleaned["paperId"]))
    write_jsonl(cleaned, cfg.path("paths.cleaned_papers"))
    write_table(cleaned_edges, cfg.path("paths.graph_edges"))
    write_table(cleaning_report, cfg.base_dir / "outputs" / "cleaning_report.csv")
    print(f"      papers={len(cleaned)} edges={len(cleaned_edges)}")
    return cleaned, cleaned_edges


def _parse_query(args, cfg: ProjectConfig, cleaned: pd.DataFrame,
                 client: Optional[LLMClient]) -> tuple[str, Optional[list[str]]]:
    desired_relation = args.desired_relation or cfg.get("ranking.desired_relation", "extension")
    target_ids = None
    if args.query:
        print("[2/7] Parsing query...")
        parsed = parse_query(args.query, cleaned, client)
        desired_relation = args.desired_relation or parsed.desired_relation
        print(f"      backend={parsed.parser_backend} relation='{parsed.desired_relation}'")
        print(f"      target -> {parsed.target_title!r} ({parsed.target_paper_id}) match={parsed.match_score:.2f}")
        if parsed.alternatives:
            print("      alternatives:", "; ".join(
                f"{a['title'][:40]} ({a['score']:.2f})" for a in parsed.alternatives[:3]
            ))
        if parsed.target_paper_id:
            target_ids = [parsed.target_paper_id]
        write_jsonl(pd.DataFrame([parsed.to_dict()]), cfg.base_dir / "outputs" / "parsed_query.jsonl")
    else:
        print("[2/7] No query given; selecting target papers automatically.")
    return desired_relation, target_ids


def _build_pairs(cfg: ProjectConfig, cleaned: pd.DataFrame, cleaned_edges: pd.DataFrame,
                 target_ids: Optional[list[str]]) -> pd.DataFrame:
    print("[3/7] Building candidate pairs...")
    pairs = build_candidate_pairs(
        cleaned,
        cleaned_edges,
        target_ids=target_ids,
        n_targets=cfg.get("pair_building.n_targets", 50),
        candidates_per_target=cfg.get("pair_building.candidates_per_target", 200),
        include_two_hop=cfg.get("pair_building.include_two_hop", True),
        max_two_hop_per_target=cfg.get("pair_building.max_two_hop_per_target", 100),
        negatives_per_target=cfg.get("pair_building.negatives_per_target", 50),
        random_state=cfg.get("pair_building.random_state", 172),
    )
    write_table(pairs, cfg.path("paths.candidate_pairs"))
    print(f"      candidate pairs={len(pairs)}")
    return pairs


def _self_training_config(args, cfg: ProjectConfig) -> SelfTrainingConfig:
    self_training = SelfTrainingConfig.from_dict(cfg.get("self_training", {}))
    if args.self_train:
        self_training.enabled = True
    if args.no_self_train:
        self_training.enabled = False
    if args.iterations is not None:
        self_training.iterations = max(1, int(args.iterations))
    return self_training


def _retrieve_survivors(
    pairs: pd.DataFrame,
    papers: pd.DataFrame,
    cfg: ProjectConfig,
    args,
    weights: RankingWeights,
):
    survivors, text_model = retrieve_candidates(
        pairs,
        papers,
        weights=weights,
        top_k=cfg.get("ranking.rerank_top_k", 30),
        full_corpus=not getattr(args, "fast_features", False),
    )
    max_targets = args.max_targets if args.max_targets is not None else cfg.get("ranking.max_rerank_targets", None)
    if max_targets:
        keep = list(dict.fromkeys(survivors["target_paper_id"]))[: int(max_targets)]
        survivors = survivors[survivors["target_paper_id"].isin(keep)].reset_index(drop=True)
    write_table(survivors, cfg.path("paths.rerank_candidates"))
    return survivors, text_model


def _score_survivors(
    survivors: pd.DataFrame,
    desired_relation: str,
    client: LLMClient,
) -> pd.DataFrame:
    return score_pairs(survivors, desired_relation=desired_relation, client=client)


def _run_self_training(
    pairs: pd.DataFrame,
    papers: pd.DataFrame,
    cfg: ProjectConfig,
    args,
    client: LLMClient,
    desired_relation: str,
    weights: RankingWeights,
    self_training: SelfTrainingConfig,
):
    relation_scores: Optional[pd.DataFrame] = _load_existing_relation_scores(cfg)
    history = []
    current_weights = weights
    final_text_model = None

    for iteration in range(1, self_training.iterations + 1):
        print(f"      iteration {iteration}/{self_training.iterations}: retrieve -> judge -> tune")
        survivors, text_model = _retrieve_survivors(pairs, papers, cfg, args, current_weights)
        final_text_model = text_model
        relation_scores = _score_missing_survivors(
            survivors, relation_scores, desired_relation, client
        )
        write_table(relation_scores, cfg.path("paths.relation_scores"))

        judged = finalize_ranking(
            survivors,
            relation_scores=relation_scores,
            desired_relation=desired_relation,
            weights=current_weights,
            mmr_lambda=cfg.get("ranking.mmr_lambda", 0.75),
            text_model=text_model,
        )
        judged["desired_relation"] = desired_relation
        judged = add_judge_reward(judged, desired_relation)

        before = evaluate_pseudo_ranking(
            judged,
            k_values=[self_training.target_k],
            reward_threshold=self_training.reward_threshold,
        )
        tuned_weights, search = tune_weights_from_judge(
            judged,
            current_weights,
            self_training,
            desired_relation=desired_relation,
        )
        tuned_ranked = rank_with_weights(judged, tuned_weights)
        after = evaluate_pseudo_ranking(
            tuned_ranked,
            k_values=[self_training.target_k],
            reward_threshold=self_training.reward_threshold,
        )

        objective_before = float(search["current_objective"].iloc[0]) if not search.empty else _metric(before)
        objective_after = float(search["objective"].iloc[0]) if not search.empty else _metric(after)
        improvement = objective_after - objective_before
        row = {
            "iteration": iteration,
            "survivors": len(survivors),
            "targets": survivors["target_paper_id"].nunique(),
            "scored_pairs_total": 0 if relation_scores is None else len(relation_scores),
            "objective_before": objective_before,
            "objective_after": objective_after,
            "improvement": improvement,
        }
        row.update({f"weight_{name}": weights_to_dict(tuned_weights)[name] for name in weights_to_dict(tuned_weights)})
        history.append(row)
        current_weights = tuned_weights

        print(f"        objective {objective_before:.4f} -> {objective_after:.4f}; "
              f"relation weight={weights_to_dict(current_weights)['relation']:.3f}")
        if improvement < self_training.min_improvement:
            print(f"        improvement < {self_training.min_improvement}; stopping self-training early.")
            break

    # Rerun retrieval once with the learned weights so the final candidate set
    # reflects what the loop learned.
    print("      final pass with learned weights")
    final_survivors, final_text_model = _retrieve_survivors(pairs, papers, cfg, args, current_weights)
    relation_scores = _score_missing_survivors(
        final_survivors, relation_scores, desired_relation, client
    )
    write_table(relation_scores, cfg.path("paths.relation_scores"))

    ranked = finalize_ranking(
        final_survivors,
        relation_scores=relation_scores,
        desired_relation=desired_relation,
        weights=current_weights,
        mmr_lambda=cfg.get("ranking.mmr_lambda", 0.75),
        text_model=final_text_model,
    )
    if history:
        write_table(pd.DataFrame(history), cfg.base_dir / "outputs" / "self_training_metrics.csv")
    return ranked, relation_scores, current_weights


def _load_existing_relation_scores(cfg: ProjectConfig) -> Optional[pd.DataFrame]:
    path = cfg.path("paths.relation_scores")
    if not path.exists():
        return None
    try:
        scores = read_table(path)
    except Exception:
        return None
    return scores if not scores.empty else None


def _score_missing_survivors(
    survivors: pd.DataFrame,
    existing_scores: Optional[pd.DataFrame],
    desired_relation: str,
    client: LLMClient,
) -> pd.DataFrame:
    keys = ["target_paper_id", "candidate_paper_id"]
    if existing_scores is None or existing_scores.empty:
        new_scores = _score_survivors(survivors, desired_relation, client)
        return merge_relation_score_frames(existing_scores, new_scores)

    scored_keys = existing_scores[keys].drop_duplicates().copy()
    scored_keys["__scored"] = 1
    marked = survivors.merge(scored_keys, on=keys, how="left")
    missing = marked[marked["__scored"].isna()].drop(columns=["__scored"])
    if missing.empty:
        print(f"        judge checkpoint: all {len(survivors)} survivors already scored")
        return existing_scores

    print(f"        judge checkpoint: scoring {len(missing)} missing / {len(survivors)} survivors")
    new_scores = _score_survivors(missing, desired_relation, client)
    return merge_relation_score_frames(existing_scores, new_scores)


def _metric(metrics: pd.DataFrame) -> float:
    if metrics.empty:
        return 0.0
    row = metrics.iloc[0]
    return float(0.70 * row["PseudoNDCG@k"] + 0.20 * row["PseudoMRR@k"] + 0.10 * row["AvgJudgeReward@k"])


def _load_weights(args, cfg: ProjectConfig) -> RankingWeights:
    """Ranking weights from config, or from a prior self-training run on request."""
    if args.use_learned_weights:
        path = cfg.base_dir / "outputs" / "learned_weights.json"
        if path.exists():
            learned = json.loads(path.read_text(encoding="utf-8"))
            print(f"[weights] using learned weights from {path.name} (relation={learned.get('relation', 0):.3f})")
            return normalize_weights(RankingWeights.from_dict(learned))
        print(f"[weights] --use-learned-weights set but {path} missing; falling back to config.")
    return normalize_weights(RankingWeights.from_dict(cfg.get("ranking.weights", {})))


def _write_learned_weights(cfg: ProjectConfig, weights: RankingWeights) -> None:
    out = cfg.base_dir / "outputs" / "learned_weights.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(weights_to_dict(weights), indent=2, sort_keys=True), encoding="utf-8")


def _write_evaluation_outputs(
    cfg: ProjectConfig,
    cleaned: pd.DataFrame,
    cleaned_edges: pd.DataFrame,
    pairs: pd.DataFrame,
    ranked: pd.DataFrame,
    desired_relation: str,
) -> None:
    metrics = evaluate_ranking(
        ranked,
        k_values=cfg.get("evaluation.k_values", [5, 10, 20]),
        desired_relation=desired_relation,
        positive_labels=cfg.get("evaluation.positive_labels", ["critique", "extension", "application"]),
    )
    write_table(metrics, cfg.path("paths.metrics"))

    pseudo_ranked = add_judge_reward(ranked, desired_relation)
    pseudo_metrics = evaluate_pseudo_ranking(
        pseudo_ranked,
        k_values=cfg.get("evaluation.k_values", [5, 10, 20]),
        reward_threshold=cfg.get("self_training.reward_threshold", 0.55),
    )
    write_table(pseudo_metrics, cfg.base_dir / "outputs" / "pseudo_metrics.csv")
    write_table(build_pseudo_qrels(ranked, desired_relation), cfg.base_dir / "outputs" / "pseudo_qrels.csv")

    rel_metrics = evaluate_relation_classification(ranked)
    if not rel_metrics.empty:
        write_table(rel_metrics, cfg.base_dir / "outputs" / "relation_classification_metrics.csv")
    write_table(dataset_statistics(cleaned, cleaned_edges, pairs), cfg.base_dir / "outputs" / "dataset_stats.csv")
    write_table(build_graph_stats(cleaned, cleaned_edges), cfg.base_dir / "outputs" / "graph_stats.csv")
    write_table(top_recommendation_examples(ranked), cfg.base_dir / "outputs" / "recommendation_examples.csv")


if __name__ == "__main__":
    main()
