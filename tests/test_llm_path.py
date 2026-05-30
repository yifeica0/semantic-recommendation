"""Offline tests for the real-LLM code path (no network, no API key needed).

Run: .venv/bin/python tests/test_llm_path.py

Monkeypatches the OpenAI client with a fake that records request params and
returns canned JSON, so we verify: reasoner param-dropping, JSON-mode gating,
disk caching, concurrent batching, relation parsing, and query parsing.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from src.llm_client import LLMClient, LLMConfig
from src.relation_scorer import parse_relation_response, score_pairs_llm
from src.query_parser import parse_query


class FakeMessage:
    def __init__(self, content):
        self.content = content


class FakeChoice:
    def __init__(self, content):
        self.message = FakeMessage(content)


class FakeResponse:
    def __init__(self, content):
        self.choices = [FakeChoice(content)]


class FakeCompletions:
    def __init__(self, recorder):
        self.recorder = recorder

    def create(self, **params):
        self.recorder.append(params)
        user = " ".join(m["content"] for m in params["messages"])
        if "Relation labels" in user:
            content = ('Here is my judgement.\n'
                       '{"label": "extension", "desired_relation_satisfied": 0.83, '
                       '"confidence": 0.9, "reason": "Proposes a faster variant of the target method."}')
        elif "structured JSON query" in user:
            content = '{"target_reference": "Attention is all you need", "desired_relation": "critique", "filters": {}, "top_k": 15}'
        else:
            content = '{}'
        return FakeResponse(content)


class FakeChat:
    def __init__(self, recorder):
        self.completions = FakeCompletions(recorder)


class FakeOpenAI:
    def __init__(self, recorder):
        self.chat = FakeChat(recorder)


def make_client(tmp: Path):
    recorder: list = []
    cfg = LLMConfig(scorer_model="deepseek-v4-pro", parser_model="deepseek-v4-flash",
                    cache_dir=str(tmp / "cache"), concurrency=4)
    client = LLMClient(cfg, base_dir=tmp)
    client.api_key = "test-key"            # pretend a key is present
    client._client = FakeOpenAI(recorder)  # inject fake transport
    return client, recorder


def test_reasoner_param_dropping(tmp: Path):
    client, rec = make_client(tmp)
    client.complete([{"role": "user", "content": "Relation labels ..."}],
                    model="deepseek-reasoner", json_mode=True)
    p = rec[-1]
    assert "temperature" not in p, "reasoner must not receive temperature"
    assert "response_format" not in p, "reasoner must not receive response_format"
    assert p["max_tokens"] > 0
    # Non-reasoner should get them.
    client.complete([{"role": "user", "content": "Relation labels ..."}],
                    model="deepseek-v4-flash", json_mode=True)
    p2 = rec[-1]
    assert "temperature" in p2 and p2.get("response_format") == {"type": "json_object"}
    print("ok: reasoner param-dropping + json-mode gating")


def test_cache(tmp: Path):
    client, rec = make_client(tmp)
    msgs = [{"role": "user", "content": "Relation labels cache-test"}]
    a = client.complete(msgs, model="deepseek-v4-pro")
    n_after_first = len(rec)
    b = client.complete(msgs, model="deepseek-v4-pro")  # should hit disk cache
    assert a == b
    assert len(rec) == n_after_first, "second identical call must hit cache (no new API call)"
    print("ok: disk cache avoids duplicate calls")


def test_parse_messy():
    pred = parse_relation_response(
        'reasoning... maybe {"label":"x"} then final '
        '{"label":"critique","desired_relation_satisfied":0.4,"confidence":0.7,"reason":"finds bias"}'
    )
    assert pred.label == "critique", pred.label
    assert abs((pred.satisfaction or 0) - 0.4) < 1e-6
    assert abs(pred.confidence - 0.7) < 1e-6
    print("ok: robust JSON extraction (last object wins, label clamped to vocab)")


def test_score_pairs_llm(tmp: Path):
    client, rec = make_client(tmp)
    pairs = pd.DataFrame([
        {"target_paper_id": "T", "candidate_paper_id": "C1", "target_title": "t", "target_abstract": "a",
         "candidate_title": "c", "candidate_abstract": "b", "edge_type": "citation", "hop": 1, "citation_path": "T|C1"},
        {"target_paper_id": "T", "candidate_paper_id": "C2", "target_title": "t", "target_abstract": "a",
         "candidate_title": "c", "candidate_abstract": "b", "edge_type": "two_hop", "hop": 2, "citation_path": "T|X|C2"},
    ])
    out = score_pairs_llm(pairs, client, desired_relation="extension")
    assert list(out.columns) == ["target_paper_id", "candidate_paper_id", "predicted_relation",
                                 "relation_confidence", "relation_satisfaction", "relation_reason"]
    assert (out["predicted_relation"] == "extension").all()
    assert abs(out["relation_satisfaction"].iloc[0] - 0.83) < 1e-6
    print("ok: score_pairs_llm parses label + satisfaction + reason for all pairs")


def test_query_parser(tmp: Path):
    client, rec = make_client(tmp)
    papers = pd.DataFrame([
        {"paperId": "a" * 40, "title": "Attention is All You Need", "abstract": "transformer"},
        {"paperId": "b" * 40, "title": "Deep Residual Learning", "abstract": "resnet"},
    ])
    parsed = parse_query("which papers challenge attention is all you need", papers, client)
    assert parsed.desired_relation == "critique", parsed.desired_relation
    assert parsed.target_paper_id == "a" * 40, parsed.target_paper_id
    assert parsed.parser_backend == "llm"
    print(f"ok: query parser -> relation={parsed.desired_relation} target={parsed.target_title!r} match={parsed.match_score:.2f}")


def main():
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        test_reasoner_param_dropping(tmp)
        test_cache(tmp)
        test_parse_messy()
        test_score_pairs_llm(tmp)
        test_query_parser(tmp)
    print("\nALL OFFLINE LLM-PATH TESTS PASSED")


if __name__ == "__main__":
    main()
