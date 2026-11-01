import math

import pytest

from ragspec.metrics import generation_metrics, ndcg_at_k, percentile, retrieval_metrics
from ragspec.models import Claim, JudgeResult


def test_retrieval_metrics_basic():
    m = retrieval_metrics(["a", "x", "b", "y"], {"a": 1, "b": 1, "c": 1}, [1, 3])
    assert m["recall@1"] == pytest.approx(1 / 3)
    assert m["recall@3"] == pytest.approx(2 / 3)
    assert m["mrr"] == 1.0


def test_retrieval_metrics_no_labels_is_none():
    m = retrieval_metrics(["a"], {}, [5])
    assert all(v is None for v in m.values())


def test_duplicates_do_not_inflate_recall():
    m = retrieval_metrics(["a", "a", "a"], {"a": 1, "b": 1}, [3])
    assert m["recall@3"] == 0.5


def test_mrr_first_relevant_rank():
    m = retrieval_metrics(["x", "y", "b"], {"b": 1}, [5])
    assert m["mrr"] == pytest.approx(1 / 3)


def test_ndcg_matches_trec_definition():
    # Graded: a=2, b=1. Retrieved b, a.
    dcg = 1 / math.log2(2) + 2 / math.log2(3)
    idcg = 2 / math.log2(2) + 1 / math.log2(3)
    assert ndcg_at_k(["b", "a"], {"a": 2, "b": 1}, 10) == pytest.approx(dcg / idcg)
    assert ndcg_at_k(["a", "b"], {"a": 2, "b": 1}, 10) == pytest.approx(1.0)


def claim(verdict, cited=(), supporting=()):
    return Claim(text="c", verdict=verdict, cited_ids=list(cited), supporting_ids=list(supporting))


def test_generation_metrics():
    j = JudgeResult(model="m", claims=[
        claim("supported", cited=["d1"], supporting=["d1"]),          # correct citation
        claim("supported", cited=["d2"], supporting=["d1"]),          # wrong citation
        claim("supported", supporting=["d3"]),                        # uncited
        claim("unsupported", cited=["d1"]),                           # cites, unsupported
    ])
    m = generation_metrics(j)
    assert m["faithfulness"] == 0.75
    assert m["citation_precision"] == pytest.approx(1 / 3)
    assert m["citation_recall"] == pytest.approx(1 / 3)


def test_generation_metrics_no_citations():
    m = generation_metrics(JudgeResult(model="m", claims=[claim("supported", supporting=["a"])]))
    assert m["citation_precision"] is None
    assert m["citation_recall"] == 0.0


def test_generation_metrics_abstain_or_error_is_none():
    for judge in [JudgeResult(model="m", abstained=True), JudgeResult(model="m", error="x")]:
        assert all(v is None for v in generation_metrics(judge).values())


def test_percentile():
    assert percentile([1, 2, 3, 4], 50) == 2.5
    assert percentile([5], 95) == 5
    assert percentile([], 50) is None
