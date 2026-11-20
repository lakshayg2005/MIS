import pytest

from eval import metrics as m


def test_retrieval_metrics_known_values():
    rel = ["3.1"]
    assert m.recall_at_k(rel, ["4.1", "3.1", "2.1"], 3) == 1.0
    assert m.recall_at_k(rel, ["4.1", "3.1", "2.1"], 1) == 0.0
    assert m.precision_at_k(rel, ["4.1", "3.1", "2.1"], 3) == pytest.approx(1 / 3)
    assert m.reciprocal_rank(rel, ["4.1", "3.1"]) == 0.5
    assert m.reciprocal_rank(rel, ["4.1"]) == 0.0
    assert m.ndcg_at_k(rel, ["3.1", "x"], 3) == 1.0
    assert m.ndcg_at_k(rel, ["x", "3.1"], 3) == pytest.approx(1 / 1.5849625, rel=1e-4)  # 1/log2(3)


def test_multi_relevant_recall_and_ndcg():
    rel = ["6.2", "6.3"]
    assert m.recall_at_k(rel, ["6.2", "x", "y"], 3) == 0.5
    assert m.ndcg_at_k(rel, ["6.2", "6.3", "x"], 3) == 1.0
    assert 0 < m.ndcg_at_k(rel, ["6.2", "x", "6.3"], 3) < 1.0
