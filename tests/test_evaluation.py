from math import log2

import pytest

from app.evaluation import (
    UserMetricAggregate,
    aggregate_user_metrics,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)


def test_precision_at_k_counts_unique_relevant_recommendations() -> None:
    assert precision_at_k(["a", "a", "b"], {"a", "b"}, 2) == 1.0
    assert precision_at_k(["a", "x", "b"], {"a", "b"}, 2) == 0.5


def test_precision_at_k_returns_zero_for_empty_recommendations() -> None:
    assert precision_at_k([], {"a"}, 3) == 0.0


def test_recall_at_k_uses_unique_hits_and_returns_zero_for_empty_ranking() -> None:
    assert recall_at_k(["a", "a", "x"], {"a", "b"}, 3) == 0.5
    assert recall_at_k([], {"a", "b"}, 3) == 0.0


def test_recall_is_undefined_without_relevant_items() -> None:
    assert recall_at_k(["a"], set(), 3) is None


def test_ndcg_at_k_uses_binary_gain_and_logarithmic_discount() -> None:
    expected = (1 / log2(3) + 1 / log2(4)) / (1 + 1 / log2(3))

    assert ndcg_at_k(["x", "a", "b"], {"a", "b"}, 3) == pytest.approx(
        expected
    )


def test_ndcg_deduplicates_before_assigning_ranks() -> None:
    expected = 1.5 / (1 + 1 / log2(3))

    assert ndcg_at_k(["a", "a", "x", "b"], {"a", "b"}, 3) == pytest.approx(
        expected
    )


def test_ndcg_is_zero_for_empty_ranking_and_undefined_without_relevant_items() -> None:
    assert ndcg_at_k([], {"a"}, 3) == 0.0
    assert ndcg_at_k(["a"], set(), 3) is None


@pytest.mark.parametrize(
    "metric",
    [precision_at_k, recall_at_k, ndcg_at_k],
)
def test_metrics_reject_non_positive_k(metric) -> None:
    with pytest.raises(ValueError, match="K must be a positive integer"):
        metric(["a"], {"a"}, 0)


def test_aggregates_per_user_and_counts_users_without_relevant_items() -> None:
    result = aggregate_user_metrics(
        recommendations_by_user={
            "complete": ["a", "b"],
            "no-relevant": ["x"],
            "missed": [],
        },
        relevant_items_by_user={
            "complete": {"a", "b"},
            "no-relevant": set(),
            "missed": {"c", "d"},
        },
        k=2,
    )

    assert result == UserMetricAggregate(
        precision_at_k=1 / 3,
        recall_at_k=0.5,
        ndcg_at_k=0.5,
        user_count=3,
        users_without_relevant=1,
    )


def test_empty_aggregation_reports_undefined_means() -> None:
    assert aggregate_user_metrics({}, {}, 5) == UserMetricAggregate(
        precision_at_k=None,
        recall_at_k=None,
        ndcg_at_k=None,
        user_count=0,
        users_without_relevant=0,
    )
