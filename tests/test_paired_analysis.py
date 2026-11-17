import pytest

import app.paired_analysis as paired_analysis
from app.movielens import MovieLensRating
from app.movielens_splits import MovieLensSplit


def rating(user_id: int, item_id: int, score: int) -> MovieLensRating:
    return MovieLensRating(user_id, item_id, score, 0)


def make_splits() -> tuple[MovieLensSplit, ...]:
    train = (
        rating(1, 1, 5),
        rating(1, 2, 1),
        rating(2, 1, 1),
        rating(2, 2, 5),
    )
    tests = (
        (),
        (rating(2, 4, 3),),
        (rating(1, 3, 4), rating(2, 4, 2)),
        (rating(1, 4, 5),),
        (rating(2, 3, 4),),
    )
    return tuple(
        MovieLensSplit(fold, train, test)
        for fold, test in enumerate(tests, start=1)
    )


def test_bootstrap_is_reproducible_for_same_seed() -> None:
    first = paired_analysis.bootstrap_paired_difference(
        [0.0, 0.2, 0.8, 1.0],
        [0.1, 0.4, 0.7, 0.9],
        iterations=500,
        seed=123,
    )
    second = paired_analysis.bootstrap_paired_difference(
        [0.0, 0.2, 0.8, 1.0],
        [0.1, 0.4, 0.7, 0.9],
        iterations=500,
        seed=123,
    )

    assert first == second


def test_bootstrap_resamples_paired_differences_together() -> None:
    estimate = paired_analysis.bootstrap_paired_difference(
        [0.0, 10.0], [1.0, 11.0], iterations=100, seed=42
    )

    assert estimate.users == 2
    assert estimate.mean_difference == 1.0
    assert estimate.confidence_interval_lower == 1.0
    assert estimate.confidence_interval_upper == 1.0


def test_bootstrap_returns_zero_interval_for_identical_models() -> None:
    estimate = paired_analysis.bootstrap_paired_difference(
        [0.2, 0.5, 1.0], [0.2, 0.5, 1.0], iterations=200, seed=7
    )

    assert estimate.mean_difference == 0.0
    assert estimate.confidence_interval_lower == 0.0
    assert estimate.confidence_interval_upper == 0.0


def test_bootstrap_empty_input_and_invalid_pairs() -> None:
    assert paired_analysis.bootstrap_paired_difference([], []).users == 0
    with pytest.raises(ValueError, match="same length"):
        paired_analysis.bootstrap_paired_difference([0.1], [])
    with pytest.raises(ValueError, match="must be positive"):
        paired_analysis.bootstrap_paired_difference([], [], iterations=0)


def test_fold_evaluation_skips_recall_and_ndcg_without_relevant_users() -> None:
    result = paired_analysis.evaluate_paired_item_based_folds(
        make_splits(), user_ids={1, 2}, item_ids={1, 2, 3, 4},
        bootstrap_iterations=50,
    )
    by_fold_metric = {
        (row.fold, row.metric): row for row in result.fold_results
    }

    assert by_fold_metric[(1, "precision_at_10")].users == 0
    assert by_fold_metric[(1, "precision_at_10")].mean_difference is None
    assert by_fold_metric[(2, "precision_at_10")].users == 1
    assert by_fold_metric[(2, "recall_at_10")].users == 0
    assert by_fold_metric[(2, "recall_at_10")].mean_difference is None
    assert by_fold_metric[(2, "ndcg_at_10")].mean_difference is None
    assert by_fold_metric[(3, "precision_at_10")].users == 2
    assert by_fold_metric[(3, "recall_at_10")].users == 1
    assert by_fold_metric[(3, "ndcg_at_10")].users == 1
    assert result.bootstrap_iterations == 50


def test_macro_difference_is_equal_weight_mean_over_eligible_folds() -> None:
    rows = [
        paired_analysis.PairedFoldMetric(1, "precision_at_10", 0.2, 0.0, 0.4, 2),
        paired_analysis.PairedFoldMetric(2, "precision_at_10", None, None, None, 0),
        paired_analysis.PairedFoldMetric(3, "precision_at_10", 0.8, 0.5, 1.0, 1),
    ]

    macro = paired_analysis._macro_results(rows)[0]

    assert macro.folds == 3
    assert macro.folds_with_users == 2
    assert macro.mean_difference == pytest.approx(0.5)
    assert macro.users_by_fold == ((1, 2), (2, 0), (3, 1))
