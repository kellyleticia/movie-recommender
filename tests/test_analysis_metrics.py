from math import log2

import pytest

from app.analysis_metrics import (
    catalog_coverage,
    intra_list_diversity,
    recommendation_novelty,
)
from app.movielens import MovieLensRating


def rating(user_id: int, item_id: int) -> MovieLensRating:
    return MovieLensRating(user_id, item_id, 4, 0)


def test_catalog_coverage_counts_distinct_ids_across_users() -> None:
    result = catalog_coverage(
        {1: [10, 10, 20], 2: [20, 30]},
        {10, 20, 30, 40},
    )

    assert result.value == 0.75
    assert result.distinct_items_recommended == 3
    assert result.catalog_size == 4


def test_catalog_coverage_for_empty_catalog_is_zero() -> None:
    assert catalog_coverage({}, set()).value == 0.0


def test_catalog_coverage_rejects_recommendations_outside_catalog() -> None:
    with pytest.raises(ValueError, match="outside the catalog"):
        catalog_coverage({1: [99]}, {10, 20})


def test_intra_list_diversity_uses_jaccard_and_macro_averages_users() -> None:
    result = intra_list_diversity(
        {"first": [1, 2, 3], "second": [4, 5], "single": [6]},
        {
            1: ["Drama", "Comedy"],
            2: ["Drama"],
            3: ["Sci-Fi"],
            4: ["SCI-FI"],
            5: ["sci-fi"],
        },
    )

    assert result.value == pytest.approx((5 / 6) / 2)
    assert result.users_evaluated == 2
    assert result.users_without_comparable_pairs == 0
    assert result.pairs_evaluated == 4
    assert result.pairs_skipped_missing_genres == 0


def test_intra_list_diversity_skips_pairs_with_missing_genres() -> None:
    result = intra_list_diversity(
        {"partial": [1, 2, 3], "unknown": [4, 5], "single": [6]},
        {1: [], 2: ["Drama"], 3: ["Sci-Fi"], 4: [], 5: []},
    )

    assert result.value == 1.0
    assert result.users_evaluated == 1
    assert result.users_without_comparable_pairs == 1
    assert result.pairs_evaluated == 1
    assert result.pairs_skipped_missing_genres == 3


def test_intra_list_diversity_is_undefined_for_empty_or_single_item_lists() -> None:
    result = intra_list_diversity({"empty": [], "one": [1]}, {1: ["Drama"]})

    assert result.value is None
    assert result.users_evaluated == 0
    assert result.users_without_comparable_pairs == 0
    assert result.pairs_evaluated == 0


def test_novelty_uses_training_rating_counts() -> None:
    result = recommendation_novelty(
        {1: [10, 20, 20]},
        [rating(1, 10), rating(2, 10), rating(1, 20)],
    )

    assert result.value == pytest.approx((log2(3 / 2) + log2(3) * 2) / 3)
    assert result.recommendations_count == 3
    assert result.recommendations_without_training_ratings == 0


def test_novelty_for_zero_count_items_is_positive_infinity() -> None:
    result = recommendation_novelty(
        {1: [99]}, [rating(1, 10), rating(2, 10)]
    )

    assert result.value == float("inf")
    assert result.recommendations_count == 1
    assert result.recommendations_without_training_ratings == 1


def test_novelty_is_undefined_without_training_ratings_or_recommendations() -> None:
    no_training = recommendation_novelty({1: [99]}, [])
    no_recommendations = recommendation_novelty({}, [rating(1, 10)])

    assert no_training.value is None
    assert no_training.recommendations_without_training_ratings == 1
    assert no_recommendations.value is None
    assert no_recommendations.recommendations_count == 0
