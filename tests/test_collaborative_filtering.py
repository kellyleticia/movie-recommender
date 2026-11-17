from math import sqrt

import pytest
from scipy.sparse import csr_matrix

from app.collaborative_filtering import (
    CollaborativeRecommendation,
    build_item_based_model,
    recommend_for_user,
)
from app.movielens import MovieLensRating


def rating(user_id: int, item_id: int, value: int) -> MovieLensRating:
    return MovieLensRating(user_id, item_id, value, 0)


def recommendation_fixture_ratings() -> list[MovieLensRating]:
    return [
        rating(1, 10, 5),
        rating(1, 20, 1),
        rating(2, 10, 1),
        rating(2, 20, 5),
        rating(3, 10, 5),
        rating(3, 20, 1),
        rating(3, 30, 5),
    ]


def test_builds_sparse_matrices_with_sorted_explicit_indexes() -> None:
    model = build_item_based_model(
        [rating(20, 30, 4), rating(10, 20, 2)],
        user_ids=[30],
        item_ids=[40],
    )

    assert model.user_ids == (10, 20, 30)
    assert model.item_ids == (20, 30, 40)
    assert model.user_index == {10: 0, 20: 1, 30: 2}
    assert model.item_index == {20: 0, 30: 1, 40: 2}
    assert isinstance(model.user_item_matrix, csr_matrix)
    assert model.user_item_matrix.shape == (3, 3)
    assert model.user_item_matrix[0, 0] == 2
    assert model.user_item_matrix[1, 1] == 4
    assert model.user_item_matrix[2].nnz == 0
    assert model.item_similarity.shape == (3, 3)
    assert model.item_similarity[2].nnz == 0


def test_computes_cosine_from_user_mean_centered_item_vectors() -> None:
    model = build_item_based_model(recommendation_fixture_ratings())

    # The centered movie-10 vector is [2, -2, 4/3]; movie 30 is [0, 0, 4/3].
    assert model.item_similarity[
        model.item_index[10], model.item_index[30]
    ] == pytest.approx(2 / sqrt(22))


def test_recommends_by_weighted_scores_excluding_rated_items() -> None:
    model = build_item_based_model(
        recommendation_fixture_ratings(), item_ids=[40]
    )

    recommendations = recommend_for_user(1, model)

    assert recommendations == [
        CollaborativeRecommendation(item_id=30, score=10 / sqrt(22))
    ]


def test_breaks_equal_scores_by_ascending_item_id() -> None:
    ratings = [
        rating(1, 10, 5),
        rating(1, 20, 1),
        rating(2, 10, 1),
        rating(2, 20, 5),
        rating(3, 10, 5),
        rating(3, 20, 1),
        rating(3, 30, 5),
        rating(3, 40, 5),
    ]
    model = build_item_based_model(ratings)

    recommendations = recommend_for_user(1, model)

    assert [item.item_id for item in recommendations] == [30, 40]
    assert recommendations[0].score == pytest.approx(recommendations[1].score)


def test_handles_unknown_and_unrated_users_and_movies() -> None:
    model = build_item_based_model(
        [rating(1, 10, 4)], user_ids=[2], item_ids=[20]
    )

    assert recommend_for_user(999, model) == []
    assert recommend_for_user(2, model) == []
    assert recommend_for_user(1, model) == []


def test_empty_training_data_can_keep_expected_users_and_items() -> None:
    model = build_item_based_model([], user_ids=[1], item_ids=[10])

    assert model.user_item_matrix.shape == (1, 1)
    assert model.item_similarity.shape == (1, 1)
    assert recommend_for_user(1, model) == []


@pytest.mark.parametrize("limit", [0, -1])
def test_non_positive_limit_returns_no_recommendations(limit: int) -> None:
    model = build_item_based_model(recommendation_fixture_ratings())

    assert recommend_for_user(1, model, limit) == []


def test_rejects_duplicate_user_item_training_ratings() -> None:
    with pytest.raises(ValueError, match="Duplicate training rating.*user 1, item 10"):
        build_item_based_model([rating(1, 10, 4), rating(1, 10, 5)])
