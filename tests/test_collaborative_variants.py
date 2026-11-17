import pytest
from scipy.sparse import csr_matrix

from app.collaborative_filtering import (
    ItemBasedModel,
    build_item_based_model,
    recommend_for_user,
)
from app.collaborative_variants import (
    build_item_based_variants,
    build_uncentered_item_model,
    recommend_with_neighbor_limit,
    recommend_with_variant,
)
from app.movielens import MovieLensRating


def rating(user_id: int, item_id: int, value: int) -> MovieLensRating:
    return MovieLensRating(user_id, item_id, value, 0)


def test_uncentered_variant_uses_raw_rating_cosine() -> None:
    model = build_item_based_model(
        [rating(1, 10, 5), rating(1, 20, 1), rating(2, 10, 1), rating(2, 20, 5)]
    )

    uncentered = build_uncentered_item_model(model)

    assert model.item_similarity[0, 1] == pytest.approx(-1.0)
    assert uncentered.item_similarity[0, 1] == pytest.approx(5 / 13)
    variant = build_item_based_variants(model, ())[1]
    assert recommend_with_variant(1, variant) == recommend_for_user(1, uncentered)


def test_limited_neighbors_select_highest_positive_similarities_per_candidate() -> None:
    model = ItemBasedModel(
        user_ids=(1,),
        item_ids=(10, 20, 30, 40),
        user_index={1: 0},
        item_index={10: 0, 20: 1, 30: 2, 40: 3},
        user_item_matrix=csr_matrix([[5.0, 4.0, 3.0, 0.0]]),
        item_similarity=csr_matrix(
            [
                [1.0, 0.0, 0.0, 0.9],
                [0.0, 1.0, 0.0, 0.8],
                [0.0, 0.0, 1.0, 0.1],
                [0.9, 0.8, 0.1, 1.0],
            ]
        ),
    )

    one_neighbor = recommend_with_neighbor_limit(
        1, model, neighbor_limit=1
    )
    two_neighbors = recommend_with_neighbor_limit(
        1, model, neighbor_limit=2
    )
    all_neighbors = recommend_with_neighbor_limit(
        1, model, neighbor_limit=3
    )

    assert [(item.item_id, item.score) for item in one_neighbor] == [(40, 4.5)]
    assert [(item.item_id, item.score) for item in two_neighbors] == [(40, 7.7)]
    assert [(item.item_id, item.score) for item in all_neighbors] == [(40, 8.0)]
    assert all_neighbors == recommend_for_user(1, model)


def test_limited_neighbor_ties_and_recommendation_ties_are_deterministic() -> None:
    model = ItemBasedModel(
        user_ids=(1,),
        item_ids=(10, 20, 30, 40),
        user_index={1: 0},
        item_index={10: 0, 20: 1, 30: 2, 40: 3},
        user_item_matrix=csr_matrix([[5.0, 5.0, 0.0, 0.0]]),
        item_similarity=csr_matrix(
            [
                [1.0, 0.0, 0.8, 0.8],
                [0.0, 1.0, 0.8, 0.8],
                [0.8, 0.8, 1.0, 0.0],
                [0.8, 0.8, 0.0, 1.0],
            ]
        ),
    )

    recommendations = recommend_with_neighbor_limit(
        1, model, neighbor_limit=1
    )

    assert [item.item_id for item in recommendations] == [30, 40]
    assert recommendations[0].score == recommendations[1].score == 4.0


def test_variant_factory_preserves_reference_and_adds_requested_limits() -> None:
    model = build_item_based_model([rating(1, 10, 4), rating(2, 20, 5)])

    variants = build_item_based_variants(model, (20, 50, 100))

    assert [variant.name for variant in variants] == [
        "centered_all_neighbors",
        "uncentered_all_neighbors",
        "centered_top_20_neighbors",
        "centered_top_50_neighbors",
        "centered_top_100_neighbors",
    ]
    assert variants[0].model is model
    assert variants[2].model is model
    assert variants[1].model.user_item_matrix is model.user_item_matrix


def test_rejects_non_positive_neighbor_limit() -> None:
    model = build_item_based_model([rating(1, 10, 4)])

    with pytest.raises(ValueError, match="Neighbor limit must be positive"):
        recommend_with_neighbor_limit(1, model, neighbor_limit=0)


def test_uncentered_empty_model_has_zero_similarity_matrix() -> None:
    model = build_item_based_model([], item_ids=[10, 20])

    uncentered = build_uncentered_item_model(model)

    assert uncentered.item_similarity.shape == (2, 2)
    assert uncentered.item_similarity.nnz == 0
