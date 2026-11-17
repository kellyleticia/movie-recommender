from collections.abc import Iterable
from math import log2

import pytest

import app.experiments as experiments
from app.movielens import MovieLensRating
from app.movielens_splits import MovieLensSplit
from app.popularity import (
    PopularityModel,
    build_popularity_model,
    recommend_by_popularity,
)


def rating(user_id: int, item_id: int, score: int) -> MovieLensRating:
    return MovieLensRating(user_id, item_id, score, 0)


def test_orders_by_training_rating_count_then_item_id_and_excludes_seen() -> None:
    train = [
        rating(1, 10, 5),
        rating(2, 20, 4),
        rating(3, 20, 2),
        rating(2, 30, 3),
    ]
    model = build_popularity_model(train, item_ids=[40])

    recommendations = recommend_by_popularity(1, model, limit=10)

    assert [(item.item_id, item.rating_count) for item in recommendations] == [
        (20, 2),
        (30, 1),
        (40, 0),
    ]
    unseen_user_recommendations = recommend_by_popularity(99, model)
    assert [item.item_id for item in unseen_user_recommendations] == [20, 10, 30, 40]


def test_excludes_high_popularity_items_user_already_rated() -> None:
    train = [rating(1, 10, 5), rating(1, 20, 5), rating(2, 20, 4)]
    model = build_popularity_model(train, item_ids=[30])

    recommendations = recommend_by_popularity(1, model)

    assert [item.item_id for item in recommendations] == [30]


def make_splits() -> tuple[MovieLensSplit, ...]:
    train = (
        rating(1, 10, 5),
        rating(2, 20, 5),
        rating(3, 20, 4),
        rating(3, 30, 2),
    )
    test = (
        rating(1, 30, 4),
        rating(1, 40, 1),
        rating(2, 10, 5),
        rating(2, 40, 1),
        rating(3, 40, 3),
    )
    return tuple(MovieLensSplit(fold, train, test) for fold in range(1, 6))


def test_popularity_experiment_uses_only_training_ratings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    splits = make_splits()
    builder = experiments.build_popularity_model
    received_training: list[tuple[MovieLensRating, ...]] = []
    built_models: list[PopularityModel] = []

    def build_spy(
        train_ratings: Iterable[MovieLensRating], *, item_ids: Iterable[int] = ()
    ) -> PopularityModel:
        materialized = tuple(train_ratings)
        received_training.append(materialized)
        model = builder(materialized, item_ids=item_ids)
        built_models.append(model)
        return model

    monkeypatch.setattr(experiments, "build_popularity_model", build_spy)

    result = experiments.evaluate_popularity_folds(
        splits,
        item_ids={10, 20, 30, 40},
        genres_by_item={
            10: ["Drama"],
            20: ["Drama"],
            30: ["Sci-Fi"],
            40: [],
        },
        k=10,
    )

    assert received_training == [split.train for split in splits]
    assert all(
        test_rating not in train_ratings
        for split, train_ratings in zip(splits, received_training, strict=True)
        for test_rating in split.test
    )
    assert all(model.rating_counts[40] == 0 for model in built_models)
    assert [fold.fold for fold in result.folds] == [1, 2, 3, 4, 5]
    for fold in result.folds:
        assert fold.users_evaluated == 3
        assert fold.users_without_relevant == 1
        assert fold.recommendations_generated == 8
        assert fold.precision_at_k == pytest.approx(2 / 30)
        assert fold.recall_at_k == 1.0
        expected_ndcg = (1 / log2(3) + 1) / 2
        assert fold.ndcg_at_k == pytest.approx(expected_ndcg)
        assert fold.distinct_items_recommended == 4
        assert fold.catalog_size == 4
        assert fold.catalog_coverage == 1.0
        assert fold.intra_list_diversity == 1.0
        assert fold.diversity_users_evaluated == 2
        assert fold.diversity_users_without_comparable_pairs == 1
        assert fold.diversity_pairs_evaluated == 2
        assert fold.diversity_pairs_skipped_missing_genres == 5
        assert fold.novelty == float("inf")
        assert fold.recommendations_without_training_ratings == 3
    assert result.mean_precision_at_k == pytest.approx(2 / 30)
    assert result.mean_recall_at_k == 1.0
    assert result.mean_ndcg_at_k == pytest.approx((1 / log2(3) + 1) / 2)
    assert result.mean_catalog_coverage == 1.0
    assert result.mean_intra_list_diversity == 1.0
    assert result.mean_novelty == float("inf")
