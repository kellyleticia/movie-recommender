from collections.abc import Iterable

import pytest

import app.variant_experiments as variant_experiments
from app.collaborative_filtering import ItemBasedModel
from app.movielens import MovieLensRating
from app.movielens_splits import MovieLensSplit


def rating(user_id: int, item_id: int, score: int) -> MovieLensRating:
    return MovieLensRating(user_id, item_id, score, 0)


def make_splits() -> tuple[MovieLensSplit, ...]:
    train = (
        rating(1, 10, 5),
        rating(1, 20, 1),
        rating(2, 10, 1),
        rating(2, 20, 5),
        rating(3, 10, 5),
        rating(3, 20, 1),
        rating(3, 30, 5),
    )
    test = (
        rating(1, 30, 5),
        rating(1, 40, 2),
        rating(2, 30, 4),
        rating(2, 40, 1),
        rating(3, 40, 3),
    )
    return tuple(MovieLensSplit(fold, train, test) for fold in range(1, 6))


def test_variant_experiment_uses_only_train_and_same_metrics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    splits = make_splits()
    builder = variant_experiments.build_item_based_model
    received_training_data: list[tuple[MovieLensRating, ...]] = []

    def build_spy(
        train_ratings: Iterable[MovieLensRating],
        *,
        user_ids: Iterable[int] = (),
        item_ids: Iterable[int] = (),
    ) -> ItemBasedModel:
        materialized = tuple(train_ratings)
        received_training_data.append(materialized)
        return builder(materialized, user_ids=user_ids, item_ids=item_ids)

    monkeypatch.setattr(variant_experiments, "build_item_based_model", build_spy)

    results = variant_experiments.evaluate_item_based_variants(
        splits,
        user_ids={1, 2, 3},
        item_ids={10, 20, 30, 40},
        genres_by_item={10: ["Drama"], 20: ["Drama"], 30: ["Sci-Fi"], 40: []},
    )

    assert received_training_data == [split.train for split in splits]
    assert all(
        test_rating not in train
        for split, train in zip(splits, received_training_data, strict=True)
        for test_rating in split.test
    )
    assert [result.variant_name for result in results] == [
        "centered_all_neighbors",
        "uncentered_all_neighbors",
        "centered_top_20_neighbors",
        "centered_top_50_neighbors",
        "centered_top_100_neighbors",
    ]
    for result in results:
        assert [fold.fold for fold in result.evaluation.folds] == [1, 2, 3, 4, 5]
        assert result.evaluation.folds[0].catalog_size == 4
        assert result.evaluation.folds[0].users_evaluated == 3
        assert result.evaluation.folds[0].users_without_relevant == 1
    assert results[0].evaluation.mean_precision_at_k == pytest.approx(2 / 30)


def test_variant_experiment_rejects_invalid_fold_set() -> None:
    with pytest.raises(ValueError, match="exactly one split.*fold 1 through 5"):
        variant_experiments.evaluate_item_based_variants(
            make_splits()[:4],
            user_ids={1, 2, 3},
            item_ids={10, 20, 30, 40},
            genres_by_item={10: [], 20: [], 30: [], 40: []},
        )
