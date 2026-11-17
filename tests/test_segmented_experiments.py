from collections.abc import Iterable

import pytest

import app.segmented_experiments as segmented_experiments
from app.collaborative_filtering import ItemBasedModel
from app.movielens import MovieLensRating
from app.movielens_splits import MovieLensSplit


def rating(user_id: int, item_id: int, score: int) -> MovieLensRating:
    return MovieLensRating(user_id, item_id, score, 0)


def make_splits() -> tuple[MovieLensSplit, ...]:
    train: list[MovieLensRating] = []
    for user_id, count in ((1, 5), (2, 6), (3, 11), (4, 21)):
        train.extend(rating(user_id, 100 + index, 5) for index in range(count))
    test = tuple(
        rating(user_id, 200 + user_id, 4)
        for user_id in (1, 2, 3, 4, 5)
    )
    return tuple(MovieLensSplit(fold, tuple(train), test) for fold in range(1, 6))


def test_history_segments_use_training_counts_and_existing_evaluation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    splits = make_splits()
    builder = segmented_experiments.build_item_based_model
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

    monkeypatch.setattr(
        segmented_experiments, "build_item_based_model", build_spy
    )

    results = segmented_experiments.evaluate_by_training_history(
        splits,
        user_ids={1, 2, 3, 4, 5},
        item_ids=set(range(100, 130)) | set(range(201, 206)),
        k=2,
    )

    assert received_training_data == [split.train for split in splits]
    assert all(
        test_rating not in train
        for split, train in zip(splits, received_training_data, strict=True)
        for test_rating in split.test
    )
    assert results.users_without_training_history_by_fold == tuple(
        (fold, 1) for fold in range(1, 6)
    )
    assert [result.fold for result in results.fold_results] == [
        fold
        for fold in range(1, 6)
        for _ in range(3 * 4)
    ]

    by_segment = {
        result.history_segment: result
        for result in results.fold_results
        if result.fold == 1 and result.model_name == "item_based_centered"
    }
    assert {segment: result.users for segment, result in by_segment.items()} == {
        "1-5": 1,
        "6-10": 1,
        "11-20": 1,
        "21+": 1,
    }
    assert all(result.catalog_size == 35 for result in results.fold_results)
    assert len(results.macro_results) == 12
    assert all(result.folds == 5 for result in results.macro_results)
    assert all(result.folds_with_users == 5 for result in results.macro_results)


def test_history_segment_boundaries() -> None:
    assert segmented_experiments._history_segment(0) is None
    assert segmented_experiments._history_segment(1) == "1-5"
    assert segmented_experiments._history_segment(5) == "1-5"
    assert segmented_experiments._history_segment(6) == "6-10"
    assert segmented_experiments._history_segment(10) == "6-10"
    assert segmented_experiments._history_segment(11) == "11-20"
    assert segmented_experiments._history_segment(20) == "11-20"
    assert segmented_experiments._history_segment(21) == "21+"


def test_segmented_experiment_requires_five_folds() -> None:
    with pytest.raises(ValueError, match="exactly one split.*fold 1 through 5"):
        segmented_experiments.evaluate_by_training_history(
            make_splits()[:4], user_ids={1, 2, 3, 4, 5}, item_ids={101}
        )
