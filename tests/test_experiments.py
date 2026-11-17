from collections.abc import Iterable

import pytest

import app.experiments as experiments
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


def test_evaluates_folds_with_train_only_and_known_relevant_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    splits = make_splits()
    builder = experiments.build_item_based_model
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

    monkeypatch.setattr(experiments, "build_item_based_model", build_spy)

    result = experiments.evaluate_item_based_folds(
        splits,
        user_ids={1, 2, 3},
        item_ids={10, 20, 30, 40},
        genres_by_item={
            10: ["Drama"],
            20: ["Drama"],
            30: ["Sci-Fi"],
            40: [],
        },
    )

    assert received_training_data == [split.train for split in splits]
    assert all(
        test_rating not in passed_train
        for split, passed_train in zip(splits, received_training_data, strict=True)
        for test_rating in split.test
    )
    assert [fold.fold for fold in result.folds] == [1, 2, 3, 4, 5]
    for fold in result.folds:
        assert fold.users_evaluated == 3
        assert fold.users_without_relevant == 1
        assert fold.recommendations_generated == 2
        assert fold.precision_at_k == pytest.approx(2 / 30)
        assert fold.recall_at_k == 1.0
        assert fold.ndcg_at_k == 1.0
        assert fold.distinct_items_recommended == 1
        assert fold.catalog_size == 4
        assert fold.catalog_coverage == 0.25
        assert fold.intra_list_diversity is None
        assert fold.novelty == pytest.approx(2.807354922057604)
        assert fold.recommendations_without_training_ratings == 0
    assert result.mean_precision_at_k == pytest.approx(2 / 30)
    assert result.mean_recall_at_k == 1.0
    assert result.mean_ndcg_at_k == 1.0
    assert result.mean_catalog_coverage == 0.25
    assert result.mean_intra_list_diversity is None
    assert result.mean_novelty == pytest.approx(2.807354922057604)


def test_requires_exactly_five_official_folds() -> None:
    with pytest.raises(ValueError, match="exactly one split.*fold 1 through 5"):
        experiments.evaluate_item_based_folds(
            make_splits()[:4],
            user_ids={1, 2, 3},
            item_ids={10, 20, 30, 40},
            genres_by_item={10: [], 20: [], 30: [], 40: []},
        )


@pytest.mark.parametrize(
    ("k", "threshold", "message"),
    [(0, 4, "K must be positive"), (10, 6, "threshold must be between 1 and 5")],
)
def test_rejects_invalid_experiment_parameters(
    k: int, threshold: int, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        experiments.evaluate_item_based_folds(
            make_splits(),
            user_ids={1, 2, 3},
            item_ids={10, 20, 30, 40},
            genres_by_item={10: [], 20: [], 30: [], 40: []},
            k=k,
            relevance_threshold=threshold,
        )
