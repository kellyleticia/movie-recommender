from collections.abc import Collection
from dataclasses import dataclass
from pathlib import Path

from app.movielens import MovieLensRating, load_movielens_ratings


@dataclass(frozen=True)
class MovieLensSplit:
    """One official MovieLens 100K fold with separate train and test ratings."""

    fold: int
    train: tuple[MovieLensRating, ...]
    test: tuple[MovieLensRating, ...]


def load_movielens_splits(
    data_dir: str | Path,
    *,
    expected_user_ids: Collection[int],
    expected_item_ids: Collection[int],
) -> tuple[MovieLensSplit, ...]:
    """Load u1 through u5 folds and validate IDs and train/test separation.

    The expected ID collections should come from the dataset's user and movie
    metadata. Ratings retain their source order and are not otherwise altered.
    """
    root = Path(data_dir)
    valid_user_ids = frozenset(expected_user_ids)
    valid_item_ids = frozenset(expected_item_ids)
    splits: list[MovieLensSplit] = []

    for fold in range(1, 6):
        train_path = root / f"u{fold}.base"
        test_path = root / f"u{fold}.test"
        train = load_movielens_ratings(train_path)
        test = load_movielens_ratings(test_path)

        _validate_expected_ids(train, train_path, valid_user_ids, valid_item_ids)
        _validate_expected_ids(test, test_path, valid_user_ids, valid_item_ids)

        train_pairs = {(rating.user_id, rating.item_id) for rating in train}
        test_pairs = {(rating.user_id, rating.item_id) for rating in test}
        overlap = train_pairs & test_pairs
        if overlap:
            user_id, item_id = min(overlap)
            raise ValueError(
                f"Data leakage in fold {fold}: user {user_id}, item {item_id} "
                "appears in both train and test"
            )

        splits.append(MovieLensSplit(fold, tuple(train), tuple(test)))

    return tuple(splits)


def _validate_expected_ids(
    ratings: list[MovieLensRating],
    path: Path,
    expected_user_ids: frozenset[int],
    expected_item_ids: frozenset[int],
) -> None:
    for row_number, rating in enumerate(ratings, start=1):
        if rating.user_id not in expected_user_ids:
            raise ValueError(
                f"Unexpected user ID {rating.user_id} in row {row_number} "
                f"of '{path}'"
            )
        if rating.item_id not in expected_item_ids:
            raise ValueError(
                f"Unexpected item ID {rating.item_id} in row {row_number} "
                f"of '{path}'"
            )
