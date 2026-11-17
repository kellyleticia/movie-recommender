from pathlib import Path

import pytest

from app.movielens import MovieLensRating
from app.movielens_splits import MovieLensSplit, load_movielens_splits


def write_folds(
    tmp_path: Path,
    *,
    train_records: dict[int, list[str]] | None = None,
    test_records: dict[int, list[str]] | None = None,
) -> Path:
    train_records = train_records or {}
    test_records = test_records or {}
    for fold in range(1, 6):
        (tmp_path / f"u{fold}.base").write_text(
            "\n".join(train_records.get(fold, [f"1\t{fold}\t4\t100"])) + "\n",
            encoding="ascii",
        )
        (tmp_path / f"u{fold}.test").write_text(
            "\n".join(test_records.get(fold, [f"2\t{fold + 10}\t5\t200"]))
            + "\n",
            encoding="ascii",
        )
    return tmp_path


def load_fixture_splits(data_dir: Path) -> tuple[MovieLensSplit, ...]:
    return load_movielens_splits(
        data_dir,
        expected_user_ids={1, 2},
        expected_item_ids=set(range(1, 20)),
    )


def test_loads_all_five_folds_with_separate_ordered_ratings(tmp_path: Path) -> None:
    data_dir = write_folds(
        tmp_path,
        train_records={1: ["1\t2\t3\t10", "1\t3\t4\t11"]},
        test_records={1: ["2\t4\t5\t12", "2\t5\t1\t13"]},
    )

    splits = load_fixture_splits(data_dir)

    assert len(splits) == 5
    assert [split.fold for split in splits] == [1, 2, 3, 4, 5]
    assert splits[0].train == (
        MovieLensRating(user_id=1, item_id=2, rating=3, timestamp=10),
        MovieLensRating(user_id=1, item_id=3, rating=4, timestamp=11),
    )
    assert splits[0].test == (
        MovieLensRating(user_id=2, item_id=4, rating=5, timestamp=12),
        MovieLensRating(user_id=2, item_id=5, rating=1, timestamp=13),
    )


@pytest.mark.parametrize(
    ("train_records", "test_records", "message"),
    [
        ({3: ["1\t2\tbad\t10"]}, None, "Invalid rating in u.data row 1"),
        (None, {2: ["99\t12\t4\t10"]}, "Unexpected user ID 99 in row 1"),
        (None, {4: ["1\t99\t4\t10"]}, "Unexpected item ID 99 in row 1"),
    ],
)
def test_rejects_invalid_records_and_unexpected_ids(
    tmp_path: Path,
    train_records: dict[int, list[str]] | None,
    test_records: dict[int, list[str]] | None,
    message: str,
) -> None:
    data_dir = write_folds(
        tmp_path, train_records=train_records, test_records=test_records
    )

    with pytest.raises(ValueError, match=message):
        load_fixture_splits(data_dir)


def test_rejects_user_item_pair_shared_between_train_and_test(
    tmp_path: Path,
) -> None:
    data_dir = write_folds(
        tmp_path,
        train_records={1: ["1\t2\t3\t10"]},
        test_records={1: ["1\t2\t5\t20"]},
    )

    with pytest.raises(ValueError, match="Data leakage in fold 1.*user 1, item 2"):
        load_fixture_splits(data_dir)


def test_missing_fold_file_has_clear_error(tmp_path: Path) -> None:
    data_dir = write_folds(tmp_path)
    (data_dir / "u5.test").unlink()

    with pytest.raises(FileNotFoundError, match="u.data file not found.*u5.test"):
        load_fixture_splits(data_dir)
