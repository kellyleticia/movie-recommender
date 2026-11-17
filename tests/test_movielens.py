from pathlib import Path

import pytest

from app.catalog import MovieCatalog
from app.models import Movie
from app.movielens import (
    MovieLensRating,
    load_movielens_movies,
    load_movielens_ratings,
)

GENRES = [
    "unknown",
    "Action",
    "Adventure",
    "Animation",
    "Children's",
    "Comedy",
    "Crime",
    "Documentary",
    "Drama",
    "Fantasy",
    "Film-Noir",
    "Horror",
    "Musical",
    "Mystery",
    "Romance",
    "Sci-Fi",
    "Thriller",
    "War",
    "Western",
]


def write_dataset_files(
    tmp_path: Path,
    item_rows: list[str],
    genre_rows: list[str] | None = None,
) -> tuple[Path, Path]:
    item_path = tmp_path / "u.item"
    genre_path = tmp_path / "u.genre"
    item_path.write_text("\n".join(item_rows) + "\n", encoding="latin-1")
    rows = genre_rows or [f"{name}|{index}" for index, name in enumerate(GENRES)]
    genre_path.write_text("\n".join(rows) + "\n", encoding="latin-1")
    return item_path, genre_path


def item_row(movie_id: str, title: str, active_genres: set[int] = frozenset()) -> str:
    fields = [movie_id, title, "01-Jan-1995", "", "http://example.test/movie"]
    fields.extend("1" if index in active_genres else "0" for index in range(19))
    return "|".join(fields)


def test_loads_ratings_in_order_and_preserves_duplicates(tmp_path: Path) -> None:
    data_path = tmp_path / "u.data"
    duplicate = "7\t12\t5\t123456789"
    data_path.write_text(
        f"{duplicate}\n8\t3\t2\t0\n{duplicate}\n", encoding="ascii"
    )

    ratings = load_movielens_ratings(data_path)

    assert ratings == [
        MovieLensRating(user_id=7, item_id=12, rating=5, timestamp=123456789),
        MovieLensRating(user_id=8, item_id=3, rating=2, timestamp=0),
        MovieLensRating(user_id=7, item_id=12, rating=5, timestamp=123456789),
    ]


@pytest.mark.parametrize(
    ("record", "message"),
    [
        ("1\t2\t3", "expected 4 tab-separated fields, got 3"),
        ("0\t2\t3\t4", "user ID"),
        ("1\t-2\t3\t4", "item ID"),
        ("1\t2\t0\t4", "rating"),
        ("1\t2\t6\t4", "rating"),
        ("1\t2\tthree\t4", "Invalid rating"),
        ("1\t2\t3\t-1", "timestamp"),
        ("1\t2\t3\twhen", "Invalid timestamp"),
    ],
)
def test_rejects_invalid_rating_with_line_number(
    tmp_path: Path, record: str, message: str
) -> None:
    data_path = tmp_path / "u.data"
    data_path.write_text(f"1\t1\t1\t1\n{record}\n", encoding="ascii")

    with pytest.raises(ValueError) as error:
        load_movielens_ratings(data_path)
    assert "row 2" in str(error.value)
    assert message in str(error.value)


def test_missing_ratings_file_has_clear_error(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="u.data file not found"):
        load_movielens_ratings(tmp_path / "u.data")


def test_loads_ids_titles_genres_and_missing_overview(tmp_path: Path) -> None:
    item_path, genre_path = write_dataset_files(
        tmp_path, [item_row("7", "A Movie", {1, 8})]
    )

    movies = load_movielens_movies(item_path, genre_path)

    assert movies == [
        Movie(id="7", title="A Movie", genres=["Action", "Drama"], overview=None)
    ]


def test_loads_multiple_genres_and_movie_without_genres(tmp_path: Path) -> None:
    item_path, genre_path = write_dataset_files(
        tmp_path,
        [item_row("1", "Several Genres", {2, 5, 18}), item_row("2", "No Genre")],
    )

    movies = load_movielens_movies(item_path, genre_path)

    assert movies[0].genres == ["Adventure", "Comedy", "Western"]
    assert movies[1].genres == []


def test_preserves_movie_file_order(tmp_path: Path) -> None:
    item_path, genre_path = write_dataset_files(
        tmp_path, [item_row("20", "Later ID"), item_row("3", "Earlier ID")]
    )

    movies = load_movielens_movies(item_path, genre_path)

    assert [movie.id for movie in movies] == ["20", "3"]


@pytest.mark.parametrize("missing", ["item", "genre"])
def test_missing_file_has_clear_error(tmp_path: Path, missing: str) -> None:
    item_path, genre_path = write_dataset_files(tmp_path, [item_row("1", "Movie")])
    if missing == "item":
        item_path.unlink()
    else:
        genre_path.unlink()

    with pytest.raises(FileNotFoundError, match=f"u\\.{missing}.*not found"):
        load_movielens_movies(item_path, genre_path)


@pytest.mark.parametrize(
    ("rows", "message"),
    [
        (["1|Title|too|few"], "expected 24 fields"),
        ([item_row("x", "Bad ID")], "Invalid movie ID"),
        ([item_row("1", "Bad Flag", {1}).rsplit("|", 1)[0] + "|2"], "genre flag"),
        ([item_row("1", "First"), item_row("1", "Duplicate")], "Duplicate movie ID"),
    ],
)
def test_invalid_movie_data_has_clear_error(
    tmp_path: Path, rows: list[str], message: str
) -> None:
    item_path, genre_path = write_dataset_files(tmp_path, rows)

    with pytest.raises(ValueError, match=message):
        load_movielens_movies(item_path, genre_path)


def test_invalid_genre_structure_has_clear_error(tmp_path: Path) -> None:
    item_path, genre_path = write_dataset_files(
        tmp_path, [item_row("1", "Movie")], ["Action|1"]
    )

    with pytest.raises(ValueError, match="expected genre IDs 0 through 18"):
        load_movielens_movies(item_path, genre_path)


def test_accepts_official_trailing_empty_genre_row(tmp_path: Path) -> None:
    item_path = tmp_path / "u.item"
    genre_path = tmp_path / "u.genre"
    item_path.write_text(item_row("1", "Movie", {1, 8}) + "\n", encoding="latin-1")
    genre_rows = [f"{name}|{index}" for index, name in enumerate(GENRES)]
    genre_path.write_text("\n".join([*genre_rows, "|"]) + "\n", encoding="latin-1")

    movies = load_movielens_movies(item_path, genre_path)

    assert movies[0].genres == ["Action", "Drama"]


def test_ignores_completely_blank_genre_line(tmp_path: Path) -> None:
    genre_rows = [f"{name}|{index}" for index, name in enumerate(GENRES)]
    item_path, genre_path = write_dataset_files(
        tmp_path, [item_row("1", "Movie", {1})], [*genre_rows, ""]
    )

    movies = load_movielens_movies(item_path, genre_path)

    assert movies[0].genres == ["Action"]


def test_rejects_malformed_non_terminator_genre_row(tmp_path: Path) -> None:
    item_path, genre_path = write_dataset_files(
        tmp_path,
        [item_row("1", "Movie")],
        [f"{name}|{index}" for index, name in enumerate(GENRES[:-1])]
        + ["Western|18|unexpected"],
    )

    with pytest.raises(ValueError, match="expected 2 fields, got 3"):
        load_movielens_movies(item_path, genre_path)


def test_loaded_movies_are_compatible_with_movie_catalog(tmp_path: Path) -> None:
    item_path, genre_path = write_dataset_files(
        tmp_path, [item_row("8", "First"), item_row("9", "Second")]
    )

    catalog = MovieCatalog(load_movielens_movies(item_path, genre_path))

    assert [movie.id for movie in catalog.list_movies()] == ["8", "9"]
    assert catalog.get_movie("9") == Movie(id="9", title="Second", genres=[])
