import csv
from dataclasses import dataclass
from pathlib import Path

from app.models import Movie

_GENRE_COUNT = 19
_ITEM_FIELD_COUNT = 5 + _GENRE_COUNT


@dataclass(frozen=True)
class MovieLensRating:
    """One MovieLens rating record; IDs, rating, and timestamp are integers."""

    user_id: int
    item_id: int
    rating: int
    timestamp: int


def load_movielens_ratings(data_path: str | Path) -> list[MovieLensRating]:
    """Load tab-separated ratings from MovieLens 100K ``u.data``.

    Records are returned in file order. Duplicate records are preserved; this
    loader validates and parses data but does not deduplicate it.
    """
    path = Path(data_path)
    try:
        with path.open(encoding="ascii", newline="") as stream:
            ratings: list[MovieLensRating] = []
            for line_number, line in enumerate(stream, start=1):
                fields = line.rstrip("\r\n").split("\t")
                if len(fields) != 4:
                    raise ValueError(
                        f"Invalid u.data row {line_number} in '{path}': expected "
                        f"4 tab-separated fields, got {len(fields)}"
                    )

                values: list[int] = []
                for field_name, field_value in zip(
                    ("user ID", "item ID", "rating", "timestamp"),
                    fields,
                    strict=True,
                ):
                    try:
                        values.append(int(field_value))
                    except ValueError:
                        raise ValueError(
                            f"Invalid {field_name} in u.data row {line_number} "
                            f"of '{path}': {field_value!r}; expected an integer"
                        ) from None

                user_id, item_id, rating, timestamp = values
                if user_id <= 0:
                    raise ValueError(
                        f"Invalid user ID in u.data row {line_number} of "
                        f"'{path}': {user_id}; expected a positive integer"
                    )
                if item_id <= 0:
                    raise ValueError(
                        f"Invalid item ID in u.data row {line_number} of "
                        f"'{path}': {item_id}; expected a positive integer"
                    )
                if not 1 <= rating <= 5:
                    raise ValueError(
                        f"Invalid rating in u.data row {line_number} of "
                        f"'{path}': {rating}; expected an integer from 1 to 5"
                    )
                if timestamp < 0:
                    raise ValueError(
                        f"Invalid timestamp in u.data row {line_number} of "
                        f"'{path}': {timestamp}; expected a non-negative integer"
                    )

                ratings.append(MovieLensRating(user_id, item_id, rating, timestamp))
    except FileNotFoundError:
        raise FileNotFoundError(f"MovieLens u.data file not found: '{path}'") from None

    return ratings


def load_movielens_movies(
    item_path: str | Path,
    genre_path: str | Path,
) -> list[Movie]:
    """Load MovieLens 100K movie metadata from ``u.item`` and ``u.genre``.

    The files use pipe-delimited rows and the MovieLens 100K Latin-1 encoding.
    Movie descriptions are not present in these files, so ``overview`` remains
    unset.
    """
    genres_by_id = _load_genres(Path(genre_path))
    item_file = Path(item_path)
    try:
        with item_file.open(encoding="latin-1", newline="") as stream:
            reader = csv.reader(stream, delimiter="|", strict=True)
            movies: list[Movie] = []
            seen_ids: set[int] = set()

            for row_number, row in enumerate(reader, start=1):
                if len(row) != _ITEM_FIELD_COUNT:
                    raise ValueError(
                        f"Invalid u.item row {row_number} in '{item_file}': "
                        f"expected {_ITEM_FIELD_COUNT} fields, got {len(row)}"
                    )

                try:
                    movie_id = int(row[0])
                except ValueError:
                    raise ValueError(
                        f"Invalid movie ID in u.item row {row_number} of "
                        f"'{item_file}': {row[0]!r}"
                    ) from None

                if movie_id in seen_ids:
                    raise ValueError(
                        f"Duplicate movie ID {movie_id} in u.item row "
                        f"{row_number} of '{item_file}'"
                    )
                seen_ids.add(movie_id)

                title = row[1].strip()
                if not title:
                    raise ValueError(
                        f"Missing movie title in u.item row {row_number} "
                        f"of '{item_file}'"
                    )

                movie_genres: list[str] = []
                for flag_index, flag in enumerate(row[5:], start=1):
                    if flag not in {"0", "1"}:
                        raise ValueError(
                            f"Invalid genre flag {flag!r} in u.item row "
                            f"{row_number}, genre column {flag_index} of "
                            f"'{item_file}': expected '0' or '1'"
                        )
                    if flag == "1":
                        movie_genres.append(genres_by_id[flag_index - 1])

                movies.append(
                    Movie(id=str(movie_id), title=title, genres=movie_genres)
                )

    except FileNotFoundError:
        raise FileNotFoundError(
            f"MovieLens u.item file not found: '{item_file}'"
        ) from None
    except csv.Error as error:
        raise ValueError(f"Invalid u.item file '{item_file}': {error}") from error

    if not movies:
        raise ValueError(
            f"MovieLens u.item file contains no movie records: '{item_file}'"
        )
    return movies


def _load_genres(genre_path: Path) -> tuple[str, ...]:
    try:
        with genre_path.open(encoding="latin-1", newline="") as stream:
            reader = csv.reader(stream, delimiter="|", strict=True)
            genres_by_id: dict[int, str] = {}

            for row_number, row in enumerate(reader, start=1):
                if row in ([], ["", ""]):
                    continue
                if len(row) != 2:
                    raise ValueError(
                        f"Invalid u.genre row {row_number} in '{genre_path}': "
                        f"expected 2 fields, got {len(row)}"
                    )
                name = row[0].strip()
                if not name:
                    raise ValueError(
                        f"Missing genre name in u.genre row {row_number} "
                        f"of '{genre_path}'"
                    )
                try:
                    genre_id = int(row[1])
                except ValueError:
                    raise ValueError(
                        f"Invalid genre ID in u.genre row {row_number} of "
                        f"'{genre_path}': {row[1]!r}"
                    ) from None
                if genre_id in genres_by_id:
                    raise ValueError(
                        f"Duplicate genre ID {genre_id} in u.genre row "
                        f"{row_number} of '{genre_path}'"
                    )
                genres_by_id[genre_id] = name

    except FileNotFoundError:
        raise FileNotFoundError(
            f"MovieLens u.genre file not found: '{genre_path}'"
        ) from None
    except csv.Error as error:
        raise ValueError(f"Invalid u.genre file '{genre_path}': {error}") from error

    expected_ids = set(range(_GENRE_COUNT))
    actual_ids = set(genres_by_id)
    if actual_ids != expected_ids:
        raise ValueError(
            f"Invalid u.genre file '{genre_path}': expected genre IDs 0 through "
            f"{_GENRE_COUNT - 1}, got {sorted(actual_ids)}"
        )
    if len(genres_by_id) != _GENRE_COUNT:
        raise ValueError(
            f"Invalid u.genre file '{genre_path}': expected {_GENRE_COUNT} genres, "
            f"got {len(genres_by_id)}"
        )
    return tuple(genres_by_id[index] for index in range(_GENRE_COUNT))
