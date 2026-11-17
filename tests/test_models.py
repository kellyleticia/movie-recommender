import pytest
from pydantic import ValidationError

from app.models import Movie


def test_creates_valid_movie() -> None:
    movie = Movie(id="movie-1", title="Arrival", genres=["Drama", "Sci-Fi"])

    assert movie.id == "movie-1"
    assert movie.title == "Arrival"
    assert movie.genres == ["Drama", "Sci-Fi"]
    assert movie.overview is None


def test_accepts_and_strips_overview() -> None:
    movie = Movie(
        id="movie-1",
        title="Arrival",
        genres=[],
        overview="  A linguist studies an alien arrival.  ",
    )

    assert movie.overview == "A linguist studies an alien arrival."


def test_accepts_explicit_none_overview() -> None:
    movie = Movie(id="movie-1", title="Arrival", genres=[], overview=None)

    assert movie.overview is None


@pytest.mark.parametrize("overview", ["", "   "])
def test_rejects_empty_or_whitespace_only_overview(overview: str) -> None:
    with pytest.raises(ValidationError):
        Movie(id="movie-1", title="Arrival", genres=[], overview=overview)


def test_strips_outer_whitespace_from_id_and_title() -> None:
    movie = Movie(id="  movie-1  ", title="  Arrival  ", genres=[])

    assert movie.id == "movie-1"
    assert movie.title == "Arrival"


def test_strips_outer_whitespace_from_genres() -> None:
    movie = Movie(id="movie-1", title="Arrival", genres=["  Drama  "])

    assert movie.genres == ["Drama"]


@pytest.mark.parametrize("movie_id", ["", "   "])
def test_rejects_empty_or_whitespace_only_id(movie_id: str) -> None:
    with pytest.raises(ValidationError):
        Movie(id=movie_id, title="Arrival", genres=[])


@pytest.mark.parametrize("title", ["", "   "])
def test_rejects_empty_or_whitespace_only_title(title: str) -> None:
    with pytest.raises(ValidationError):
        Movie(id="movie-1", title=title, genres=[])


@pytest.mark.parametrize("genre", ["", "   "])
def test_rejects_empty_or_whitespace_only_genres(genre: str) -> None:
    with pytest.raises(ValidationError):
        Movie(id="movie-1", title="Arrival", genres=[genre])


def test_accepts_empty_genre_list() -> None:
    movie = Movie(id="movie-1", title="Arrival", genres=[])

    assert movie.genres == []


def test_normalizes_genres_for_case_insensitive_comparison() -> None:
    movie = Movie(id="movie-1", title="Arrival", genres=["Drama", "SCI-FI"])

    assert movie.normalized_genres == frozenset({"drama", "sci-fi"})


def test_preserves_original_genre_presentation() -> None:
    movie = Movie(id="movie-1", title="Arrival", genres=["dRaMa", "Sci-Fi"])

    assert movie.genres == ["dRaMa", "Sci-Fi"]
    assert movie.normalized_genres == frozenset({"drama", "sci-fi"})
