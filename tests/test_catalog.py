import pytest

from app.catalog import MovieCatalog
from app.models import Movie


def make_movie(movie_id: str, title: str) -> Movie:
    return Movie(id=movie_id, title=title, genres=[])


def test_empty_catalog_lists_no_movies() -> None:
    catalog = MovieCatalog()

    assert catalog.list_movies() == []


def test_lists_all_movies_in_insertion_order() -> None:
    first = make_movie("movie-1", "Arrival")
    second = make_movie("movie-2", "Moonlight")
    catalog = MovieCatalog([first, second])

    assert catalog.list_movies() == [first, second]


def test_finds_movie_by_id() -> None:
    movie = make_movie("movie-1", "Arrival")
    catalog = MovieCatalog([movie])

    assert catalog.get_movie("movie-1") is movie


def test_returns_none_when_movie_id_does_not_exist() -> None:
    catalog = MovieCatalog([make_movie("movie-1", "Arrival")])

    assert catalog.get_movie("missing") is None


def test_rejects_duplicate_ids_during_initialization() -> None:
    first = make_movie("movie-1", "Arrival")
    duplicate = make_movie("movie-1", "Moonlight")

    with pytest.raises(ValueError, match="Duplicate movie ID: movie-1"):
        MovieCatalog([first, duplicate])
