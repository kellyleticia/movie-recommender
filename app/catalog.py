from collections.abc import Iterable

from app.models import Movie


class MovieCatalog:
    """An in-memory catalog of movies indexed by their IDs."""

    def __init__(self, movies: Iterable[Movie] = ()) -> None:
        self._movies: dict[str, Movie] = {}
        for movie in movies:
            if movie.id in self._movies:
                raise ValueError(f"Duplicate movie ID: {movie.id}")
            self._movies[movie.id] = movie

    def list_movies(self) -> list[Movie]:
        """Return all movies in insertion order."""
        return list(self._movies.values())

    def get_movie(self, movie_id: str) -> Movie | None:
        """Return the movie with the given ID, or None if it is absent."""
        return self._movies.get(movie_id)
