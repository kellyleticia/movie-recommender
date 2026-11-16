from collections.abc import Iterable
from dataclasses import dataclass

from app.models import Movie


@dataclass(frozen=True)
class Recommendation:
    movie: Movie
    shared_genres_count: int


def recommend_movies(
    reference_movie: Movie,
    movies: Iterable[Movie],
    limit: int = 10,
) -> list[Recommendation]:
    """Recommend distinct movies ranked by shared genre count and title."""
    reference_genres = reference_movie.normalized_genres
    recommendations: list[Recommendation] = []
    seen_ids: set[str] = set()

    for movie in movies:
        if movie.id == reference_movie.id or movie.id in seen_ids:
            continue
        seen_ids.add(movie.id)

        shared_genres_count = len(reference_genres & movie.normalized_genres)
        if shared_genres_count:
            recommendations.append(Recommendation(movie, shared_genres_count))

    recommendations.sort(
        key=lambda recommendation: (
            -recommendation.shared_genres_count,
            recommendation.movie.title.casefold(),
            recommendation.movie.id.casefold(),
        )
    )
    return recommendations[:limit]
