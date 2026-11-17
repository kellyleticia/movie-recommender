from collections.abc import Iterable
from dataclasses import dataclass

from app.models import Movie
from app.similarity import SimilarityMatrix


@dataclass(frozen=True)
class ContentRecommendation:
    movie: Movie
    similarity_score: float


def recommend_by_content(
    reference_movie: Movie,
    similarity: SimilarityMatrix,
    movies: Iterable[Movie],
    limit: int = 10,
) -> list[ContentRecommendation]:
    """Rank movies by content similarity; non-positive limits return no results.

    Raises ValueError when the reference movie ID is absent from the similarity
    matrix. Candidate movies are resolved by ID, independent of input ordering.
    """
    try:
        reference_index = similarity.movie_ids.index(reference_movie.id)
    except ValueError:
        raise ValueError(
            f"Reference movie ID not found in similarity matrix: {reference_movie.id}"
        ) from None

    if limit <= 0:
        return []

    movies_by_id = {movie.id: movie for movie in movies}
    row = similarity.matrix.getrow(reference_index)
    recommendations: list[ContentRecommendation] = []

    for candidate_index, score in zip(row.indices, row.data, strict=True):
        candidate_id = similarity.movie_ids[candidate_index]
        if candidate_id == reference_movie.id or score <= 0:
            continue

        candidate = movies_by_id.get(candidate_id)
        if candidate is not None:
            recommendations.append(ContentRecommendation(candidate, float(score)))

    recommendations.sort(
        key=lambda recommendation: (
            -recommendation.similarity_score,
            recommendation.movie.title.casefold(),
            recommendation.movie.id.casefold(),
            recommendation.movie.title,
            recommendation.movie.id,
        )
    )
    return recommendations[:limit]
