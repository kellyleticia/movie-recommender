import pytest

from app.content_recommender import ContentRecommendation, recommend_by_content
from app.models import Movie
from app.similarity import SimilarityMatrix, compute_similarity
from app.text_features import TfidfRepresentation, build_tfidf_matrix


def make_movie(movie_id: str, overview: str | None = None) -> Movie:
    return Movie(id=movie_id, title=movie_id, genres=[], overview=overview)


def run_pipeline(
    movies: list[Movie], reference_id: str
) -> tuple[TfidfRepresentation, SimilarityMatrix, list[ContentRecommendation]]:
    representation = build_tfidf_matrix(movies)
    similarities = compute_similarity(representation)
    reference = next(movie for movie in movies if movie.id == reference_id)
    recommendations = recommend_by_content(reference, similarities, movies)
    return representation, similarities, recommendations


def test_pipeline_ranks_related_movies_and_preserves_id_object_alignment() -> None:
    movies = [
        make_movie("punctuation", "!!!"),
        make_movie("partial", "A distant planet appears"),
        make_movie("reference", "A crew explores a distant alien planet"),
        make_movie("missing"),
        make_movie(
            "related",
            "A crew explores a distant alien planet with astronauts aboard",
        ),
        make_movie("unrelated", "A chef prepares a meal in a restaurant"),
    ]

    representation, similarities, recommendations = run_pipeline(movies, "reference")
    movies_by_id = {movie.id: movie for movie in movies}

    assert representation.movie_ids == tuple(movie.id for movie in movies)
    assert similarities.movie_ids == representation.movie_ids
    assert representation.matrix[0].nnz == 0
    assert representation.matrix[3].nnz == 0
    assert [item.movie.id for item in recommendations] == ["related", "partial"]
    assert recommendations[0].similarity_score > recommendations[1].similarity_score
    assert all(item.movie is movies_by_id[item.movie.id] for item in recommendations)


def test_pipeline_handles_reference_with_zero_tfidf_vector() -> None:
    movies = [
        make_movie("candidate", "An astronaut explores an alien planet"),
        make_movie("reference"),
    ]

    representation, similarities, recommendations = run_pipeline(movies, "reference")
    reference_index = representation.movie_ids.index("reference")

    assert representation.matrix[reference_index].nnz == 0
    assert similarities.matrix.getrow(reference_index).nnz == 0
    assert recommendations == []


def test_pipeline_handles_empty_catalog() -> None:
    representation = build_tfidf_matrix([])
    similarities = compute_similarity(representation)

    assert representation.movie_ids == ()
    assert representation.matrix.shape == (0, 0)
    assert similarities.movie_ids == ()
    assert similarities.matrix.shape == (0, 0)

    with pytest.raises(ValueError, match="Reference movie ID not found"):
        recommend_by_content(make_movie("missing"), similarities, [])


def test_pipeline_with_one_movie_returns_no_recommendations() -> None:
    movie = make_movie("only", "A single complete synopsis")

    representation, similarities, recommendations = run_pipeline([movie], "only")

    assert representation.movie_ids == ("only",)
    assert similarities.matrix.shape == (1, 1)
    assert similarities.matrix[0, 0] == 1
    assert recommendations == []
