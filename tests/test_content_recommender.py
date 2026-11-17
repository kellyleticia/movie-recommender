import pytest
from scipy.sparse import csr_matrix

from app.content_recommender import recommend_by_content
from app.models import Movie
from app.similarity import SimilarityMatrix


def make_movie(movie_id: str, title: str | None = None) -> Movie:
    return Movie(id=movie_id, title=title or movie_id, genres=[])


def make_similarity(
    movie_ids: tuple[str, ...], values: list[list[float]]
) -> SimilarityMatrix:
    return SimilarityMatrix(movie_ids, csr_matrix(values, dtype=float))


def test_orders_recommendations_by_similarity_and_excludes_reference() -> None:
    similarity = make_similarity(
        ("reference", "lower", "higher", "zero"),
        [
            [1.0, 0.3, 0.9, 0.0],
            [0.3, 1.0, 0.2, 0.0],
            [0.9, 0.2, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ],
    )
    movies = [
        make_movie("zero"),
        make_movie("higher"),
        make_movie("reference"),
        make_movie("lower"),
    ]

    recommendations = recommend_by_content(
        make_movie("reference"), similarity, movies
    )

    assert [item.movie.id for item in recommendations] == ["higher", "lower"]
    assert [item.similarity_score for item in recommendations] == [0.9, 0.3]


def test_applies_recommendation_limit() -> None:
    similarity = make_similarity(
        ("reference", "one", "two"),
        [[1.0, 0.8, 0.7], [0.8, 1.0, 0.0], [0.7, 0.0, 1.0]],
    )

    recommendations = recommend_by_content(
        make_movie("reference"),
        similarity,
        [make_movie("one"), make_movie("two")],
        limit=1,
    )

    assert [item.movie.id for item in recommendations] == ["one"]


@pytest.mark.parametrize("limit", [0, -1])
def test_non_positive_limit_returns_no_recommendations(limit: int) -> None:
    similarity = make_similarity(
        ("reference", "candidate"), [[1.0, 0.8], [0.8, 1.0]]
    )

    assert (
        recommend_by_content(
            make_movie("reference"), similarity, [make_movie("candidate")], limit
        )
        == []
    )


def test_does_not_recommend_movies_with_zero_similarity() -> None:
    similarity = make_similarity(
        ("reference", "zero", "positive"),
        [[1.0, 0.0, 0.4], [0.0, 1.0, 0.0], [0.4, 0.0, 1.0]],
    )

    recommendations = recommend_by_content(
        make_movie("reference"),
        similarity,
        [make_movie("zero"), make_movie("positive")],
    )

    assert [item.movie.id for item in recommendations] == ["positive"]


def test_breaks_ties_by_casefolded_title_then_id_deterministically() -> None:
    similarity = make_similarity(
        ("reference", "z-id", "a-id", "same-title"),
        [
            [1.0, 0.5, 0.5, 0.5],
            [0.5, 1.0, 0.0, 0.0],
            [0.5, 0.0, 1.0, 0.0],
            [0.5, 0.0, 0.0, 1.0],
        ],
    )
    movies = [
        make_movie("z-id", "Beta"),
        make_movie("a-id", "alpha"),
        make_movie("same-title", "Alpha"),
    ]

    recommendations = recommend_by_content(
        make_movie("reference"), similarity, movies
    )

    assert [item.movie.id for item in recommendations] == [
        "a-id",
        "same-title",
        "z-id",
    ]


def test_raises_when_reference_is_absent_from_similarity_matrix() -> None:
    similarity = make_similarity(("other",), [[1.0]])

    with pytest.raises(ValueError, match="Reference movie ID not found"):
        recommend_by_content(make_movie("missing"), similarity, [make_movie("other")])


def test_returns_empty_when_catalog_has_no_positive_recommendations() -> None:
    similarity = make_similarity(
        ("reference", "candidate"), [[1.0, 0.0], [0.0, 1.0]]
    )

    recommendations = recommend_by_content(
        make_movie("reference"), similarity, [make_movie("candidate")]
    )

    assert recommendations == []


def test_resolves_candidate_movies_by_id_not_input_order() -> None:
    similarity = make_similarity(
        ("reference", "candidate-a", "candidate-b"),
        [[1.0, 0.8, 0.6], [0.8, 1.0, 0.0], [0.6, 0.0, 1.0]],
    )
    movies = [
        make_movie("candidate-b", "B Movie"),
        make_movie("candidate-a", "A Movie"),
    ]

    recommendations = recommend_by_content(
        make_movie("reference"), similarity, movies
    )

    assert [item.movie.title for item in recommendations] == ["A Movie", "B Movie"]
