import numpy as np
from scipy.sparse import csr_matrix

from app.models import Movie
from app.text_features import build_tfidf_matrix


def make_movie(movie_id: str, overview: str | None = None) -> Movie:
    return Movie(id=movie_id, title=movie_id, genres=[], overview=overview)


def test_builds_sparse_tfidf_matrix_and_features() -> None:
    movies = [
        make_movie("arrival", "A linguist studies an alien signal"),
        make_movie("contact", "An astronomer receives an alien signal"),
    ]

    result = build_tfidf_matrix(movies)

    assert isinstance(result.matrix, csr_matrix)
    assert result.matrix.shape == (2, len(result.feature_names))
    assert set(result.feature_names) == {
        "an",
        "alien",
        "astronomer",
        "linguist",
        "receives",
        "signal",
        "studies",
    }
    assert result.matrix[0].nnz > 0
    assert result.matrix[1].nnz > 0


def test_preserves_movie_order_and_aligns_ids_with_rows() -> None:
    movies = [
        make_movie("third", "Third unique description"),
        make_movie("first", "First unique description"),
        make_movie("second", "Second unique description"),
    ]

    result = build_tfidf_matrix(movies)
    feature_indexes = {
        feature: index for index, feature in enumerate(result.feature_names)
    }

    assert result.movie_ids == ("third", "first", "second")
    assert result.matrix[0, feature_indexes["third"]] > 0
    assert result.matrix[1, feature_indexes["first"]] > 0
    assert result.matrix[2, feature_indexes["second"]] > 0


def test_reserves_zero_rows_for_movies_without_overviews() -> None:
    movies = [
        make_movie("without-before"),
        make_movie("with", "A detailed cinematic story"),
        make_movie("without-after"),
    ]

    result = build_tfidf_matrix(movies)

    assert result.movie_ids == ("without-before", "with", "without-after")
    assert result.matrix.shape[0] == 3
    assert result.matrix[0].nnz == 0
    assert result.matrix[1].nnz > 0
    assert result.matrix[2].nnz == 0


def test_empty_catalog_returns_empty_sparse_matrix() -> None:
    result = build_tfidf_matrix([])

    assert result.movie_ids == ()
    assert result.feature_names == ()
    assert isinstance(result.matrix, csr_matrix)
    assert result.matrix.shape == (0, 0)


def test_all_missing_overviews_return_zero_rows_without_features() -> None:
    result = build_tfidf_matrix([make_movie("one"), make_movie("two")])

    assert result.movie_ids == ("one", "two")
    assert result.feature_names == ()
    assert result.matrix.shape == (2, 0)


def test_overviews_without_recognized_terms_return_zero_matrix() -> None:
    result = build_tfidf_matrix(
        [make_movie("one", "!!!"), make_movie("two", "x 7")]
    )

    assert result.movie_ids == ("one", "two")
    assert result.feature_names == ()
    assert result.matrix.shape == (2, 0)


def test_empty_overview_keeps_its_zero_row() -> None:
    empty_overview = Movie.model_construct(
        id="empty", title="Empty", genres=[], overview=""
    )
    movies = [empty_overview, make_movie("useful", "A story with useful words")]

    result = build_tfidf_matrix(movies)

    assert result.movie_ids == ("empty", "useful")
    assert result.matrix.shape[0] == 2
    assert result.matrix[0].nnz == 0
    assert result.matrix[1].nnz > 0


def test_tfidf_values_are_non_negative() -> None:
    result = build_tfidf_matrix(
        [make_movie("one", "space voyage"), make_movie("two", "space station")]
    )

    assert np.all(result.matrix.data >= 0)


def test_missing_overviews_do_not_change_other_movies_tfidf_weights() -> None:
    movies = [
        make_movie("one", "shared alpha"),
        make_movie("two", "shared beta"),
    ]
    baseline = build_tfidf_matrix(movies)
    with_missing = build_tfidf_matrix(
        [make_movie("missing-before"), *movies, make_movie("missing-after")]
    )

    np.testing.assert_allclose(
        baseline.matrix.toarray(), with_missing.matrix[1:3].toarray()
    )
    assert with_missing.matrix[0].nnz == 0
    assert with_missing.matrix[3].nnz == 0


def test_unrecognized_overviews_do_not_change_other_movies_tfidf_weights() -> None:
    movies = [
        make_movie("one", "shared alpha"),
        make_movie("two", "shared beta"),
    ]
    baseline = build_tfidf_matrix(movies)
    with_unrecognized = build_tfidf_matrix(
        [make_movie("punctuation", "!!!"), *movies, make_movie("single-token", "x 7")]
    )

    np.testing.assert_allclose(
        baseline.matrix.toarray(), with_unrecognized.matrix[1:3].toarray()
    )
    assert with_unrecognized.matrix[0].nnz == 0
    assert with_unrecognized.matrix[3].nnz == 0
