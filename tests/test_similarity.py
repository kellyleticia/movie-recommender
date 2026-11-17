import numpy as np
from scipy.sparse import csr_matrix

from app.models import Movie
from app.similarity import compute_similarity
from app.text_features import build_tfidf_matrix


def make_movie(movie_id: str, overview: str | None = None) -> Movie:
    return Movie(id=movie_id, title=movie_id, genres=[], overview=overview)


def test_related_texts_have_positive_similarity_and_unrelated_texts_do_not() -> None:
    representation = build_tfidf_matrix(
        [
            make_movie("space-one", "An astronaut explores a distant planet"),
            make_movie("space-two", "A planet mission follows an astronaut"),
            make_movie("garden", "A gardener grows flowers in spring"),
        ]
    )

    result = compute_similarity(representation)

    assert result.matrix[0, 1] > 0
    assert result.matrix[0, 2] == 0


def test_preserves_movie_id_and_matrix_index_alignment() -> None:
    representation = build_tfidf_matrix(
        [
            make_movie("third", "Third orbital journey"),
            make_movie("first", "First forest adventure"),
            make_movie("second", "Second ocean voyage"),
        ]
    )

    result = compute_similarity(representation)

    assert result.movie_ids == ("third", "first", "second")
    assert result.matrix.shape == (3, 3)
    assert result.matrix[0, 0] == 1
    assert result.matrix[1, 1] == 1
    assert result.matrix[2, 2] == 1


def test_similarity_matrix_is_symmetric_and_values_are_bounded() -> None:
    representation = build_tfidf_matrix(
        [
            make_movie("one", "shared words in space"),
            make_movie("two", "shared words at sea"),
            make_movie("three", "different topic entirely"),
        ]
    )

    result = compute_similarity(representation)

    assert isinstance(result.matrix, csr_matrix)
    np.testing.assert_allclose(
        result.matrix.toarray(), result.matrix.transpose().toarray()
    )
    assert np.all(result.matrix.data >= 0)
    assert np.all(result.matrix.data <= 1)


def test_movies_without_overviews_have_zero_similarity_rows_and_diagonal() -> None:
    representation = build_tfidf_matrix(
        [
            make_movie("missing-before"),
            make_movie("with-overview", "A story about distant planets"),
            make_movie("missing-after"),
        ]
    )

    result = compute_similarity(representation)

    assert result.movie_ids == (
        "missing-before",
        "with-overview",
        "missing-after",
    )
    assert result.matrix[0].nnz == 0
    assert result.matrix[0, 0] == 0
    assert result.matrix[1, 1] == 1
    assert result.matrix[2].nnz == 0
    assert result.matrix[2, 2] == 0


def test_empty_catalog_returns_empty_csr_matrix() -> None:
    result = compute_similarity(build_tfidf_matrix([]))

    assert result.movie_ids == ()
    assert isinstance(result.matrix, csr_matrix)
    assert result.matrix.shape == (0, 0)


def test_single_movie_has_diagonal_one_when_vector_is_valid() -> None:
    result = compute_similarity(
        build_tfidf_matrix([make_movie("only", "A single useful synopsis")])
    )

    assert result.movie_ids == ("only",)
    assert result.matrix.shape == (1, 1)
    assert result.matrix[0, 0] == 1


def test_single_movie_has_zero_diagonal_when_vector_is_zero() -> None:
    result = compute_similarity(build_tfidf_matrix([make_movie("only")]))

    assert result.movie_ids == ("only",)
    assert result.matrix.shape == (1, 1)
    assert result.matrix[0, 0] == 0


def test_catalog_with_no_useful_text_returns_zero_similarity_matrix() -> None:
    representation = build_tfidf_matrix(
        [make_movie("punctuation", "!!!"), make_movie("missing")]
    )

    result = compute_similarity(representation)

    assert result.movie_ids == ("punctuation", "missing")
    assert isinstance(result.matrix, csr_matrix)
    assert result.matrix.shape == (2, 2)
    assert result.matrix.nnz == 0
