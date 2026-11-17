from collections.abc import Sequence
from dataclasses import dataclass

from scipy.sparse import coo_matrix, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer

from app.models import Movie


@dataclass(frozen=True)
class TfidfRepresentation:
    """Sparse TF-IDF features aligned with the supplied movie order."""

    movie_ids: tuple[str, ...]
    feature_names: tuple[str, ...]
    matrix: csr_matrix


def build_tfidf_matrix(movies: Sequence[Movie]) -> TfidfRepresentation:
    """Build a sparse TF-IDF matrix from movie overviews."""
    movie_ids = tuple(movie.id for movie in movies)
    documents = [movie.overview or "" for movie in movies]
    vectorizer = TfidfVectorizer()
    analyze = vectorizer.build_analyzer()
    useful_indexes = [
        index for index, document in enumerate(documents) if analyze(document)
    ]

    if not useful_indexes:
        return TfidfRepresentation(
            movie_ids=movie_ids,
            feature_names=(),
            matrix=csr_matrix((len(movies), 0), dtype=float),
        )

    useful_documents = [documents[index] for index in useful_indexes]
    useful_matrix = vectorizer.fit_transform(useful_documents).tocoo()
    matrix = coo_matrix(
        (
            useful_matrix.data,
            ([useful_indexes[row] for row in useful_matrix.row], useful_matrix.col),
        ),
        shape=(len(movies), useful_matrix.shape[1]),
    ).tocsr()
    feature_names = tuple(vectorizer.get_feature_names_out())
    return TfidfRepresentation(movie_ids, feature_names, matrix)
