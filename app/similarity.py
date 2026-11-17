from dataclasses import dataclass

from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity

from app.text_features import TfidfRepresentation


@dataclass(frozen=True)
class SimilarityMatrix:
    """Sparse cosine similarities aligned with the supplied movie IDs."""

    movie_ids: tuple[str, ...]
    matrix: csr_matrix


def compute_similarity(
    representation: TfidfRepresentation,
) -> SimilarityMatrix:
    """Compute pairwise cosine similarity without densifying TF-IDF vectors."""
    movie_count = len(representation.movie_ids)
    tfidf_matrix = representation.matrix

    if tfidf_matrix.shape[1] == 0:
        similarities = csr_matrix((movie_count, movie_count), dtype=float)
    else:
        similarities = cosine_similarity(tfidf_matrix, dense_output=False).tocsr()
        similarities.data = similarities.data.clip(0.0, 1.0)

    return SimilarityMatrix(representation.movie_ids, similarities)
