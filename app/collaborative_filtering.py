from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass

from scipy.sparse import coo_matrix, csr_matrix
from sklearn.metrics.pairwise import cosine_similarity

from app.movielens import MovieLensRating


@dataclass(frozen=True)
class ItemBasedModel:
    """Sparse user-item data and item-item cosine similarities.

    User and item IDs are sorted numerically; each tuple's position is its
    matrix index. The explicit maps provide the inverse lookup.
    """

    user_ids: tuple[int, ...]
    item_ids: tuple[int, ...]
    user_index: dict[int, int]
    item_index: dict[int, int]
    user_item_matrix: csr_matrix
    item_similarity: csr_matrix


@dataclass(frozen=True)
class CollaborativeRecommendation:
    item_id: int
    score: float


def build_item_based_model(
    train_ratings: Iterable[MovieLensRating],
    *,
    user_ids: Iterable[int] = (),
    item_ids: Iterable[int] = (),
) -> ItemBasedModel:
    """Build sparse rating and cosine-similarity matrices from training data.

    Only ``train_ratings`` are used. Optional IDs keep users or movies with no
    ratings in the matrix. IDs are sorted before assigning indexes, making the
    mapping deterministic. Duplicate user-item ratings are rejected because a
    matrix cell can represent only one rating.

    Item vectors use ratings centered by each user's mean before cosine
    similarity is computed. Missing ratings remain implicit zeros.
    """
    ratings = list(train_ratings)
    observed_users = {rating.user_id for rating in ratings}
    observed_items = {rating.item_id for rating in ratings}
    all_users = set(user_ids) | observed_users
    all_items = set(item_ids) | observed_items

    if any(user_id <= 0 for user_id in all_users):
        raise ValueError("User IDs must be positive integers")
    if any(item_id <= 0 for item_id in all_items):
        raise ValueError("Item IDs must be positive integers")

    ordered_users = tuple(sorted(all_users))
    ordered_items = tuple(sorted(all_items))
    user_index = {user_id: index for index, user_id in enumerate(ordered_users)}
    item_index = {item_id: index for index, item_id in enumerate(ordered_items)}

    row_indexes: list[int] = []
    column_indexes: list[int] = []
    values: list[float] = []
    seen_pairs: set[tuple[int, int]] = set()

    for rating in ratings:
        if not 1 <= rating.rating <= 5:
            raise ValueError(
                f"Rating for user {rating.user_id}, item {rating.item_id} "
                "must be between 1 and 5"
            )
        pair = (rating.user_id, rating.item_id)
        if pair in seen_pairs:
            raise ValueError(
                f"Duplicate training rating for user {rating.user_id}, "
                f"item {rating.item_id}"
            )
        seen_pairs.add(pair)
        row_indexes.append(user_index[rating.user_id])
        column_indexes.append(item_index[rating.item_id])
        values.append(float(rating.rating))

    shape = (len(ordered_users), len(ordered_items))
    user_item_matrix = coo_matrix(
        (values, (row_indexes, column_indexes)), shape=shape, dtype=float
    ).tocsr()
    centered = user_item_matrix.copy()
    for user_row in range(centered.shape[0]):
        start, end = centered.indptr[user_row : user_row + 2]
        if start != end:
            centered.data[start:end] -= centered.data[start:end].mean()
    centered.eliminate_zeros()

    if not ordered_items or centered.nnz == 0:
        item_similarity = csr_matrix(
            (len(ordered_items), len(ordered_items)), dtype=float
        )
    else:
        item_similarity = cosine_similarity(
            centered.transpose(), dense_output=False
        ).tocsr()
        item_similarity.data = item_similarity.data.clip(-1.0, 1.0)

    return ItemBasedModel(
        user_ids=ordered_users,
        item_ids=ordered_items,
        user_index=user_index,
        item_index=item_index,
        user_item_matrix=user_item_matrix,
        item_similarity=item_similarity,
    )


def recommend_for_user(
    user_id: int,
    model: ItemBasedModel,
    limit: int = 10,
) -> list[CollaborativeRecommendation]:
    """Rank unseen items by positive-similarity weighted training ratings.

    A candidate's score is ``sum(similarity(candidate, rated_item) * rating)``
    over the user's training ratings and positive similarities. Already-rated
    items are excluded. Ties are broken by ascending item ID. Users without
    training ratings, unknown users, and items without ratings produce no
    recommendations. Non-positive limits return an empty list.
    """
    if limit <= 0:
        return []

    user_row_index = model.user_index.get(user_id)
    if user_row_index is None:
        return []

    user_row = model.user_item_matrix.getrow(user_row_index)
    if user_row.nnz == 0:
        return []

    rated_item_indexes = set(user_row.indices)
    scores: defaultdict[int, float] = defaultdict(float)
    for rated_index, rating in zip(user_row.indices, user_row.data, strict=True):
        similarity_row = model.item_similarity.getrow(rated_index)
        for candidate_index, similarity in zip(
            similarity_row.indices, similarity_row.data, strict=True
        ):
            if candidate_index in rated_item_indexes or similarity <= 0:
                continue
            scores[candidate_index] += float(similarity) * float(rating)

    recommendations = [
        CollaborativeRecommendation(model.item_ids[index], score)
        for index, score in scores.items()
        if score > 0
    ]
    recommendations.sort(
        key=lambda recommendation: (-recommendation.score, recommendation.item_id)
    )
    return recommendations[:limit]
