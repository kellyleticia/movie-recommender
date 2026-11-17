from collections import defaultdict
from dataclasses import dataclass, replace

from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity

from app.collaborative_filtering import (
    CollaborativeRecommendation,
    ItemBasedModel,
    recommend_for_user,
)


@dataclass(frozen=True)
class ItemBasedVariant:
    name: str
    model: ItemBasedModel
    neighbor_limit: int | None


def build_uncentered_item_model(model: ItemBasedModel) -> ItemBasedModel:
    """Return a model with cosine similarity on original ratings, uncentered."""
    if model.user_item_matrix.nnz == 0:
        similarity = csr_matrix(
            (len(model.item_ids), len(model.item_ids)), dtype=float
        )
    else:
        similarity = cosine_similarity(
            model.user_item_matrix.transpose(), dense_output=False
        ).tocsr()
        similarity.data = similarity.data.clip(-1.0, 1.0)
    return replace(model, item_similarity=similarity)


def recommend_with_neighbor_limit(
    user_id: int,
    model: ItemBasedModel,
    *,
    neighbor_limit: int,
    limit: int = 10,
) -> list[CollaborativeRecommendation]:
    """Recommend using only the strongest positive rated-item neighbors.

    For each unseen candidate, select up to ``neighbor_limit`` of the user's
    already-rated items with greatest positive similarity. The score remains
    the sum of similarity times the original rating. Similarity ties for
    neighbor selection break by rated item ID; recommendation ties break by
    candidate item ID. A non-positive recommendation ``limit`` yields no rows.
    """
    if neighbor_limit <= 0:
        raise ValueError(
            f"Neighbor limit must be positive, got {neighbor_limit}"
        )
    if limit <= 0:
        return []

    user_row_index = model.user_index.get(user_id)
    if user_row_index is None:
        return []
    user_row = model.user_item_matrix.getrow(user_row_index)
    if user_row.nnz == 0:
        return []

    rated_item_indexes = set(user_row.indices)
    neighbors_by_candidate: defaultdict[int, list[tuple[float, int, float]]] = (
        defaultdict(list)
    )
    for rated_index, rating in zip(user_row.indices, user_row.data, strict=True):
        similarity_row = model.item_similarity.getrow(rated_index)
        for candidate_index, similarity in zip(
            similarity_row.indices, similarity_row.data, strict=True
        ):
            if candidate_index in rated_item_indexes or similarity <= 0:
                continue
            neighbors_by_candidate[candidate_index].append(
                (float(similarity), int(rated_index), float(rating))
            )

    recommendations: list[CollaborativeRecommendation] = []
    for candidate_index, neighbors in neighbors_by_candidate.items():
        strongest = sorted(
            neighbors,
            key=lambda neighbor: (
                -neighbor[0],
                model.item_ids[neighbor[1]],
            ),
        )[:neighbor_limit]
        strongest.sort(key=lambda neighbor: neighbor[1])
        score = sum(similarity * rating for similarity, _, rating in strongest)
        if score > 0:
            recommendations.append(
                CollaborativeRecommendation(model.item_ids[candidate_index], score)
            )

    recommendations.sort(
        key=lambda recommendation: (-recommendation.score, recommendation.item_id)
    )
    return recommendations[:limit]


def build_item_based_variants(
    model: ItemBasedModel,
    neighbor_limits: tuple[int, ...] = (20, 50, 100),
) -> tuple[ItemBasedVariant, ...]:
    """Create centered, uncentered, and centered top-neighbor variants."""
    if any(neighbor_limit <= 0 for neighbor_limit in neighbor_limits):
        raise ValueError("All neighbor limits must be positive integers")
    if len(set(neighbor_limits)) != len(neighbor_limits):
        raise ValueError("Neighbor limits must be unique")

    variants = [
        ItemBasedVariant("centered_all_neighbors", model, None),
        ItemBasedVariant(
            "uncentered_all_neighbors", build_uncentered_item_model(model), None
        ),
    ]
    variants.extend(
        ItemBasedVariant(
            f"centered_top_{neighbor_limit}_neighbors", model, neighbor_limit
        )
        for neighbor_limit in neighbor_limits
    )
    return tuple(variants)


def recommend_with_variant(
    user_id: int,
    variant: ItemBasedVariant,
    limit: int = 10,
) -> list[CollaborativeRecommendation]:
    """Dispatch to the established unlimited recommender or limited variant."""
    if variant.neighbor_limit is None:
        return recommend_for_user(user_id, variant.model, limit)
    return recommend_with_neighbor_limit(
        user_id,
        variant.model,
        neighbor_limit=variant.neighbor_limit,
        limit=limit,
    )
