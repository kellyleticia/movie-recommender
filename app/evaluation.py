from collections.abc import Hashable, Mapping, Sequence, Set
from dataclasses import dataclass
from math import log2
from statistics import fmean


@dataclass(frozen=True)
class UserMetricAggregate:
    """Macro-averaged ranking metrics and the number of users evaluated."""

    precision_at_k: float | None
    recall_at_k: float | None
    ndcg_at_k: float | None
    user_count: int
    users_without_relevant: int


def precision_at_k[MovieId: Hashable](
    recommendations: Sequence[MovieId],
    relevant_items: Set[MovieId],
    k: int,
) -> float:
    """Return binary Precision@K, using K as the denominator.

    Recommendation IDs are deduplicated in ranking order before taking the top
    K. A short or empty list therefore has missing positions counted as misses.
    Relevant items are supplied by the caller, for example from held-out
    ratings with a preselected threshold such as rating >= 4. K must be positive.
    """
    _validate_k(k)
    top_items = _unique_items(recommendations)[:k]
    hits = sum(item in relevant_items for item in top_items)
    return hits / k


def recall_at_k[MovieId: Hashable](
    recommendations: Sequence[MovieId],
    relevant_items: Set[MovieId],
    k: int,
) -> float | None:
    """Return binary Recall@K, or None when there are no relevant items.

    Recommendation IDs are deduplicated in ranking order. Relevant items must
    be defined externally; an empty relevant set makes recall undefined.
    K must be positive.
    """
    _validate_k(k)
    if not relevant_items:
        return None
    top_items = _unique_items(recommendations)[:k]
    hits = sum(item in relevant_items for item in top_items)
    return hits / len(relevant_items)


def ndcg_at_k[MovieId: Hashable](
    recommendations: Sequence[MovieId],
    relevant_items: Set[MovieId],
    k: int,
) -> float | None:
    """Return binary NDCG@K with logarithmic rank discount.

    A relevant item has gain 1 and any other item gain 0. The discount at
    1-based rank ``r`` is ``1 / log2(r + 1)``. Recommendations are deduplicated
    in ranking order. NDCG is undefined (None) when there are no relevant items;
    relevance labels are supplied externally. K must be positive.
    """
    _validate_k(k)
    if not relevant_items:
        return None

    top_items = _unique_items(recommendations)[:k]
    dcg = sum(
        1 / log2(rank + 1)
        for rank, item in enumerate(top_items, start=1)
        if item in relevant_items
    )
    ideal_count = min(k, len(relevant_items))
    ideal_dcg = sum(1 / log2(rank + 1) for rank in range(1, ideal_count + 1))
    return dcg / ideal_dcg


def aggregate_user_metrics[UserId: Hashable, MovieId: Hashable](
    recommendations_by_user: Mapping[UserId, Sequence[MovieId]],
    relevant_items_by_user: Mapping[UserId, Set[MovieId]],
    k: int,
) -> UserMetricAggregate:
    """Macro-average metrics across users and report users without positives.

    Precision includes every user present in either mapping. Users whose
    relevant set is absent or empty are excluded from Recall@K and NDCG@K and
    counted in ``users_without_relevant``. Empty aggregate means are None.
    """
    _validate_k(k)
    user_ids = set(recommendations_by_user) | set(relevant_items_by_user)
    if not user_ids:
        return UserMetricAggregate(None, None, None, 0, 0)

    precisions: list[float] = []
    recalls: list[float] = []
    ndcgs: list[float] = []
    users_without_relevant = 0

    for user_id in user_ids:
        recommendations = recommendations_by_user.get(user_id, ())
        relevant_items = relevant_items_by_user.get(user_id, frozenset())
        precisions.append(precision_at_k(recommendations, relevant_items, k))

        if not relevant_items:
            users_without_relevant += 1
            continue

        recall = recall_at_k(recommendations, relevant_items, k)
        ndcg = ndcg_at_k(recommendations, relevant_items, k)
        assert recall is not None
        assert ndcg is not None
        recalls.append(recall)
        ndcgs.append(ndcg)

    return UserMetricAggregate(
        precision_at_k=fmean(precisions) if precisions else None,
        recall_at_k=fmean(recalls) if recalls else None,
        ndcg_at_k=fmean(ndcgs) if ndcgs else None,
        user_count=len(user_ids),
        users_without_relevant=users_without_relevant,
    )


def _unique_items[MovieId: Hashable](items: Sequence[MovieId]) -> list[MovieId]:
    seen: set[MovieId] = set()
    unique: list[MovieId] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            unique.append(item)
    return unique


def _validate_k(k: int) -> None:
    if k <= 0:
        raise ValueError(f"K must be a positive integer, got {k}")
