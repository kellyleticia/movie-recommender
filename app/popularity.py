from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass

from app.movielens import MovieLensRating


@dataclass(frozen=True)
class PopularityModel:
    """Training-set rating counts and per-user seen items."""

    item_ids: tuple[int, ...]
    rating_counts: dict[int, int]
    user_rated_items: dict[int, frozenset[int]]


@dataclass(frozen=True)
class PopularityRecommendation:
    item_id: int
    rating_count: int


def build_popularity_model(
    train_ratings: Iterable[MovieLensRating],
    *,
    item_ids: Iterable[int] = (),
) -> PopularityModel:
    """Count training ratings per item, optionally retaining unrated catalog IDs."""
    ratings = list(train_ratings)
    counts = Counter(rating.item_id for rating in ratings)
    user_rated: defaultdict[int, set[int]] = defaultdict(set)
    for rating in ratings:
        user_rated[rating.user_id].add(rating.item_id)

    all_item_ids = set(item_ids) | set(counts)
    if any(item_id <= 0 for item_id in all_item_ids):
        raise ValueError("Item IDs must be positive integers")

    return PopularityModel(
        item_ids=tuple(sorted(all_item_ids)),
        rating_counts={item_id: counts[item_id] for item_id in all_item_ids},
        user_rated_items={
            user_id: frozenset(seen_items) for user_id, seen_items in user_rated.items()
        },
    )


def recommend_by_popularity(
    user_id: int,
    model: PopularityModel,
    limit: int = 10,
) -> list[PopularityRecommendation]:
    """Rank unseen movies by training rating count, then ascending item ID."""
    if limit <= 0:
        return []

    seen_items = model.user_rated_items.get(user_id, frozenset())
    candidates = (item_id for item_id in model.item_ids if item_id not in seen_items)
    ordered_items = sorted(
        candidates,
        key=lambda item_id: (-model.rating_counts[item_id], item_id),
    )
    return [
        PopularityRecommendation(item_id, model.rating_counts[item_id])
        for item_id in ordered_items[:limit]
    ]
