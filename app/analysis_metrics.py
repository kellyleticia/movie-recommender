from collections import Counter
from collections.abc import Collection, Hashable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from itertools import combinations
from math import inf, log2
from statistics import fmean

from app.movielens import MovieLensRating


@dataclass(frozen=True)
class CatalogCoverage:
    value: float
    distinct_items_recommended: int
    catalog_size: int


@dataclass(frozen=True)
class IntraListDiversity:
    value: float | None
    users_evaluated: int
    users_without_comparable_pairs: int
    pairs_evaluated: int
    pairs_skipped_missing_genres: int


@dataclass(frozen=True)
class Novelty:
    value: float | None
    recommendations_count: int
    recommendations_without_training_ratings: int


def catalog_coverage[UserId: Hashable, ItemId: Hashable](
    recommendations_by_user: Mapping[UserId, Sequence[ItemId]],
    catalog_item_ids: Collection[ItemId],
) -> CatalogCoverage:
    """Measure distinct recommended catalog items divided by catalog size.

    Empty catalogs have coverage 0. Recommendations outside the supplied
    catalog are rejected instead of being silently counted or discarded.
    """
    catalog = set(catalog_item_ids)
    recommended = {
        item_id
        for recommendations in recommendations_by_user.values()
        for item_id in recommendations
    }
    outside_catalog = recommended - catalog
    if outside_catalog:
        raise ValueError(
            f"Recommendations contain IDs outside the catalog: "
            f"{sorted(outside_catalog, key=str)}"
        )
    distinct_count = len(recommended)
    value = distinct_count / len(catalog) if catalog else 0.0
    return CatalogCoverage(value, distinct_count, len(catalog))


def intra_list_diversity[UserId: Hashable, ItemId: Hashable](
    recommendations_by_user: Mapping[UserId, Sequence[ItemId]],
    genres_by_item: Mapping[ItemId, Collection[str]],
) -> IntraListDiversity:
    """Macro-average per-user mean pairwise genre diversity.

    Pairwise similarity is Jaccard and diversity is ``1 - similarity``.
    Recommendations are deduplicated per user. Pairs involving a movie with
    no known genres are omitted because missing metadata is not evidence of
    dissimilarity. Users with fewer than two unique recommendations are not
    eligible; users with no comparable genre pairs are counted and excluded
    from the macro mean.
    """
    per_user_values: list[float] = []
    users_without_comparable_pairs = 0
    pairs_evaluated = 0
    pairs_skipped = 0

    for recommendations in recommendations_by_user.values():
        unique_items = list(dict.fromkeys(recommendations))
        if len(unique_items) < 2:
            continue

        user_pair_values: list[float] = []
        for left_item, right_item in combinations(unique_items, 2):
            left_genres = {
                genre.casefold() for genre in genres_by_item.get(left_item, ())
            }
            right_genres = {
                genre.casefold() for genre in genres_by_item.get(right_item, ())
            }
            if not left_genres or not right_genres:
                pairs_skipped += 1
                continue

            union = left_genres | right_genres
            similarity = len(left_genres & right_genres) / len(union)
            user_pair_values.append(1.0 - similarity)

        if user_pair_values:
            per_user_values.append(fmean(user_pair_values))
            pairs_evaluated += len(user_pair_values)
        else:
            users_without_comparable_pairs += 1

    return IntraListDiversity(
        value=fmean(per_user_values) if per_user_values else None,
        users_evaluated=len(per_user_values),
        users_without_comparable_pairs=users_without_comparable_pairs,
        pairs_evaluated=pairs_evaluated,
        pairs_skipped_missing_genres=pairs_skipped,
    )


def recommendation_novelty[UserId: Hashable, ItemId: Hashable](
    recommendations_by_user: Mapping[UserId, Sequence[ItemId]],
    train_ratings: Iterable[MovieLensRating],
) -> Novelty:
    """Average self-information from training-set item rating frequencies.

    For a nonempty training set, popularity is ``count(item) / rating_count``.
    An item with zero training ratings has probability zero and novelty +infinity,
    the mathematical limit of ``-log2(p)`` as p approaches zero. With no
    training ratings, the probability distribution is undefined and novelty is
    None. With no recommendations, the mean is also None.
    """
    ratings = list(train_ratings)
    counts = Counter(rating.item_id for rating in ratings)
    recommended_items = [
        item_id
        for recommendations in recommendations_by_user.values()
        for item_id in recommendations
    ]
    zero_count = sum(counts[item_id] == 0 for item_id in recommended_items)
    if not recommended_items or not ratings:
        return Novelty(None, len(recommended_items), zero_count)

    total_ratings = len(ratings)
    values = [
        inf
        if counts[item_id] == 0
        else -log2(counts[item_id] / total_ratings)
        for item_id in recommended_items
    ]
    return Novelty(fmean(values), len(recommended_items), zero_count)
