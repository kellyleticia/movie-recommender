from collections import defaultdict
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass

from app.collaborative_filtering import build_item_based_model
from app.collaborative_variants import (
    build_item_based_variants,
    recommend_with_variant,
)
from app.experiments import (
    FoldEvaluationResult,
    OfflineEvaluationResult,
    _experiment_result,
    _fold_result,
    _validate_experiment_parameters,
)
from app.movielens_splits import MovieLensSplit


@dataclass(frozen=True)
class VariantExperimentResult:
    variant_name: str
    neighbor_limit: int | None
    evaluation: OfflineEvaluationResult


def evaluate_item_based_variants(
    splits: Sequence[MovieLensSplit],
    *,
    user_ids: Collection[int],
    item_ids: Collection[int],
    genres_by_item: Mapping[int, Collection[str]],
    k: int = 10,
    relevance_threshold: int = 4,
    neighbor_limits: tuple[int, ...] = (20, 50, 100),
) -> tuple[VariantExperimentResult, ...]:
    """Compare isolated Item-Based variants over the official five folds.

    Each fold's variants derive from one model built with its training ratings.
    Test ratings provide only relevance labels; all variants use the same users,
    complete catalog, metrics, K, and relevance threshold.
    """
    _validate_experiment_parameters(splits, k, relevance_threshold)
    if any(value <= 0 for value in neighbor_limits):
        raise ValueError("All neighbor limits must be positive integers")
    if len(set(neighbor_limits)) != len(neighbor_limits):
        raise ValueError("Neighbor limits must be unique")

    folds_by_variant: dict[str, list[FoldEvaluationResult]] = {}
    limits_by_variant: dict[str, int | None] = {}

    for split in sorted(splits, key=lambda item: item.fold):
        reference_model = build_item_based_model(
            split.train,
            user_ids=user_ids,
            item_ids=item_ids,
        )
        variants = build_item_based_variants(reference_model, neighbor_limits)
        for variant in variants:
            limits_by_variant[variant.name] = variant.neighbor_limit
            test_users = {rating.user_id for rating in split.test}
            relevant_items: defaultdict[int, set[int]] = defaultdict(set)
            for rating in split.test:
                if rating.rating >= relevance_threshold:
                    relevant_items[rating.user_id].add(rating.item_id)

            recommendations_by_user = {
                user_id: [
                    recommendation.item_id
                    for recommendation in recommend_with_variant(
                        user_id, variant, limit=k
                    )
                ]
                for user_id in test_users
            }
            relevant_by_user = {
                user_id: relevant_items[user_id] for user_id in test_users
            }
            folds_by_variant.setdefault(variant.name, []).append(
                _fold_result(
                    split.fold,
                    recommendations_by_user,
                    relevant_by_user,
                    split.train,
                    item_ids,
                    genres_by_item,
                    k,
                )
            )

    results = []
    for variant_name, fold_results in folds_by_variant.items():
        results.append(
            VariantExperimentResult(
                variant_name=variant_name,
                neighbor_limit=limits_by_variant[variant_name],
                evaluation=_experiment_result(k, relevance_threshold, fold_results),
            )
        )
    return tuple(results)
