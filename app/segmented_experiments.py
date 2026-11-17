from collections import Counter, defaultdict
from collections.abc import Collection, Iterable, Sequence
from dataclasses import dataclass
from statistics import fmean

from app.analysis_metrics import catalog_coverage
from app.collaborative_filtering import (
    build_item_based_model,
    recommend_for_user,
)
from app.collaborative_variants import build_uncentered_item_model
from app.evaluation import aggregate_user_metrics
from app.experiments import _validate_experiment_parameters
from app.movielens_splits import MovieLensSplit
from app.popularity import build_popularity_model, recommend_by_popularity

HISTORY_SEGMENTS = ("1-5", "6-10", "11-20", "21+")
MODEL_NAMES = ("item_based_centered", "item_based_uncentered", "popularity")


@dataclass(frozen=True)
class SegmentFoldResult:
    fold: int
    model_name: str
    history_segment: str
    users: int
    users_without_relevant: int
    precision_at_k: float | None
    recall_at_k: float | None
    ndcg_at_k: float | None
    distinct_items_recommended: int
    catalog_size: int
    catalog_coverage: float | None


@dataclass(frozen=True)
class SegmentMacroResult:
    model_name: str
    history_segment: str
    folds: int
    folds_with_users: int
    folds_with_relevant_users: int
    mean_users_per_fold: float
    mean_precision_at_k: float | None
    mean_recall_at_k: float | None
    mean_ndcg_at_k: float | None
    mean_catalog_coverage: float | None


@dataclass(frozen=True)
class SegmentedEvaluationResult:
    relevance_threshold: int
    k: int
    fold_results: tuple[SegmentFoldResult, ...]
    macro_results: tuple[SegmentMacroResult, ...]
    users_without_training_history_by_fold: tuple[tuple[int, int], ...]


def evaluate_by_training_history(
    splits: Sequence[MovieLensSplit],
    *,
    user_ids: Collection[int],
    item_ids: Collection[int],
    k: int = 10,
    relevance_threshold: int = 4,
) -> SegmentedEvaluationResult:
    """Evaluate current Item-Based, uncentered, and popularity by train history.

    Only users appearing in the fold's test set are evaluated. Their segment is
    determined solely by the number of training ratings in that fold. Users
    with no training ratings are omitted from the four requested ranges and
    reported separately. Precision includes all users in each segment; Recall
    and NDCG use the existing evaluator's empty-relevance exclusion rule.
    """
    _validate_experiment_parameters(splits, k, relevance_threshold)
    fold_results: list[SegmentFoldResult] = []
    unsegmented_by_fold: list[tuple[int, int]] = []

    for split in sorted(splits, key=lambda item: item.fold):
        centered_model = build_item_based_model(
            split.train,
            user_ids=user_ids,
            item_ids=item_ids,
        )
        uncentered_model = build_uncentered_item_model(centered_model)
        popularity_model = build_popularity_model(split.train, item_ids=item_ids)
        train_counts = Counter(rating.user_id for rating in split.train)

        test_users = {rating.user_id for rating in split.test}
        users_by_segment: dict[str, set[int]] = {
            segment: set() for segment in HISTORY_SEGMENTS
        }
        users_without_training_history = 0
        for user_id in test_users:
            rating_count = train_counts[user_id]
            segment = _history_segment(rating_count)
            if segment is None:
                users_without_training_history += 1
            else:
                users_by_segment[segment].add(user_id)
        unsegmented_by_fold.append((split.fold, users_without_training_history))

        relevant_items: defaultdict[int, set[int]] = defaultdict(set)
        for rating in split.test:
            if rating.rating >= relevance_threshold:
                relevant_items[rating.user_id].add(rating.item_id)

        recommendations_by_model = {
            "item_based_centered": {
                user_id: [
                    recommendation.item_id
                    for recommendation in recommend_for_user(
                        user_id, centered_model, limit=k
                    )
                ]
                for user_id in test_users
            },
            "item_based_uncentered": {
                user_id: [
                    recommendation.item_id
                    for recommendation in recommend_for_user(
                        user_id, uncentered_model, limit=k
                    )
                ]
                for user_id in test_users
            },
            "popularity": {
                user_id: [
                    recommendation.item_id
                    for recommendation in recommend_by_popularity(
                        user_id, popularity_model, limit=k
                    )
                ]
                for user_id in test_users
            },
        }

        for model_name in MODEL_NAMES:
            recommendations = recommendations_by_model[model_name]
            for segment in HISTORY_SEGMENTS:
                segment_users = users_by_segment[segment]
                segment_recommendations = {
                    user_id: recommendations[user_id] for user_id in segment_users
                }
                segment_relevant = {
                    user_id: relevant_items[user_id] for user_id in segment_users
                }
                user_metrics = aggregate_user_metrics(
                    segment_recommendations, segment_relevant, k
                )
                coverage = (
                    catalog_coverage(segment_recommendations, item_ids)
                    if segment_users
                    else None
                )
                fold_results.append(
                    SegmentFoldResult(
                        fold=split.fold,
                        model_name=model_name,
                        history_segment=segment,
                        users=user_metrics.user_count,
                        users_without_relevant=user_metrics.users_without_relevant,
                        precision_at_k=user_metrics.precision_at_k,
                        recall_at_k=user_metrics.recall_at_k,
                        ndcg_at_k=user_metrics.ndcg_at_k,
                        distinct_items_recommended=(
                            coverage.distinct_items_recommended if coverage else 0
                        ),
                        catalog_size=len(item_ids),
                        catalog_coverage=coverage.value if coverage else None,
                    )
                )

    macro_results = _macro_results(fold_results)
    return SegmentedEvaluationResult(
        relevance_threshold=relevance_threshold,
        k=k,
        fold_results=tuple(fold_results),
        macro_results=macro_results,
        users_without_training_history_by_fold=tuple(unsegmented_by_fold),
    )


def _history_segment(rating_count: int) -> str | None:
    if 1 <= rating_count <= 5:
        return "1-5"
    if 6 <= rating_count <= 10:
        return "6-10"
    if 11 <= rating_count <= 20:
        return "11-20"
    if rating_count >= 21:
        return "21+"
    return None


def _macro_results(
    fold_results: Sequence[SegmentFoldResult],
) -> tuple[SegmentMacroResult, ...]:
    macros: list[SegmentMacroResult] = []
    for model_name in MODEL_NAMES:
        for segment in HISTORY_SEGMENTS:
            folds = [
                result
                for result in fold_results
                if result.model_name == model_name
                and result.history_segment == segment
            ]
            folds_with_users = [result for result in folds if result.users > 0]
            folds_with_relevant = [
                result
                for result in folds_with_users
                if result.users > result.users_without_relevant
            ]
            macros.append(
                SegmentMacroResult(
                    model_name=model_name,
                    history_segment=segment,
                    folds=len(folds),
                    folds_with_users=len(folds_with_users),
                    folds_with_relevant_users=len(folds_with_relevant),
                    mean_users_per_fold=(
                        fmean(result.users for result in folds) if folds else 0.0
                    ),
                    mean_precision_at_k=_mean(
                        result.precision_at_k for result in folds_with_users
                    ),
                    mean_recall_at_k=_mean(
                        result.recall_at_k for result in folds_with_relevant
                    ),
                    mean_ndcg_at_k=_mean(
                        result.ndcg_at_k for result in folds_with_relevant
                    ),
                    mean_catalog_coverage=_mean(
                        result.catalog_coverage for result in folds_with_users
                    ),
                )
            )
    return tuple(macros)


def _mean(values: Iterable[float | None]) -> float | None:
    defined = [value for value in values if value is not None]
    return fmean(defined) if defined else None
