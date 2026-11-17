from collections import defaultdict
from collections.abc import Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass
from statistics import fmean

from app.analysis_metrics import (
    catalog_coverage,
    intra_list_diversity,
    recommendation_novelty,
)
from app.collaborative_filtering import build_item_based_model, recommend_for_user
from app.evaluation import aggregate_user_metrics
from app.movielens import MovieLensRating
from app.movielens_splits import MovieLensSplit
from app.popularity import build_popularity_model, recommend_by_popularity


@dataclass(frozen=True)
class FoldEvaluationResult:
    fold: int
    users_evaluated: int
    users_without_relevant: int
    recommendations_generated: int
    precision_at_k: float | None
    recall_at_k: float | None
    ndcg_at_k: float | None
    distinct_items_recommended: int
    catalog_size: int
    catalog_coverage: float
    intra_list_diversity: float | None
    diversity_users_evaluated: int
    diversity_users_without_comparable_pairs: int
    diversity_pairs_evaluated: int
    diversity_pairs_skipped_missing_genres: int
    novelty: float | None
    recommendations_without_training_ratings: int


@dataclass(frozen=True)
class OfflineEvaluationResult:
    k: int
    relevance_threshold: int
    folds: tuple[FoldEvaluationResult, ...]
    mean_precision_at_k: float | None
    mean_recall_at_k: float | None
    mean_ndcg_at_k: float | None
    mean_catalog_coverage: float | None
    mean_intra_list_diversity: float | None
    mean_novelty: float | None


def evaluate_item_based_folds(
    splits: Sequence[MovieLensSplit],
    *,
    user_ids: Collection[int],
    item_ids: Collection[int],
    genres_by_item: Mapping[int, Collection[str]],
    k: int = 10,
    relevance_threshold: int = 4,
) -> OfflineEvaluationResult:
    """Evaluate item-based recommendations over all five official folds.

    For each fold, the model is built only from ``split.train``. The provided
    item IDs are all catalog items, including those absent from training. Users
    are those appearing in that fold's test records. Test ratings greater than
    or equal to ``relevance_threshold`` define each user's relevant item set;
    missing ratings are never treated as negative labels.

    Precision includes all test users. Recall and NDCG include only test users
    with at least one relevant item; the excluded count is reported per fold.
    Fold metrics are macro-averaged equally, not pooled across users or ratings.
    """
    _validate_experiment_parameters(splits, k, relevance_threshold)
    ordered_splits = sorted(splits, key=lambda split: split.fold)
    fold_results: list[FoldEvaluationResult] = []
    for split in ordered_splits:
        model = build_item_based_model(
            split.train,
            user_ids=user_ids,
            item_ids=item_ids,
        )

        test_users = {rating.user_id for rating in split.test}
        relevant_items: defaultdict[int, set[int]] = defaultdict(set)
        for rating in split.test:
            if rating.rating >= relevance_threshold:
                relevant_items[rating.user_id].add(rating.item_id)

        recommendations_by_user = {
            user_id: [
                recommendation.item_id
                for recommendation in recommend_for_user(user_id, model, limit=k)
            ]
            for user_id in test_users
        }
        relevant_by_user = {
            user_id: relevant_items[user_id] for user_id in test_users
        }
        fold_results.append(
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

    return _experiment_result(k, relevance_threshold, fold_results)


def evaluate_popularity_folds(
    splits: Sequence[MovieLensSplit],
    *,
    item_ids: Collection[int],
    genres_by_item: Mapping[int, Collection[str]],
    k: int = 10,
    relevance_threshold: int = 4,
) -> OfflineEvaluationResult:
    """Evaluate a rating-count popularity baseline on the official five folds.

    The model and each user's seen-item set are built exclusively from that
    fold's training ratings. Items with no training ratings remain candidates
    with popularity zero and are ordered by ascending ID after rated items.
    Test users and relevance labels use the same definitions as the item-based
    experiment: users in test, with relevant items rated at least the threshold.
    Precision includes every test user; Recall and NDCG exclude users with no
    relevant test items. Fold metrics are macro-averaged equally.
    """
    _validate_experiment_parameters(splits, k, relevance_threshold)
    fold_results: list[FoldEvaluationResult] = []

    for split in sorted(splits, key=lambda item: item.fold):
        model = build_popularity_model(split.train, item_ids=item_ids)
        test_users = {rating.user_id for rating in split.test}
        relevant_items: defaultdict[int, set[int]] = defaultdict(set)
        for rating in split.test:
            if rating.rating >= relevance_threshold:
                relevant_items[rating.user_id].add(rating.item_id)

        recommendations_by_user = {
            user_id: [
                recommendation.item_id
                for recommendation in recommend_by_popularity(
                    user_id, model, limit=k
                )
            ]
            for user_id in test_users
        }
        relevant_by_user = {
            user_id: relevant_items[user_id] for user_id in test_users
        }
        fold_results.append(
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

    return _experiment_result(k, relevance_threshold, fold_results)


def _validate_experiment_parameters(
    splits: Sequence[MovieLensSplit], k: int, relevance_threshold: int
) -> None:
    if k <= 0:
        raise ValueError(f"K must be positive, got {k}")
    if not 1 <= relevance_threshold <= 5:
        raise ValueError(
            "Relevance threshold must be between 1 and 5, "
            f"got {relevance_threshold}"
        )
    fold_numbers = [split.fold for split in splits]
    if sorted(fold_numbers) != [1, 2, 3, 4, 5]:
        raise ValueError(
            "Expected exactly one split for each official fold 1 through 5, "
            f"got {fold_numbers}"
        )


def _fold_result(
    fold: int,
    recommendations_by_user: dict[int, list[int]],
    relevant_by_user: dict[int, set[int]],
    train_ratings: Sequence[MovieLensRating],
    catalog_item_ids: Collection[int],
    genres_by_item: Mapping[int, Collection[str]],
    k: int,
) -> FoldEvaluationResult:
    metrics = aggregate_user_metrics(recommendations_by_user, relevant_by_user, k)
    coverage = catalog_coverage(recommendations_by_user, catalog_item_ids)
    diversity = intra_list_diversity(recommendations_by_user, genres_by_item)
    novelty = recommendation_novelty(recommendations_by_user, train_ratings)
    return FoldEvaluationResult(
        fold=fold,
        users_evaluated=metrics.user_count,
        users_without_relevant=metrics.users_without_relevant,
        recommendations_generated=sum(
            len(recommendations) for recommendations in recommendations_by_user.values()
        ),
        precision_at_k=metrics.precision_at_k,
        recall_at_k=metrics.recall_at_k,
        ndcg_at_k=metrics.ndcg_at_k,
        distinct_items_recommended=coverage.distinct_items_recommended,
        catalog_size=coverage.catalog_size,
        catalog_coverage=coverage.value,
        intra_list_diversity=diversity.value,
        diversity_users_evaluated=diversity.users_evaluated,
        diversity_users_without_comparable_pairs=(
            diversity.users_without_comparable_pairs
        ),
        diversity_pairs_evaluated=diversity.pairs_evaluated,
        diversity_pairs_skipped_missing_genres=diversity.pairs_skipped_missing_genres,
        novelty=novelty.value,
        recommendations_without_training_ratings=(
            novelty.recommendations_without_training_ratings
        ),
    )


def _experiment_result(
    k: int, relevance_threshold: int, fold_results: list[FoldEvaluationResult]
) -> OfflineEvaluationResult:
    return OfflineEvaluationResult(
        k=k,
        relevance_threshold=relevance_threshold,
        folds=tuple(fold_results),
        mean_precision_at_k=_mean_fold_metric(
            fold.precision_at_k for fold in fold_results
        ),
        mean_recall_at_k=_mean_fold_metric(
            fold.recall_at_k for fold in fold_results
        ),
        mean_ndcg_at_k=_mean_fold_metric(fold.ndcg_at_k for fold in fold_results),
        mean_catalog_coverage=_mean_fold_metric(
            fold.catalog_coverage for fold in fold_results
        ),
        mean_intra_list_diversity=_mean_fold_metric(
            fold.intra_list_diversity for fold in fold_results
        ),
        mean_novelty=_mean_fold_metric(fold.novelty for fold in fold_results),
    )


def _mean_fold_metric(values: Iterable[float | None]) -> float | None:
    defined_values = [value for value in values if value is not None]
    return fmean(defined_values) if defined_values else None
