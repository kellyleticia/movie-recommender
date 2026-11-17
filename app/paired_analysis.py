import random
from collections import defaultdict
from collections.abc import Collection, Sequence
from dataclasses import dataclass
from statistics import fmean

from app.collaborative_filtering import build_item_based_model, recommend_for_user
from app.collaborative_variants import build_uncentered_item_model
from app.evaluation import ndcg_at_k, precision_at_k, recall_at_k
from app.experiments import _validate_experiment_parameters
from app.movielens_splits import MovieLensSplit

K = 10
RELEVANCE_THRESHOLD = 4
DEFAULT_BOOTSTRAP_ITERATIONS = 10_000
DEFAULT_SEED = 42
METRIC_NAMES = ("precision_at_10", "recall_at_10", "ndcg_at_10")


@dataclass(frozen=True)
class BootstrapEstimate:
    mean_difference: float | None
    confidence_interval_lower: float | None
    confidence_interval_upper: float | None
    users: int


@dataclass(frozen=True)
class PairedFoldMetric:
    fold: int
    metric: str
    mean_difference: float | None
    confidence_interval_lower: float | None
    confidence_interval_upper: float | None
    users: int


@dataclass(frozen=True)
class PairedMacroMetric:
    metric: str
    folds: int
    folds_with_users: int
    mean_difference: float | None
    users_by_fold: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class PairedEvaluationResult:
    k: int
    relevance_threshold: int
    bootstrap_iterations: int
    seed: int
    fold_results: tuple[PairedFoldMetric, ...]
    macro_results: tuple[PairedMacroMetric, ...]


def bootstrap_paired_difference(
    centered_scores: Sequence[float],
    uncentered_scores: Sequence[float],
    *,
    iterations: int = DEFAULT_BOOTSTRAP_ITERATIONS,
    seed: int = DEFAULT_SEED,
) -> BootstrapEstimate:
    """Bootstrap paired per-user scores and return a percentile 95% interval.

    Each resample draws user indices with replacement, then retains the
    centered and uncentered score at each selected index as a pair. The
    bootstrap distribution is formed from per-user differences. Bounds use
    linear interpolation between empirical quantiles. Empty inputs have
    undefined estimates; arrays of unequal length and non-positive iteration
    counts are rejected.
    """
    if len(centered_scores) != len(uncentered_scores):
        raise ValueError("Paired score lists must have the same length")
    if iterations <= 0:
        raise ValueError(f"Bootstrap iterations must be positive, got {iterations}")
    if not centered_scores:
        return BootstrapEstimate(None, None, None, 0)

    paired_differences = [
        uncentered - centered
        for centered, uncentered in zip(
            centered_scores, uncentered_scores, strict=True
        )
    ]
    user_count = len(paired_differences)
    observed_mean = fmean(paired_differences)
    generator = random.Random(seed)
    bootstrap_means = sorted(
        sum(generator.choices(paired_differences, k=user_count)) / user_count
        for _ in range(iterations)
    )

    return BootstrapEstimate(
        mean_difference=observed_mean,
        confidence_interval_lower=_percentile(bootstrap_means, 0.025),
        confidence_interval_upper=_percentile(bootstrap_means, 0.975),
        users=user_count,
    )


def evaluate_paired_item_based_folds(
    splits: Sequence[MovieLensSplit],
    *,
    user_ids: Collection[int],
    item_ids: Collection[int],
    bootstrap_iterations: int = DEFAULT_BOOTSTRAP_ITERATIONS,
    seed: int = DEFAULT_SEED,
) -> PairedEvaluationResult:
    """Compare centered and uncentered Item-Based scores per test user.

    Both models use only each fold's training data and the same catalog. Every
    user in that fold's test ratings contributes Precision@10; only users with
    at least one test rating >=4 contribute Recall@10 and NDCG@10. Per-fold
    bootstrap samples users with replacement and never pools across folds.
    Each fold/metric uses a reproducible seed derived from ``seed``.
    """
    _validate_experiment_parameters(splits, K, RELEVANCE_THRESHOLD)
    if bootstrap_iterations <= 0:
        raise ValueError(
            "Bootstrap iterations must be positive, "
            f"got {bootstrap_iterations}"
        )

    fold_results: list[PairedFoldMetric] = []
    for split in sorted(splits, key=lambda item: item.fold):
        centered_model = build_item_based_model(
            split.train,
            user_ids=user_ids,
            item_ids=item_ids,
        )
        uncentered_model = build_uncentered_item_model(centered_model)

        relevant_by_user: defaultdict[int, set[int]] = defaultdict(set)
        test_users: set[int] = set()
        for rating in split.test:
            test_users.add(rating.user_id)
            if rating.rating >= RELEVANCE_THRESHOLD:
                relevant_by_user[rating.user_id].add(rating.item_id)

        precision_pairs: list[tuple[float, float]] = []
        recall_pairs: list[tuple[float, float]] = []
        ndcg_pairs: list[tuple[float, float]] = []
        for user_id in sorted(test_users):
            centered_recommendations = [
                recommendation.item_id
                for recommendation in recommend_for_user(
                    user_id, centered_model, limit=K
                )
            ]
            uncentered_recommendations = [
                recommendation.item_id
                for recommendation in recommend_for_user(
                    user_id, uncentered_model, limit=K
                )
            ]
            relevant_items = relevant_by_user[user_id]

            precision_pairs.append(
                (
                    precision_at_k(centered_recommendations, relevant_items, K),
                    precision_at_k(uncentered_recommendations, relevant_items, K),
                )
            )
            if relevant_items:
                centered_recall = recall_at_k(
                    centered_recommendations, relevant_items, K
                )
                uncentered_recall = recall_at_k(
                    uncentered_recommendations, relevant_items, K
                )
                centered_ndcg = ndcg_at_k(
                    centered_recommendations, relevant_items, K
                )
                uncentered_ndcg = ndcg_at_k(
                    uncentered_recommendations, relevant_items, K
                )
                assert centered_recall is not None and uncentered_recall is not None
                assert centered_ndcg is not None and uncentered_ndcg is not None
                recall_pairs.append((centered_recall, uncentered_recall))
                ndcg_pairs.append((centered_ndcg, uncentered_ndcg))

        pairs_by_metric = {
            "precision_at_10": precision_pairs,
            "recall_at_10": recall_pairs,
            "ndcg_at_10": ndcg_pairs,
        }
        for metric_index, metric in enumerate(METRIC_NAMES):
            pairs = pairs_by_metric[metric]
            estimate = bootstrap_paired_difference(
                [pair[0] for pair in pairs],
                [pair[1] for pair in pairs],
                iterations=bootstrap_iterations,
                seed=seed + split.fold * 100 + metric_index,
            )
            fold_results.append(
                PairedFoldMetric(
                    fold=split.fold,
                    metric=metric,
                    mean_difference=estimate.mean_difference,
                    confidence_interval_lower=estimate.confidence_interval_lower,
                    confidence_interval_upper=estimate.confidence_interval_upper,
                    users=estimate.users,
                )
            )

    macro_results = _macro_results(fold_results)
    return PairedEvaluationResult(
        k=K,
        relevance_threshold=RELEVANCE_THRESHOLD,
        bootstrap_iterations=bootstrap_iterations,
        seed=seed,
        fold_results=tuple(fold_results),
        macro_results=macro_results,
    )


def _macro_results(
    fold_results: Sequence[PairedFoldMetric],
) -> tuple[PairedMacroMetric, ...]:
    results: list[PairedMacroMetric] = []
    for metric in METRIC_NAMES:
        rows = sorted(
            (row for row in fold_results if row.metric == metric),
            key=lambda row: row.fold,
        )
        eligible_rows = [row for row in rows if row.mean_difference is not None]
        results.append(
            PairedMacroMetric(
                metric=metric,
                folds=len(rows),
                folds_with_users=len(eligible_rows),
                mean_difference=(
                    fmean(row.mean_difference for row in eligible_rows)
                    if eligible_rows
                    else None
                ),
                users_by_fold=tuple((row.fold, row.users) for row in rows),
            )
        )
    return tuple(results)


def _percentile(sorted_values: Sequence[float], probability: float) -> float:
    if not sorted_values:
        raise ValueError("Cannot calculate a percentile from an empty sequence")
    position = (len(sorted_values) - 1) * probability
    lower_index = int(position)
    upper_index = min(lower_index + 1, len(sorted_values) - 1)
    fraction = position - lower_index
    return sorted_values[lower_index] + fraction * (
        sorted_values[upper_index] - sorted_values[lower_index]
    )
