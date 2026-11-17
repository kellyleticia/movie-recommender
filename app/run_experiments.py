import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any

from app.experiments import evaluate_item_based_folds, evaluate_popularity_folds
from app.models import Movie
from app.movielens import load_movielens_movies
from app.movielens_splits import MovieLensSplit, load_movielens_splits
from app.paired_analysis import (
    DEFAULT_BOOTSTRAP_ITERATIONS,
    DEFAULT_SEED,
    evaluate_paired_item_based_folds,
)
from app.segmented_experiments import (
    HISTORY_SEGMENTS,
    evaluate_by_training_history,
)
from app.variant_experiments import evaluate_item_based_variants

DEFAULT_DATA_DIR = Path("data/movielens/raw/ml-100k")
DEFAULT_REPORTS_DIR = Path("reports")
K = 10
RELEVANCE_THRESHOLD = 4
NEIGHBOR_LIMITS = (20, 50, 100)


@dataclass(frozen=True)
class ExperimentData:
    movies: tuple[Movie, ...]
    user_ids: frozenset[int]
    splits: tuple[MovieLensSplit, ...]


def experiment_configuration() -> dict[str, Any]:
    """Return the fixed, documented settings shared by the experiment suite."""
    return {
        "dataset": "MovieLens 100K",
        "official_folds": [1, 2, 3, 4, 5],
        "catalog_size": "all movies loaded from u.item",
        "k": K,
        "relevance_threshold": RELEVANCE_THRESHOLD,
        "relevance_rule": "test rating >= 4",
        "neighbor_limits": list(NEIGHBOR_LIMITS),
        "history_segments": list(HISTORY_SEGMENTS),
        "paired_bootstrap": {
            "iterations": DEFAULT_BOOTSTRAP_ITERATIONS,
            "seed": DEFAULT_SEED,
            "confidence_level": 0.95,
            "method": "paired user percentile bootstrap, within each fold",
        },
    }


def load_experiment_data(data_dir: str | Path) -> ExperimentData:
    """Load the local MovieLens metadata and official five train/test folds."""
    root = Path(data_dir)
    required_files = [
        root / "u.item",
        root / "u.genre",
        root / "u.user",
        *(
            root / f"u{fold}.{part}"
            for fold in range(1, 6)
            for part in ("base", "test")
        ),
    ]
    missing = [path for path in required_files if not path.is_file()]
    if missing:
        missing_list = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(
            "MovieLens 100K files are missing. Download the official ml-100k.zip "
            "from https://grouplens.org/datasets/movielens/100k/, extract it, "
            f"and place its files in '{root}'. Missing files:\n{missing_list}"
        )

    movies = tuple(load_movielens_movies(root / "u.item", root / "u.genre"))
    item_ids = {int(movie.id) for movie in movies}
    user_ids = _load_user_ids(root / "u.user")
    splits = load_movielens_splits(
        root,
        expected_user_ids=user_ids,
        expected_item_ids=item_ids,
    )
    return ExperimentData(movies, frozenset(user_ids), splits)


def run_experiment_suite(
    *,
    data_dir: str | Path = DEFAULT_DATA_DIR,
    reports_dir: str | Path = DEFAULT_REPORTS_DIR,
) -> tuple[dict[str, Any], Path, Path]:
    """Run all established MovieLens evaluations and write JSON and Markdown."""
    data = load_experiment_data(data_dir)
    item_ids = {int(movie.id) for movie in data.movies}
    genres_by_item = {int(movie.id): movie.genres for movie in data.movies}

    item_based = evaluate_item_based_folds(
        data.splits,
        user_ids=data.user_ids,
        item_ids=item_ids,
        genres_by_item=genres_by_item,
        k=K,
        relevance_threshold=RELEVANCE_THRESHOLD,
    )
    popularity = evaluate_popularity_folds(
        data.splits,
        item_ids=item_ids,
        genres_by_item=genres_by_item,
        k=K,
        relevance_threshold=RELEVANCE_THRESHOLD,
    )
    variants = evaluate_item_based_variants(
        data.splits,
        user_ids=data.user_ids,
        item_ids=item_ids,
        genres_by_item=genres_by_item,
        k=K,
        relevance_threshold=RELEVANCE_THRESHOLD,
        neighbor_limits=NEIGHBOR_LIMITS,
    )
    segmented = evaluate_by_training_history(
        data.splits,
        user_ids=data.user_ids,
        item_ids=item_ids,
        k=K,
        relevance_threshold=RELEVANCE_THRESHOLD,
    )
    paired = evaluate_paired_item_based_folds(
        data.splits,
        user_ids=data.user_ids,
        item_ids=item_ids,
        bootstrap_iterations=DEFAULT_BOOTSTRAP_ITERATIONS,
        seed=DEFAULT_SEED,
    )

    payload = {
        "configuration": experiment_configuration(),
        "dataset": {
            "catalog_movies": len(data.movies),
            "genre_count": len(
                {genre for movie in data.movies for genre in movie.genres}
            ),
            "user_count": len(data.user_ids),
            "folds": [
                {
                    "fold": split.fold,
                    "training_ratings": len(split.train),
                    "test_ratings": len(split.test),
                    "test_users": len({rating.user_id for rating in split.test}),
                }
                for split in data.splits
            ],
        },
        "evaluations": {
            "item_based": item_based,
            "popularity": popularity,
            "variants": variants,
            "segmented": segmented,
            "paired_bootstrap": paired,
        },
    }
    reports_path = Path(reports_dir)
    json_path = reports_path / "experiment_results.json"
    markdown_path = reports_path / "experiment_report.md"
    write_reports(payload, json_path=json_path, markdown_path=markdown_path)
    return payload, json_path, markdown_path


def write_reports(
    payload: dict[str, Any], *, json_path: str | Path, markdown_path: str | Path
) -> None:
    """Write deterministic JSON and Markdown representations of results."""
    json_file = Path(json_path)
    markdown_file = Path(markdown_path)
    json_file.parent.mkdir(parents=True, exist_ok=True)
    markdown_file.parent.mkdir(parents=True, exist_ok=True)
    safe_payload = _to_json_safe(payload)
    json_file.write_text(
        json.dumps(safe_payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    markdown_file.write_text(render_markdown_report(safe_payload), encoding="utf-8")


def render_markdown_report(payload: dict[str, Any]) -> str:
    """Render a deterministic human-readable report from the result payload."""
    configuration = payload["configuration"]
    dataset = payload["dataset"]
    evaluations = payload["evaluations"]
    lines = [
        "# Relatório de experimentos — MovieLens 100K",
        "",
        "## Dados e configurações",
        "",
        "Avaliação feita no MovieLens 100K local, usando os cinco pares oficiais "
        "`u1.base/u1.test` até `u5.base/u5.test`. Em cada fold, o modelo é "
        "construído somente com o arquivo `.base`; o `.test` fornece relevância. "
        "O catálogo completo de `u.item` permanece disponível, mesmo para filmes "
        "sem avaliações no treino.",
        "",
        f"- Filmes no catálogo: **{dataset['catalog_movies']}**.",
        f"- Gêneros no catálogo: **{dataset['genre_count']}**.",
        f"- Usuários: **{dataset['user_count']}**.",
        f"- K: **{configuration['k']}**; relevância: "
        f"**{configuration['relevance_rule']}**.",
        f"- Variantes top-K: **{configuration['neighbor_limits']} vizinhos**.",
        f"- Bootstrap pareado: **{configuration['paired_bootstrap']['iterations']:,}** "
        f"reamostragens; semente **{configuration['paired_bootstrap']['seed']}**; "
        "IC percentil de 95%.",
        "",
        "| Fold | Avaliações de treino | Avaliações de teste | Usuários no teste |",
        "|---:|---:|---:|---:|",
    ]
    lines.extend(
        f"| {row['fold']} | {row['training_ratings']} | {row['test_ratings']} | "
        f"{row['test_users']} |"
        for row in dataset["folds"]
    )
    lines.extend(
        [
            "",
            "## Metodologia e métricas",
            "",
            "Precision@10 usa todos os usuários do teste e divide acertos por 10; "
            "listas menores contam posições ausentes como não acertos. Recall@10 "
            "e NDCG@10 omitem usuários sem itens relevantes. Itens relevantes são "
            "ratings de teste >= 4; ausência de rating não é tratada como nota "
            "negativa. Cobertura é a fração do catálogo que aparece nas listas. "
            "Diversidade é a média macro por usuário de `1 - Jaccard` dos gêneros; "
            "pares sem gêneros disponíveis não são comparáveis. Novidade é a média "
            "de `-log2(freqência no treino / ratings de treino)`. As médias por fold "
            "são macro, com peso igual entre folds definidos.",
            "",
            "## Item-Based atual e baseline de popularidade",
            "",
            "A popularidade é contada exclusivamente em cada `.base`; recomendações "
            "excluem filmes já avaliados pelo usuário nesse treino.",
            "",
            _core_results_table(evaluations["item_based"], evaluations["popularity"]),
            "",
            "## Variantes Item-Based",
            "",
            "A variante sem centralização usa cosseno nos ratings originais. As "
            "variantes top-K limitam vizinhos positivos por filme candidato. O "
            "Item-Based centralizado sem limite permanece como referência.",
            "",
            _variant_results_table(evaluations["variants"]),
            "",
            "## Resultados segmentados por histórico de treino",
            "",
            "Cada usuário é classificado pela contagem de avaliações em `split.train`: "
            "1–5, 6–10, 11–20 ou 21+. Precision inclui todos os usuários; Recall e "
            "NDCG omitem os sem positivos. A cobertura de cada segmento usa o "
            "catálogo completo.",
            "",
            _segmented_macro_table(evaluations["segmented"]),
            "",
            "### Resultados segmentados por fold",
            "",
            _segmented_fold_table(evaluations["segmented"]),
            "",
            "## Comparação pareada e bootstrap",
            "",
            "Diferença = variante sem centralização menos Item-Based centralizado. "
            "O bootstrap reamostra usuários com reposição separadamente dentro "
            "de cada fold, mantendo os dois resultados do usuário pareados. "
            "Os limites usam quantis percentis com interpolação linear. A média "
            "macro dá peso igual aos folds com usuários elegíveis.",
            "",
            _paired_results_table(evaluations["paired_bootstrap"]),
            "",
            "## Limitações e interpretação",
            "",
            "Os resultados descrevem somente esta execução e este protocolo no "
            "MovieLens 100K. Os cinco folds compartilham usuários e não são "
            "amostras independentes; intervalos por fold não provam superioridade "
            "universal nem garantem desempenho em outros catálogos. A métrica de "
            "relevância binária ignora a diferença entre ratings 4 e 5, e avaliações "
            "não observadas não significam desinteresse. Gêneros são metadados "
            "amplos, e a novidade depende da distribuição de treino. Resultados "
            "indefinidos são exibidos como `—` e como `null` no JSON, nunca como zero.",
            "",
        ]
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Executa a avaliação offline completa do MovieLens 100K."
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--reports-dir", type=Path, default=DEFAULT_REPORTS_DIR)
    args = parser.parse_args(argv)
    try:
        _, json_path, markdown_path = run_experiment_suite(
            data_dir=args.data_dir,
            reports_dir=args.reports_dir,
        )
    except FileNotFoundError as error:
        print(str(error), file=sys.stderr)
        return 2
    print(f"JSON: {json_path}")
    print(f"Markdown: {markdown_path}")
    return 0


def _load_user_ids(path: Path) -> set[int]:
    user_ids: set[int] = set()
    with path.open(encoding="ascii") as stream:
        for row_number, line in enumerate(stream, start=1):
            fields = line.rstrip("\r\n").split("|")
            if len(fields) != 5:
                raise ValueError(
                    f"Invalid u.user row {row_number} in '{path}': expected "
                    f"5 pipe-separated fields, got {len(fields)}"
                )
            try:
                user_id = int(fields[0])
            except ValueError:
                raise ValueError(
                    f"Invalid user ID in u.user row {row_number} of '{path}': "
                    f"{fields[0]!r}"
                ) from None
            if user_id <= 0 or user_id in user_ids:
                raise ValueError(
                    f"Invalid or duplicate user ID {user_id} in u.user row "
                    f"{row_number} of '{path}'"
                )
            user_ids.add(user_id)
    if not user_ids:
        raise ValueError(f"MovieLens u.user file contains no users: '{path}'")
    return user_ids


def _to_json_safe(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return _to_json_safe(asdict(value))
    if isinstance(value, dict):
        return {key: _to_json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return "Infinity" if value > 0 else "-Infinity"
    return value


def _format(value: Any, digits: int = 4) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        if math.isinf(value):
            return "+∞" if value > 0 else "−∞"
        return f"{value:.{digits}f}"
    return str(value)


def _core_results_table(item_based: dict[str, Any], popularity: dict[str, Any]) -> str:
    rows = [
        ("Item-Based", item_based),
        ("Popularidade", popularity),
    ]
    lines = [
        "| Fold | Modelo | Usuários (sem relevantes) | P@10 | R@10 | "
        "NDCG@10 | Cobertura | Diversidade | Novidade |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, result in rows:
        for fold in result["folds"]:
            lines.append(_core_table_row(fold["fold"], name, fold))
        macro = {
            "users_evaluated": "macro",
            "users_without_relevant": "—",
            "precision_at_k": result["mean_precision_at_k"],
            "recall_at_k": result["mean_recall_at_k"],
            "ndcg_at_k": result["mean_ndcg_at_k"],
            "catalog_coverage": result["mean_catalog_coverage"],
            "intra_list_diversity": result["mean_intra_list_diversity"],
            "novelty": result["mean_novelty"],
        }
        lines.append(_core_table_row("Macro", name, macro))
    return "\n".join(lines)


def _core_table_row(fold: Any, name: str, row: dict[str, Any]) -> str:
    users = row["users_evaluated"]
    if users == "macro":
        users_text = "média macro por fold"
    else:
        users_text = f"{users} ({row['users_without_relevant']})"
    values = [
        fold,
        name,
        users_text,
        _format(row["precision_at_k"]),
        _format(row["recall_at_k"]),
        _format(row["ndcg_at_k"]),
        _format(row["catalog_coverage"]),
        _format(row["intra_list_diversity"]),
        _format(row["novelty"]),
    ]
    return "| " + " | ".join(map(str, values)) + " |"


def _variant_results_table(variants: list[dict[str, Any]]) -> str:
    lines = [
        "| Variante | Fold | Precision@10 | Recall@10 | NDCG@10 | "
        "Cobertura | Diversidade | Novidade |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for variant in variants:
        evaluation = variant["evaluation"]
        for fold in evaluation["folds"]:
            lines.append(
                "| "
                + " | ".join(
                    map(
                        str,
                        (
                            variant["variant_name"],
                            fold["fold"],
                            _format(fold["precision_at_k"]),
                            _format(fold["recall_at_k"]),
                            _format(fold["ndcg_at_k"]),
                            _format(fold["catalog_coverage"]),
                            _format(fold["intra_list_diversity"]),
                            _format(fold["novelty"]),
                        ),
                    )
                )
                + " |"
            )
        lines.append(
            "| **"
            + variant["variant_name"]
            + " — macro** | — | "
            + " | ".join(
                _format(evaluation[key])
                for key in (
                    "mean_precision_at_k",
                    "mean_recall_at_k",
                    "mean_ndcg_at_k",
                    "mean_catalog_coverage",
                    "mean_intra_list_diversity",
                    "mean_novelty",
                )
            )
            + " |"
        )
    return "\n".join(lines)


def _segmented_macro_table(segmented: dict[str, Any]) -> str:
    lines = [
        "| Modelo | Histórico | Folds com usuários/relevantes | "
        "Usuários médios por fold | P@10 | R@10 | NDCG@10 | Cobertura |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in segmented["macro_results"]:
        lines.append(
            "| "
            + " | ".join(
                map(
                    str,
                    (
                        row["model_name"],
                        row["history_segment"],
                        f"{row['folds_with_users']}/{row['folds']} / "
                        f"{row['folds_with_relevant_users']}/{row['folds']}",
                        _format(row["mean_users_per_fold"], 2),
                        _format(row["mean_precision_at_k"]),
                        _format(row["mean_recall_at_k"]),
                        _format(row["mean_ndcg_at_k"]),
                        _format(row["mean_catalog_coverage"]),
                    ),
                )
            )
            + " |"
        )
    return "\n".join(lines)


def _segmented_fold_table(segmented: dict[str, Any]) -> str:
    lines = [
        "| Fold | Modelo | Histórico | Usuários (sem relevantes) | P@10 | R@10 | "
        "NDCG@10 | Itens distintos / catálogo | Cobertura |",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in segmented["fold_results"]:
        lines.append(
            "| "
            + " | ".join(
                map(
                    str,
                    (
                        row["fold"],
                        row["model_name"],
                        row["history_segment"],
                        f"{row['users']} ({row['users_without_relevant']})",
                        _format(row["precision_at_k"]),
                        _format(row["recall_at_k"]),
                        _format(row["ndcg_at_k"]),
                        f"{row['distinct_items_recommended']} / {row['catalog_size']}",
                        _format(row["catalog_coverage"]),
                    ),
                )
            )
            + " |"
        )
    return "\n".join(lines)


def _paired_results_table(paired: dict[str, Any]) -> str:
    lines = [
        "| Fold | Métrica | Usuários | Diferença média | IC 95% |",
        "|---:|---|---:|---:|---:|",
    ]
    for row in paired["fold_results"]:
        interval = (
            "—"
            if row["confidence_interval_lower"] is None
            else "["
            + _format(row["confidence_interval_lower"], 5)
            + "; "
            + _format(row["confidence_interval_upper"], 5)
            + "]"
        )
        lines.append(
            f"| {row['fold']} | {row['metric']} | {row['users']} | "
            f"{_format(row['mean_difference'], 5)} | {interval} |"
        )
    for row in paired["macro_results"]:
        counts = ", ".join(
            f"F{fold}: {count}" for fold, count in row["users_by_fold"]
        )
        lines.append(
            f"| **Macro ({row['folds_with_users']}/{row['folds']} folds)** | "
            f"{row['metric']} | {counts} | "
            f"**{_format(row['mean_difference'], 5)}** | — |"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
