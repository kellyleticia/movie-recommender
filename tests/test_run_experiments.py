import json
from pathlib import Path
from typing import Any

import pytest

import app.run_experiments as runner


def minimal_payload() -> dict[str, Any]:
    fold = {
        "fold": 1,
        "users_evaluated": 2,
        "users_without_relevant": 1,
        "precision_at_k": 0.25,
        "recall_at_k": None,
        "ndcg_at_k": None,
        "catalog_coverage": 0.5,
        "intra_list_diversity": None,
        "novelty": None,
    }
    macro = {
        "mean_precision_at_k": 0.25,
        "mean_recall_at_k": None,
        "mean_ndcg_at_k": None,
        "mean_catalog_coverage": 0.5,
        "mean_intra_list_diversity": None,
        "mean_novelty": None,
    }
    segment_fold = {
        "fold": 1,
        "model_name": "item_based_centered",
        "history_segment": "1-5",
        "users": 2,
        "users_without_relevant": 1,
        "precision_at_k": 0.25,
        "recall_at_k": None,
        "ndcg_at_k": None,
        "distinct_items_recommended": 1,
        "catalog_size": 2,
        "catalog_coverage": 0.5,
    }
    segment_macro = {
        "model_name": "item_based_centered",
        "history_segment": "1-5",
        "folds": 5,
        "folds_with_users": 1,
        "folds_with_relevant_users": 0,
        "mean_users_per_fold": 0.4,
        "mean_precision_at_k": 0.25,
        "mean_recall_at_k": None,
        "mean_ndcg_at_k": None,
        "mean_catalog_coverage": 0.5,
    }
    paired_fold = {
        "fold": 1,
        "metric": "precision_at_10",
        "mean_difference": 0.0,
        "confidence_interval_lower": 0.0,
        "confidence_interval_upper": 0.0,
        "users": 2,
    }
    paired_macro = {
        "metric": "precision_at_10",
        "folds": 5,
        "folds_with_users": 1,
        "mean_difference": 0.0,
        "users_by_fold": [[1, 2]],
    }
    return {
        "configuration": runner.experiment_configuration(),
        "dataset": {
            "catalog_movies": 2,
            "genre_count": 1,
            "user_count": 2,
            "folds": [
                {
                    "fold": 1,
                    "training_ratings": 4,
                    "test_ratings": 2,
                    "test_users": 2,
                }
            ],
        },
        "evaluations": {
            "item_based": {
                **macro,
                "k": 10,
                "relevance_threshold": 4,
                "folds": [fold],
            },
            "popularity": {
                **macro,
                "k": 10,
                "relevance_threshold": 4,
                "folds": [fold],
            },
            "variants": [
                {
                    "variant_name": "centered_all_neighbors",
                    "neighbor_limit": None,
                    "evaluation": {
                        **macro,
                        "folds": [fold],
                    },
                }
            ],
            "segmented": {
                "relevance_threshold": 4,
                "k": 10,
                "fold_results": [segment_fold],
                "macro_results": [segment_macro],
                "users_without_training_history_by_fold": [[1, 0]],
            },
            "paired_bootstrap": {
                "k": 10,
                "relevance_threshold": 4,
                "bootstrap_iterations": 10_000,
                "seed": 42,
                "fold_results": [paired_fold],
                "macro_results": [paired_macro],
            },
        },
    }


def test_experiment_configuration_preserves_established_settings() -> None:
    configuration = runner.experiment_configuration()

    assert configuration["official_folds"] == [1, 2, 3, 4, 5]
    assert configuration["k"] == 10
    assert configuration["relevance_threshold"] == 4
    assert configuration["neighbor_limits"] == [20, 50, 100]
    assert configuration["history_segments"] == ["1-5", "6-10", "11-20", "21+"]
    assert configuration["paired_bootstrap"]["iterations"] == 10_000
    assert configuration["paired_bootstrap"]["seed"] == 42


def test_cli_entry_point_forwards_data_and_report_directories(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    calls: list[tuple[Path, Path]] = []

    def fake_run_experiment_suite(*, data_dir: Path, reports_dir: Path):
        calls.append((data_dir, reports_dir))
        return (
            {},
            reports_dir / "experiment_results.json",
            reports_dir / "experiment_report.md",
        )

    monkeypatch.setattr(
        runner, "run_experiment_suite", fake_run_experiment_suite
    )
    data_dir = tmp_path / "dataset"
    reports_dir = tmp_path / "reports"

    result = runner.main(
        ["--data-dir", str(data_dir), "--reports-dir", str(reports_dir)]
    )

    assert result == 0
    assert calls == [(data_dir, reports_dir)]
    output = capsys.readouterr().out
    assert "experiment_results.json" in output
    assert "experiment_report.md" in output


def test_missing_dataset_message_contains_expected_instructions(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    missing_dir = tmp_path / "missing" / "ml-100k"

    result = runner.main(["--data-dir", str(missing_dir)])

    assert result == 2
    error = capsys.readouterr().err
    assert "MovieLens 100K files are missing" in error
    assert "ml-100k.zip" in error
    assert str(missing_dir / "u.item") in error


def test_reports_generate_json_and_markdown_and_are_reproducible(
    tmp_path: Path,
) -> None:
    payload = minimal_payload()
    json_path = tmp_path / "reports" / "experiment_results.json"
    markdown_path = tmp_path / "reports" / "experiment_report.md"

    runner.write_reports(payload, json_path=json_path, markdown_path=markdown_path)
    first_json = json_path.read_text(encoding="utf-8")
    first_markdown = markdown_path.read_text(encoding="utf-8")
    runner.write_reports(payload, json_path=json_path, markdown_path=markdown_path)

    assert json_path.read_text(encoding="utf-8") == first_json
    assert markdown_path.read_text(encoding="utf-8") == first_markdown
    decoded = json.loads(first_json)
    assert decoded["evaluations"]["item_based"]["mean_recall_at_k"] is None
    assert decoded["evaluations"]["paired_bootstrap"]["fold_results"][0]["users"] == 2
    assert "null" in first_json
    assert "—" in first_markdown
    assert "MovieLens 100K" in first_markdown
    assert "Bootstrap" in first_markdown


def test_renderer_preserves_zero_and_undefined_as_distinct_values() -> None:
    payload = minimal_payload()
    report = runner.render_markdown_report(runner._to_json_safe(payload))

    assert "0.2500" in report
    assert "| 1 |" in report
    assert "—" in report
    assert "| null |" not in report
