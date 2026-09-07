import importlib.util
from pathlib import Path

import pandas as pd
import pytest


SCRIPT = Path(__file__).parents[1] / "review-finalize-existing-run.py"
SPEC = importlib.util.spec_from_file_location("review_finalize_existing_run", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_load_fold_predictors_prefers_engine_specific_selection(tmp_path):
    fold_dir = tmp_path / "output" / "Winter" / "test_2022"
    engine_dir = fold_dir / "elapid"
    stack_dir = tmp_path / "input-data-test-2022" / "Winter_stack"
    engine_dir.mkdir(parents=True)
    stack_dir.mkdir(parents=True)
    (engine_dir / "selected_predictors.txt").write_text(
        "bio1\nhunting_bag\n", encoding="utf-8"
    )
    (stack_dir / "selected_predictors.txt").write_text(
        "bio1\nbio2\nhunting_bag\n", encoding="utf-8"
    )

    assert MODULE.load_fold_predictors(fold_dir, "elapid", stack_dir) == [
        "bio1",
        "hunting_bag",
    ]


def test_load_fold_predictors_falls_back_to_materialized_stack(tmp_path):
    fold_dir = tmp_path / "output" / "Winter" / "test_2022"
    stack_dir = tmp_path / "input-data-test-2022" / "Winter_stack"
    stack_dir.mkdir(parents=True)
    (stack_dir / "selected_predictors.txt").write_text(
        "bio1,hunting_bag\n", encoding="utf-8"
    )

    assert MODULE.load_fold_predictors(fold_dir, "elapid", stack_dir) == [
        "bio1",
        "hunting_bag",
    ]


@pytest.mark.parametrize("owner", ["model", "transformer"])
def test_model_predictors_reads_elapid_labels(owner):
    class SavedModel:
        pass

    model = SavedModel()
    target = model
    if owner == "transformer":
        model.transformer = SavedModel()
        target = model.transformer
    target.labels_ = ["bio1", "hunting_bag"]

    assert MODULE.predictors_for_model(
        model, ["bio1", "bio2", "hunting_bag"]
    ) == ["bio1", "hunting_bag"]


def test_model_predictors_ignores_method_named_labels():
    class SavedModel:
        def labels(self):
            return ["not", "metadata"]

    assert MODULE.model_predictors(SavedModel()) == []


def test_predictors_for_model_ignores_elapid_expanded_feature_class_labels():
    class SavedModel:
        pass

    model = SavedModel()
    model.transformer = SavedModel()
    model.transformer.labels_ = [
        "linear",
        "linear",
        "quadratic",
        "product",
        "hinge",
        "threshold",
    ]
    selected = ["bio1", "bio2", "hunting_bag"]

    assert MODULE.predictors_for_model(model, selected) is selected


def test_predictors_for_model_still_rejects_unknown_missing_labels():
    class SavedModel:
        labels_ = ["bio1", "missing_predictor", "missing_predictor"]

    with pytest.raises(RuntimeError, match=r": missing_predictor$"):
        MODULE.predictors_for_model(SavedModel(), ["bio1"])


def test_allfold_hunting_importance_runs_only_maxent(monkeypatch, tmp_path):
    calls = []

    def fake_fold(record, *, year, engine, **kwargs):
        calls.append((record["id"], engine, year))
        return {
            "scenario": record["id"],
            "scenario_role": "corrected",
            "dataset": "allpoints",
            "temporal": "monotemporal",
            "engine": engine,
            "test_year": year,
            "variable": "hunting_bag",
            "n_repeats": kwargs["n_repeats"],
            "importance_mean_auc_drop": 0.1,
        }

    monkeypatch.setattr(MODULE, "compute_fold_hunting_importance", fake_fold)
    registry = [{
        "id": "scenario",
        "role": "corrected",
        "years": [2022, 2023],
        "engines": ["elapid", "gbm"],
    }]

    fold_df, _ = MODULE.compute_allfold_hunting_importance(
        registry,
        tmp_path,
        n_repeats=2,
        random_state=42,
        predict_threads=1,
    )

    assert calls == [
        ("scenario", "elapid", 2022),
        ("scenario", "elapid", 2023),
    ]
    assert set(fold_df["engine"]) == {"elapid"}


def test_allfold_hunting_importance_discards_gbm_checkpoint_rows(
    monkeypatch, tmp_path
):
    checkpoint = tmp_path / "hunting_permutation_importance_all_folds.partial.csv"
    pd.DataFrame([{
        "scenario": "scenario",
        "scenario_role": "corrected",
        "dataset": "allpoints",
        "temporal": "monotemporal",
        "engine": "gbm",
        "test_year": 2022,
        "variable": "hunting_bag",
        "n_repeats": 99,
        "importance_mean_auc_drop": 0.2,
    }]).to_csv(checkpoint, index=False)

    def fake_fold(record, *, year, engine, **kwargs):
        return {
            "scenario": record["id"],
            "scenario_role": "corrected",
            "dataset": "allpoints",
            "temporal": "monotemporal",
            "engine": engine,
            "test_year": year,
            "variable": "hunting_bag",
            "n_repeats": kwargs["n_repeats"],
            "importance_mean_auc_drop": 0.1,
        }

    monkeypatch.setattr(MODULE, "compute_fold_hunting_importance", fake_fold)
    registry = [{
        "id": "scenario",
        "role": "corrected",
        "years": [2022],
        "engines": ["elapid", "gbm"],
    }]

    fold_df, _ = MODULE.compute_allfold_hunting_importance(
        registry,
        tmp_path,
        n_repeats=2,
        random_state=42,
        predict_threads=1,
    )

    assert fold_df[["engine", "n_repeats"]].to_dict("records") == [
        {"engine": "elapid", "n_repeats": 2}
    ]
