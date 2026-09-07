from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_origin

from scripts import mess_evaluation


def _write_raster(path: Path, data: np.ndarray, *, dtype: str = "float32", nodata: float | int | None = np.nan) -> None:
    transform = from_origin(0, data.shape[0], 1, 1)
    profile = {
        "driver": "GTiff",
        "height": data.shape[0],
        "width": data.shape[1],
        "count": 1,
        "dtype": dtype,
        "crs": "EPSG:2154",
        "transform": transform,
        "nodata": nodata,
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data.astype(dtype), 1)


def test_mess_zero_variance_predictor() -> None:
    stats = mess_evaluation.ReferenceStats("const", np.array([5.0, 5.0, 5.0], dtype="float64"))
    values = np.array([5.0, 4.0, 6.0], dtype="float32")
    scores, valid, _, _ = mess_evaluation._mess_scores(values, stats)
    assert valid.tolist() == [True, True, True]
    assert np.isclose(scores[0], 100.0)
    assert scores[1] < 0
    assert scores[2] < 0


def test_mess_computation(tmp_path: Path) -> None:
    exp_root = tmp_path / "experiment"
    test_stack = exp_root / "output" / "winter" / "test_2020" / "test_stack"
    test_stack.mkdir(parents=True)

    ref_dir = exp_root / "input-data-test-2020"
    ref_dir.mkdir(parents=True)
    stack_dir = ref_dir / "winter_stack"
    stack_dir.mkdir()
    (stack_dir / "selected_predictors.txt").write_text("var1\nvar2\n", encoding="utf-8")

    train_df = pd.DataFrame({"var1": [0, 5, 10], "var2": [0, 2, 4]})
    bg_df = pd.DataFrame({"var1": [0, 5, 10], "var2": [0, 2, 4]})
    train_df.to_csv(ref_dir / "winter_train_features.csv", index=False)
    bg_df.to_csv(ref_dir / "winter_background_features.csv", index=False)

    var1 = np.array([[0, 5], [10, 15]], dtype="float32")
    var2 = np.array([[0, 1], [2, 3]], dtype="float32")
    _write_raster(test_stack / "var1.tif", var1)
    _write_raster(test_stack / "var2.tif", var2)

    session_root = tmp_path / "mess_session"
    outputs = mess_evaluation.run_mess(
        exp_root / "output" / "winter",
        years=[2020],
        run_type="single",
        output_root=session_root,
    )
    assert outputs.output_root == session_root
    season_outputs = outputs.results[("winter", 2020)]
    mess_path = season_outputs["mess_raster"]
    with rasterio.open(mess_path) as src:
        mess_data = src.read(1)

    expected = np.array([[66.6667, 66.6667], [0.0, -50.0]], dtype="float32")
    np.testing.assert_allclose(mess_data, expected, rtol=1e-3, atol=1e-3)

    summary = pd.read_csv(season_outputs["overall_summary"])
    assert summary.loc[0, "valid_cells"] == 4
    assert summary.loc[0, "share_negative"] == 0.25
    assert summary.loc[0, "min_mess"] == -50.0
    assert "mean_abs_corr_change" in summary.columns
    assert "fro_corr_change" in summary.columns
    assert "nt2_only_share_of_classified" in summary.columns
    assert summary.loc[0, "classified_cells"] == 4

    limiting = pd.read_csv(season_outputs["limiting_summary"])
    assert limiting.loc[limiting["variable"] == "var1", "limiting_cells"].iloc[0] == 4
    assert limiting.loc[limiting["variable"] == "var2", "limiting_cells"].iloc[0] == 0

    metadata = json.loads((season_outputs["metadata"]).read_text())
    assert metadata["predictors"]["0"] == "var1"
    assert metadata["predictors"]["1"] == "var2"
    assert metadata["run_type"] == "single"
    assert "nt2_threshold_md2" in metadata
    assert metadata["nt2_quantile"] == mess_evaluation.NT2_QUANTILE
    assert "novelty_class_counts" in metadata
    assert "visualizations" in metadata

    qml_path = season_outputs["limiting_qml"]
    assert "paletteEntry" in qml_path.read_text()
    assert season_outputs["correlation_change"].exists()
    assert season_outputs["nt2_raster"].exists()
    assert season_outputs["exdet_class_raster"].exists()
    assert season_outputs["mess_jpeg"].exists()
    assert season_outputs["limiting_jpeg"].exists()
    assert season_outputs["nt2_jpeg"].exists()
    assert season_outputs["exdet_class_jpeg"].exists()
    assert season_outputs["mess_histogram_jpeg"].exists()
    assert season_outputs["nt2_histogram_jpeg"].exists()
    assert season_outputs["correlation_heatmap_jpeg"].exists()
    assert set(
        metadata["visualizations"].keys()
    ).issuperset(
        {
            "mess_jpeg",
            "limiting_jpeg",
            "nt2_jpeg",
            "exdet_class_jpeg",
            "mess_histogram_jpeg",
            "nt2_histogram_jpeg",
            "correlation_heatmap_jpeg",
        }
    )


def test_nt2_classification_and_boundary_mask(tmp_path: Path) -> None:
    season_dir = tmp_path / "winter"
    season_dir.mkdir(parents=True)
    test_dir = season_dir / "test_2025"
    stack_dir = test_dir / "test_stack"
    stack_dir.mkdir(parents=True)

    x = np.array([[0.0, 0.9], [0.2, 0.0]], dtype="float32")
    y = np.array([[0.0, 0.9], [0.0, 0.2]], dtype="float32")
    mask = np.array([[1, 1], [1, 0]], dtype="uint8")
    _write_raster(stack_dir / "x.tif", x)
    _write_raster(stack_dir / "y.tif", y)
    _write_raster(stack_dir / "boundary_mask.tif", mask, dtype="uint8", nodata=0)

    stats = {
        "x": mess_evaluation.ReferenceStats("x", np.array([-1.0, 0.0, 1.0], dtype="float64")),
        "y": mess_evaluation.ReferenceStats("y", np.array([-1.0, 0.0, 1.0], dtype="float64")),
    }
    training = mess_evaluation.TrainingContext(
        predictors=["x", "y"],
        reference=stats,
        correlation=np.eye(2, dtype="float64"),
        mean=np.array([0.0, 0.0], dtype="float64"),
        cov=np.eye(2, dtype="float64"),
        inv_cov=np.eye(2, dtype="float64"),
        md2_threshold=1.0,
        nt2_quantile=mess_evaluation.NT2_QUANTILE,
    )

    out_dir = tmp_path / "session" / "synthetic" / "winter" / "test_2025"
    outputs = mess_evaluation.process_season_year(
        season_dir=season_dir,
        test_dir=test_dir,
        year=2025,
        predictors=["x", "y"],
        training=training,
        out_dir=out_dir,
        run_label="synthetic",
    )

    with rasterio.open(outputs["nt2_raster"]) as src:
        nt2 = src.read(1)
    with rasterio.open(outputs["exdet_class_raster"]) as src:
        cls = src.read(1)

    assert nt2[0, 1] > 1.0
    assert cls[0, 1] == 2
    assert cls[1, 1] == 255
    assert np.isnan(nt2[1, 1])

    summary = pd.read_csv(outputs["overall_summary"])
    assert summary.loc[0, "nt2_any_share_of_classified"] > 0

    metadata = json.loads(outputs["metadata"].read_text())
    assert metadata["novelty_class_counts"]["class_2_nt2_only"] >= 1
    assert metadata["boundary_mask"].endswith("boundary_mask.tif")


def test_compare_mono_multi(tmp_path: Path) -> None:
    exp_root = tmp_path / "experiment"
    mono_dir = exp_root / "output" / "mono" / "winter"
    multi_dir = exp_root / "output" / "multi" / "winter"
    for path in [mono_dir, multi_dir]:
        (path / "test_2020" / "test_stack").mkdir(parents=True)

    ref_dir = exp_root / "input-data-test-2020"
    ref_dir.mkdir(parents=True)
    stack_dir = ref_dir / "winter_stack"
    stack_dir.mkdir()
    (stack_dir / "selected_predictors.txt").write_text("var1\nvar2\n", encoding="utf-8")

    train_df = pd.DataFrame({"var1": [0, 5, 10], "var2": [0, 2, 4]})
    bg_df = pd.DataFrame({"var1": [0, 5, 10], "var2": [0, 2, 4]})
    train_df.to_csv(ref_dir / "winter_train_features.csv", index=False)
    bg_df.to_csv(ref_dir / "winter_background_features.csv", index=False)

    mono_var1 = np.array([[0, 5], [10, 15]], dtype="float32")
    mono_var2 = np.array([[0, 1], [2, 3]], dtype="float32")
    multi_var1 = np.array([[0, 2], [4, 6]], dtype="float32")
    multi_var2 = np.array([[0, 0.5], [1, 1.5]], dtype="float32")

    _write_raster(mono_dir / "test_2020" / "test_stack" / "var1.tif", mono_var1)
    _write_raster(mono_dir / "test_2020" / "test_stack" / "var2.tif", mono_var2)
    _write_raster(multi_dir / "test_2020" / "test_stack" / "var1.tif", multi_var1)
    _write_raster(multi_dir / "test_2020" / "test_stack" / "var2.tif", multi_var2)

    session_root = tmp_path / "comparison_session"
    mono_outputs = mess_evaluation.run_mess(
        mono_dir, years=[2020], run_type="monotemporal", output_root=session_root
    )
    multi_outputs = mess_evaluation.run_mess(
        multi_dir, years=[2020], run_type="multitemporal", output_root=session_root
    )

    comparison = mess_evaluation.compare_runs(
        mono_outputs, multi_outputs, output_root=session_root
    )
    assert ("winter", 2020) in comparison
    comparison_path = comparison[("winter", 2020)]
    df = pd.read_csv(comparison_path)
    assert {"monotemporal", "multitemporal"}.issubset(set(df["run_type"]))
    assert any(df["run_type"].str.contains("multitemporal_minus_monotemporal", na=False))
    overview = pd.read_csv(session_root / "comparisons" / "mess_comparison_overview.csv")
    assert not overview.empty

