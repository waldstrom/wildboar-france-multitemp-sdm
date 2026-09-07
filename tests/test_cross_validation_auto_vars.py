import geopandas as gpd
from shapely.geometry import Point
import numpy as np
import pandas as pd
import xarray as xr
from unittest.mock import patch
from scripts.cross_validation import run_cv


def test_run_cv_uses_stack_vars_when_missing(tmp_path):
    cfg = {
        "evaluation": {
            "cross_validation": {
                "method": "random_kfold",
                "params": {"k_folds": 2},
            }
        },
        "experiment": {"random_seed": 0, "stats_dir": str(tmp_path / "stats")},
        "outputs": {
            "root": str(tmp_path / "out"),
            "logs_dir": str(tmp_path / "logs"),
            "model_dir": str(tmp_path / "models"),
            "maps_dir": str(tmp_path / "maps"),
            "figures_dir": str(tmp_path / "figs"),
        },
    }

    pres_gdf = gpd.GeoDataFrame(
        geometry=[Point(0, 0), Point(1, 0)], crs="EPSG:2154"
    )
    bg_gdf = gpd.GeoDataFrame(
        geometry=[Point(0, 1), Point(1, 1)], crs="EPSG:2154"
    )

    ds = xr.Dataset(
        {"var1": (("y", "x"), np.zeros((2, 2), dtype=float))},
        coords={"x": [0, 1], "y": [0, 1]},
    )
    stack = tmp_path / "stack.zarr"
    ds.to_zarr(stack)

    class DummyModel:
        def predict(self, X):
            return np.zeros(len(X))

    with patch("scripts.cross_validation.train_model", return_value=(DummyModel(), None, ["var1"])) as mock_train:
        run_cv(
            cfg,
            pres_gdf=pres_gdf,
            bg_gdf=bg_gdf,
            stack_path=stack,
            vars_used=[],
        )
        assert mock_train.call_args[0][2] == ["var1"]
