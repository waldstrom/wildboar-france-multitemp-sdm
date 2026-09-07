import numpy as np
import geopandas as gpd
import xarray as xr
from shapely.geometry import Point, Polygon
from scripts.multitemporal_optuna_regularization import _precompute, YearOffset


def test_precompute_uses_bias_grid(monkeypatch, tmp_path):
    import scripts.multitemporal_optuna_regularization as mor

    ds = xr.Dataset({"var1": (("y", "x"), [[1]])}, coords={"y": [0], "x": [0]})
    stack_path = tmp_path / "stack.zarr"
    ds.to_zarr(stack_path)

    cfg = {
        "multitemporal": {
            "background_per_year": 1,
            "validation_fraction": 0.0,
        },
        "background": {"use_bias_grid": True},
        "experiment": {},
        "presence_data": {"thinning": {"percentage": 1.0}},
    }
    season = "summer"
    mapping = [YearOffset(2020, 0.0, 0.0)]
    pres_raw = {
        2020: gpd.GeoDataFrame({"geometry": [Point(0, 0)]}, crs="EPSG:2154")
    }
    env_cache = {2020: np.array([[0]])}
    pres_test = gpd.GeoDataFrame({"geometry": [Point(0, 0)]}, crs="EPSG:2154")
    test_year = 2021
    fr_poly = gpd.GeoSeries(
        [Polygon([(0, 0), (2, 0), (2, 2), (0, 2)])], crs="EPSG:2154"
    )
    bias = xr.DataArray(
        [[1, 0], [0, 0]],
        coords={"y": [1.5, 0.5], "x": [0.5, 1.5]},
        dims=("y", "x"),
    )

    monkeypatch.setattr(mor, "_load_stack", lambda cfg, season, year: (ds, None))
    monkeypatch.setattr(mor, "fuse_yearly_layers", lambda d: d)
    monkeypatch.setattr(mor, "_vif_filter", lambda d, cfg: (d, list(d.data_vars)))
    monkeypatch.setattr(
        mor, "extract_from_stack", lambda stack, gdf, vars: np.zeros((len(gdf), len(vars)))
    )

    calls = {"bias": 0}

    def fake_bias_sampler(b, n, strength=1.0):
        calls["bias"] += 1
        return gpd.GeoDataFrame(
            {"x": [0.5] * n, "y": [1.5] * n},
            geometry=gpd.points_from_xy([0.5] * n, [1.5] * n),
            crs="EPSG:2154",
        )

    def fake_poly_sampler(poly, n):
        raise AssertionError("Polygon sampler should not be used")

    monkeypatch.setattr(mor, "_sample_background_bias", fake_bias_sampler)
    monkeypatch.setattr(mor, "_sample_background_polygon", fake_poly_sampler)

    data = mor._precompute(
        cfg,
        season,
        mapping,
        stack_path,
        pres_raw,
        env_cache,
        pres_test,
        test_year,
        fr_poly,
        bias,
        bias,
    )

    assert calls["bias"] == 2
    assert len(data["bg_train"]) == 1
