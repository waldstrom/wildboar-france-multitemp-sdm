import json

import numpy as np
import xarray as xr

from scripts.monotemporal import YearOffset, _merge_stacks


def test_merge_stacks_keeps_partially_available_predictors(tmp_path):
    coords = {"y": [0, 1], "x": [0, 1]}
    ds_master = xr.Dataset(
        {
            "2018_bar": xr.DataArray(
                np.full((2, 2), 1.0), coords=coords, dims=("y", "x")
            ),
            "2019_bar": xr.DataArray(
                np.full((2, 2), 5.0), coords=coords, dims=("y", "x")
            ),
            "2018_foo": xr.DataArray(
                np.full((2, 2), 2.0), coords=coords, dims=("y", "x")
            ),
            "2019_foo": xr.DataArray(
                np.full((2, 2), 6.0), coords=coords, dims=("y", "x")
            ),
            "2018_baz": xr.DataArray(
                np.full((2, 2), 3.0), coords=coords, dims=("y", "x")
            ),
            "2019_baz": xr.DataArray(
                np.full((2, 2), 4.0), coords=coords, dims=("y", "x")
            ),
        },
        coords=coords,
    )
    stack_dir = tmp_path / "Summer_master.zarr"
    ds_master.to_zarr(stack_dir, mode="w", consolidated=False)

    feature_cfg = tmp_path / "featureconfig.yaml"
    feature_cfg.write_text(
        "variables:\n"
        "  2018_baz: drop\n"
        "  2019_baz: drop\n"
        "  2019_bar: drop\n"
    )

    cfg = {
        "multitemporal": {
            "stack_base_dir": str(tmp_path),
            "n_workers": 1,
            "vif": {"feature_config": str(feature_cfg)},
        },
        "predictors": {"layers": {}, "nodata_value": np.nan},
        "maxent": {"geotiff_compression": "LZW"},
    }

    out_dir = tmp_path / "out"
    out_dir.mkdir()
    mapping = [YearOffset(2018), YearOffset(2019)]

    stack_path, vars_used = _merge_stacks(
        cfg,
        "Summer",
        mapping,
        out_dir,
        apply_vif=False,
        stack_name="test_stack",
    )

    zarr_path = stack_path.with_suffix(".zarr")
    meta_path = zarr_path.with_suffix(".metadata.json")
    assert zarr_path.exists()
    assert meta_path.exists()
    assert {"bar", "foo"}.issubset(set(vars_used))

    ds_avg = xr.open_zarr(zarr_path)
    try:
        assert "bar" in ds_avg.data_vars
        assert "foo" in ds_avg.data_vars
        assert "baz" not in ds_avg.data_vars
        np.testing.assert_allclose(ds_avg["bar"].values, 1.0)
        np.testing.assert_allclose(ds_avg["foo"].values, 4.0)
    finally:
        ds_avg.close()

    summary = json.loads(meta_path.read_text())
    bar_info = summary["variables"]["bar"]
    assert bar_info["available_years"] == [2018]
    assert bar_info["missing"] == {"2019": "feature_config_drop"}
    assert "excluded" not in bar_info

    baz_info = summary["variables"]["baz"]
    assert baz_info["available_years"] == []
    assert baz_info["missing"] == {
        "2018": "feature_config_drop",
        "2019": "feature_config_drop",
    }
    assert baz_info["excluded"] is True
