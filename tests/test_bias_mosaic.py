import numpy as np
import rasterio
from rasterio.transform import from_origin
import xarray as xr

from scripts.multitemporal import _merge_bias_maps, _sample_background_bias, YearOffset


def _write_bias(path, value):
    data = np.array([[value]], dtype="float32")
    transform = from_origin(0, 1, 1, 1)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=1,
        width=1,
        count=1,
        dtype="float32",
        crs="EPSG:2154",
        transform=transform,
    ) as dst:
        dst.write(data, 1)


def test_merge_bias_maps_mosaics(tmp_path):
    f1 = tmp_path / "2018.tif"
    f2 = tmp_path / "2019.tif"
    _write_bias(f1, 1)
    _write_bias(f2, 2)
    cfg = {"bias_grid": {"enable": True, "pattern": str(tmp_path / "{year}.tif")}}
    mapping = [YearOffset(2018, 0.0, 0.0), YearOffset(2019, 1.0, 0.0)]
    merged = _merge_bias_maps(cfg, "summer", mapping)
    assert merged is not None
    merged = merged.sortby("x")
    assert merged.shape == (1, 2)
    assert merged.values[0, 0] == 1
    assert merged.values[0, 1] == 2


def test_sample_background_bias(tmp_path):
    da = xr.DataArray(
        [[1, 0], [0, 1]],
        coords={"y": [1.5, 0.5], "x": [0.5, 1.5]},
        dims=("y", "x"),
        name="bias",
    )
    gdf = _sample_background_bias(da, 2)
    assert len(gdf) == 2
    assert gdf["x"].between(0.5, 1.5).all()
    assert gdf["y"].between(0.5, 1.5).all()


def test_sample_background_bias_resizes_mask():
    bias = xr.DataArray(
        np.ones((4, 4), dtype="float32"),
        coords={"y": [3.5, 2.5, 1.5, 0.5], "x": [0.5, 1.5, 2.5, 3.5]},
        dims=("y", "x"),
        name="bias",
    )
    mask = np.array([[1, 0], [0, 1]], dtype=float)
    gdf = _sample_background_bias(bias, 5, mask=mask)
    assert len(gdf) == 5
    valid = ((gdf["x"] < 2) & (gdf["y"] > 2)) | (
        (gdf["x"] > 2) & (gdf["y"] < 2)
    )
    assert valid.all()
