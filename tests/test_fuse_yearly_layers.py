import numpy as np
import xarray as xr

from scripts.multitemporal import fuse_yearly_layers


def test_fuse_yearly_layers_merges_year_prefixed_variables():
    da2018 = xr.DataArray([[1]], coords={"y": [0], "x": [0]}, dims=("y", "x"))
    da2019 = xr.DataArray([[2]], coords={"y": [0], "x": [1]}, dims=("y", "x"))
    static = xr.DataArray([[3, 3]], coords={"y": [0], "x": [0, 1]}, dims=("y", "x"))
    ds = xr.Dataset({
        "2018_conifers": da2018,
        "2019_conifers": da2019,
        "elevation": static,
    })

    fused = fuse_yearly_layers(ds)

    assert set(fused.data_vars) == {"conifers", "elevation"}
    assert fused["conifers"].sel(x=0).values.item() == 1
    assert fused["conifers"].sel(x=1).values.item() == 2
    assert fused["elevation"].equals(static)


def test_fuse_yearly_layers_merges_year_suffixed_variables():
    da2018 = xr.DataArray([[1]], coords={"y": [0], "x": [0]}, dims=("y", "x"))
    da2019 = xr.DataArray([[2]], coords={"y": [0], "x": [1]}, dims=("y", "x"))
    ds = xr.Dataset({
        "hunting_bag_2018": da2018,
        "hunting_bag_2019": da2019,
    })

    fused = fuse_yearly_layers(ds)

    assert set(fused.data_vars) == {"hunting_bag"}
    assert fused["hunting_bag"].sel(x=0).values.item() == 1
    assert fused["hunting_bag"].sel(x=1).values.item() == 2


def test_fuse_yearly_layers_merges_year_in_middle():
    da2018 = xr.DataArray([[1]], coords={"y": [0], "x": [0]}, dims=("y", "x"))
    da2019 = xr.DataArray([[2]], coords={"y": [0], "x": [1]}, dims=("y", "x"))
    ds = xr.Dataset(
        {
            "tmax_2018_summer": da2018,
            "tmax_2019_summer": da2019,
        }
    )

    fused = fuse_yearly_layers(ds)

    assert set(fused.data_vars) == {"tmax_summer"}
    assert fused["tmax_summer"].sel(x=0).values.item() == 1
    assert fused["tmax_summer"].sel(x=1).values.item() == 2
