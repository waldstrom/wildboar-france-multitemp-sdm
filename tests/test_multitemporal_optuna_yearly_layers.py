import numpy as np
import xarray as xr

from scripts.multitemporal import fuse_yearly_layers


def test_year_prefixed_layers_retained_after_fusing():
    ds_pruned = xr.Dataset(
        {
            "class1": (("y", "x"), np.ones((1, 1))),
            "boundary_mask": (("y", "x"), np.ones((1, 1))),
        },
        coords={"y": [0], "x": [0]},
    )
    vars_used = ["class1"]

    ds_test = xr.Dataset(
        {"2021_class1": (("y", "x"), np.ones((1, 1)))},
        coords={"y": [0], "x": [0]},
    )
    ds_test = fuse_yearly_layers(ds_test)

    missing = [v for v in vars_used if v not in ds_test.data_vars]
    if missing:
        vars_used = [v for v in vars_used if v in ds_test.data_vars]
        select_vars = vars_used
        if "boundary_mask" in ds_pruned.data_vars:
            select_vars = vars_used + ["boundary_mask"]
        ds_pruned = ds_pruned[select_vars]

    assert vars_used == ["class1"]
    assert "boundary_mask" in ds_pruned.data_vars
