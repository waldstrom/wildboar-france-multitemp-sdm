import numpy as np
import xarray as xr

from scripts.multitemporal import _vif_filter


def _cfg():
    # Disable VIF filtering to keep variables as-is
    return {"multitemporal": {"vif": {"enable": False}}, "experiment": {}}


def test_boundary_mask_preserved_after_dropping_missing_predictors():
    # base stack with two predictors and boundary mask
    ds_base = xr.Dataset(
        {
            "var1": (("y", "x"), np.ones((2, 2))),
            "var2": (("y", "x"), np.ones((2, 2))),
            "boundary_mask": (("y", "x"), np.ones((2, 2))),
        },
        coords={"y": [0, 1], "x": [0, 1]},
    )

    ds_pruned, vars_used = _vif_filter(ds_base, _cfg())

    # test stack missing "var2"
    ds_test = xr.Dataset(
        {
            "var1": (("y", "x"), np.ones((2, 2))),
            "boundary_mask": (("y", "x"), np.ones((2, 2))),
        },
        coords={"y": [0, 1], "x": [0, 1]},
    )

    missing = [v for v in vars_used if v not in ds_test.data_vars]
    if missing:
        vars_used = [v for v in vars_used if v in ds_test.data_vars]
        select_vars = vars_used
        if "boundary_mask" in ds_pruned.data_vars:
            select_vars = vars_used + ["boundary_mask"]
        ds_pruned = ds_pruned[select_vars]

    assert "boundary_mask" in ds_pruned.data_vars
    assert vars_used == ["var1"]
