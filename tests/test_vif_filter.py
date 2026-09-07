import numpy as np
import xarray as xr

from scripts.multitemporal import _vif_filter


def _cfg():
    return {"multitemporal": {"vif": {"enable": True}}, "experiment": {}}


def test_vif_filter_skips_empty_dataframe():
    ds = xr.Dataset({"var1": (("y", "x"), np.full((2, 2), np.nan))}, coords={"y": [0, 1], "x": [0, 1]})
    out, vars_used = _vif_filter(ds, _cfg())
    assert list(out.data_vars) == ["var1"]
    assert vars_used == ["var1"]


def test_vif_filter_single_variable():
    ds = xr.Dataset({"var1": (("y", "x"), np.ones((2, 2)))}, coords={"y": [0, 1], "x": [0, 1]})
    out, vars_used = _vif_filter(ds, _cfg())
    assert list(out.data_vars) == ["var1"]
    assert vars_used == ["var1"]


def test_vif_filter_excludes_bio08_and_bio09():
    ds = xr.Dataset(
        {
            "bio09_mean_temp_driest_q": (("y", "x"), np.ones((2, 2))),
            "bio08_mean_temp_wettest_q": (("y", "x"), np.ones((2, 2))),
            "bio10": (("y", "x"), np.ones((2, 2))),
        },
        coords={"y": [0, 1], "x": [0, 1]},
    )
    out, vars_used = _vif_filter(ds, _cfg())
    assert "bio09_mean_temp_driest_q" not in vars_used
    assert "bio08_mean_temp_wettest_q" not in vars_used
    assert list(out.data_vars) == ["bio10"]


def test_vif_filter_limits_era5():
    vars = {
        f"2m_temperature_var{i}": (("y", "x"), np.random.rand(2, 2))
        for i in range(8)
    }
    ds = xr.Dataset(vars, coords={"y": [0, 1], "x": [0, 1]})
    cfg = {
        "multitemporal": {"vif": {"enable": True, "n_era5": 6, "sample_fraction": 1.0}},
        "experiment": {"random_seed": 0},
    }
    out, vars_used = _vif_filter(ds, cfg)
    era = [v for v in vars_used if "2m_temperature" in v]
    assert len(era) == 6
    assert len(list(out.data_vars)) == 6


def test_vif_filter_avoids_duplicate_bio_and_era5_names():
    ds = xr.Dataset(
        {
            "bio15_precip_seasonality": (("y", "x"), np.random.rand(2, 2)),
            "2m_temperature_var1": (("y", "x"), np.random.rand(2, 2)),
        },
        coords={"y": [0, 1], "x": [0, 1]},
    )
    cfg = {
        "multitemporal": {
            "vif": {"enable": True, "n_era5": 1, "sample_fraction": 1.0}
        },
        "experiment": {"random_seed": 0},
    }
    out, vars_used = _vif_filter(ds, cfg)
    assert len(vars_used) == len(set(vars_used)) == 2
    assert "bio15_precip_seasonality" in vars_used
    assert "2m_temperature_var1" in vars_used


def test_vif_filter_drops_correlated_bio_era5():
    arr = np.arange(9, dtype=float).reshape(3, 3)
    ds = xr.Dataset(
        {
            "bio10": (("y", "x"), arr),
            "2m_temperature_var1": (("y", "x"), arr * 2),
            "bio11": (("y", "x"), np.random.rand(3, 3)),
        },
        coords={"y": [0, 1, 2], "x": [0, 1, 2]},
    )
    cfg = {
        "multitemporal": {
            "vif": {
                "enable": True,
                "sample_fraction": 1.0,
                "n_bioclim": 3,
                "n_era5": 3,
                "correlation_cutoff": 0.8,
            }
        },
        "experiment": {"random_seed": 0},
    }
    out, vars_used = _vif_filter(ds, cfg)
    assert not ({"bio10", "2m_temperature_var1"} <= set(vars_used))
    assert len({"bio10", "2m_temperature_var1"} & set(vars_used)) == 1
