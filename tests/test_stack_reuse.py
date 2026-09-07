import pytest
import xarray as xr
from scripts import multitemporal


def test_load_stack_missing_raises(tmp_path):
    cfg = {
        "multitemporal": {"stack_base_dir": str(tmp_path)},
        "predictors": {"layers": {}},
    }
    with pytest.raises(FileNotFoundError):
        multitemporal._load_stack(cfg, "Winter", 2024, rebuild=False, validate=False)


def test_load_stack_filters_correct_year(tmp_path):
    ds = xr.Dataset(
        {
            "tmax_2018_summer": xr.DataArray([[1]], dims=("y", "x")),
            "tmax_2019_summer": xr.DataArray([[2]], dims=("y", "x")),
            "elevation": xr.DataArray([[3]], dims=("y", "x")),
        }
    )
    path = tmp_path / "Summer_master.zarr"
    ds.to_zarr(path, mode="w", consolidated=False)
    cfg = {
        "multitemporal": {"stack_base_dir": str(tmp_path), "years": [2018, 2019]},
        "predictors": {"layers": {}},
    }
    ds_year, _ = multitemporal._load_stack(cfg, "Summer", 2018, rebuild=False, validate=False)
    assert set(ds_year.data_vars) == {"tmax_2018_summer", "elevation"}
    ds_year.close()
