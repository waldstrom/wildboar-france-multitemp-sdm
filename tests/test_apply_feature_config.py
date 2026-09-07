import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

pytest.importorskip("pandas")
xr = pytest.importorskip("xarray")

from scripts.multitemporal import _apply_feature_config, _fill_missing_satellite_layers


def _make_feature_cfg(path: Path):
    path.write_text("variables:\n  drop_me: drop\n  keep_me: keep\n")
    return path


def test_apply_feature_config_respects_ignore_flag():
    ds = xr.Dataset({"drop_me": ("x", [1]), "keep_me": ("x", [2])})
    with tempfile.TemporaryDirectory() as tmpdir:
        fc = _make_feature_cfg(Path(tmpdir) / "fc.yaml")
        cfg = {"multitemporal": {"ignore_featureconfig_drop": True, "vif": {"feature_config": str(fc)}}}
        out = _apply_feature_config(ds, cfg)
        assert set(out.data_vars) == {"drop_me", "keep_me"}


def test_apply_feature_config_drops_when_enabled():
    ds = xr.Dataset({"drop_me": ("x", [1]), "keep_me": ("x", [2])})
    with tempfile.TemporaryDirectory() as tmpdir:
        fc = _make_feature_cfg(Path(tmpdir) / "fc.yaml")
        cfg = {"multitemporal": {"ignore_featureconfig_drop": False, "vif": {"feature_config": str(fc)}}}
        out = _apply_feature_config(ds, cfg)
        assert set(out.data_vars) == {"keep_me"}


def test_fill_missing_satellite_layers_inserts_mean_from_other_years():
    coords = {"y": [0, 1], "x": [0, 1]}
    master = xr.Dataset(
        {
            "2018_04_ndvi": xr.DataArray([[2.0, 2.0], [2.0, 2.0]], coords=coords, dims=("y", "x")),
            "2019_04_ndvi": xr.DataArray([[4.0, 4.0], [4.0, 4.0]], coords=coords, dims=("y", "x")),
        }
    )
    year_ds = xr.Dataset()
    filled = _fill_missing_satellite_layers(year_ds, master, ["2017_04_ndvi"], 2017)
    assert "2017_04_ndvi" in filled.data_vars
    xr.testing.assert_allclose(
        filled["2017_04_ndvi"],
        xr.DataArray([[3.0, 3.0], [3.0, 3.0]], coords=coords, dims=("y", "x")),
    )


def test_fill_missing_satellite_layers_skips_non_satellite_variables():
    coords = {"y": [0], "x": [0]}
    master = xr.Dataset({"2018_class0_BGROUND_distance": xr.DataArray([1.0], coords=coords, dims=("y", "x"))})
    year_ds = xr.Dataset()
    filled = _fill_missing_satellite_layers(
        year_ds, master, ["2017_class0_BGROUND_distance"], 2017
    )
    assert "2017_class0_BGROUND_distance" not in filled.data_vars
