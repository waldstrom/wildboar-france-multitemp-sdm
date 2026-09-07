import pytest
from pathlib import Path

from scripts.multitemporal import _expected_var_names


def test_expected_var_names_handles_wildcard_patterns(tmp_path):
    (tmp_path / "fragmentation_density.asc").write_text("x")
    (tmp_path / "small_woody_features.asc").write_text("x")
    cfg = {
        "predictors": {
            "layers": {
                "special": {
                    "pattern": str(tmp_path / "*.asc"),
                }
            }
        }
    }
    names = _expected_var_names(cfg, 2020, "winter")
    assert "fragmentation_density" in names
    assert "small_woody_features" in names


def test_expected_var_names_filters_by_season(tmp_path):
    (tmp_path / "2m_temperature_2020_winter_min.asc").write_text("x")
    (tmp_path / "2m_temperature_2020_summer_mean.asc").write_text("x")
    cfg = {
        "predictors": {
            "layers": {
                "era5": {
                    "pattern": str(tmp_path / "*_{year}_{season}_*.asc"),
                    "years": [2020],
                    "seasons": ["winter", "summer"],
                }
            }
        }
    }
    names = _expected_var_names(cfg, 2020, "winter")
    assert "2m_temperature_2020_winter_min" in names
    assert "2m_temperature_2020_summer_mean" not in names
