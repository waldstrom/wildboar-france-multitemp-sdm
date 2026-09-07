import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
from pathlib import Path

from scripts.run_pipeline import _load_selected_vars, _load_thinned_presences


def test_load_selected_vars_fallback(tmp_path):
    ds = xr.Dataset(
        {
            "var1": ("y", np.random.rand(2)),
            "var2": ("y", np.random.rand(2)),
            "boundary_mask": ("y", np.ones(2)),
        },
        coords={"y": [0, 1]},
    )
    stack_path = tmp_path / "stack.zarr"
    ds.to_zarr(stack_path)
    cfg = {"predictors": {"variable_selection": {"save_selected_list": str(tmp_path / "sel.txt")}}}
    vars_used = _load_selected_vars(cfg, stack_path)
    assert set(vars_used) == {"var1", "var2"}
    assert (tmp_path / "sel.txt").read_text().splitlines() == vars_used


def test_load_thinned_presences_fallback(tmp_path):
    csv = tmp_path / "points.csv"
    df = pd.DataFrame(
        {
            "dd long": [0, 1],
            "dd lat": [0, 1],
            "year-int": [2022, 2022],
            "month-int": [1, 1],
            "source": ["GBIF", "GBIF"],
            "season": ["Winter", "Winter"],
            "semester": ["2022-Winter", "2022-Winter"],
            "month": [1, 1],
        }
    )
    df.to_csv(csv, index=False)

    import rasterio
    from rasterio.transform import from_origin

    mask = tmp_path / "mask.tif"
    with rasterio.open(
        mask,
        "w",
        driver="GTiff",
        height=1,
        width=1,
        count=1,
        dtype="uint8",
        crs="EPSG:2154",
        transform=from_origin(0, 1, 1, 1),
    ) as dst:
        dst.write(np.array([[1]], dtype="uint8"), 1)

    cfg = {
        "presence_data": {
            "file": str(csv),
            "thinning": {"save_thinned_points": str(tmp_path / "out.geojson")},
            "year_filter": [2022, 2022],
        },
        "seasons": {"active": []},
        "boundary_mask": str(mask),
    }

    gdf = _load_thinned_presences(cfg)
    assert len(gdf) == 2
    assert Path(cfg["presence_data"]["thinning"]["save_thinned_points"]).exists()
