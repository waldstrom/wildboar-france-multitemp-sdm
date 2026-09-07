import pandas as pd
import geopandas as gpd
import numpy as np
from scripts.thinning import run_thinning, grid_thin


def test_source_filter(tmp_path):
    csv = tmp_path / "points.csv"
    df = pd.DataFrame({
        "dd long": [0,1,2],
        "dd lat": [0,1,2],
        "year-int": [2022,2022,2022],
        "month-int": [1,1,1],
        "source": ["GBIF","RRN","GBIF"],
        "season": ["Winter","Winter","Winter"],
        "semester": ["2022-Winter"]*3,
        "month": [1,1,1],
    })
    df.to_csv(csv, index=False)
    cfg = {
        "presence_data": {
            "file": str(csv),
            "thinning": {"enable": False, "save_thinned_points": str(tmp_path/"out.geojson")},
            "year_filter": [2022,2022],
            "source_filter": ["RRN"],
        },
        "seasons": {"active": []},
        "boundary_mask": str(csv),
    }
    # create dummy mask raster
    import numpy as np
    import rasterio
    from rasterio.transform import from_origin
    mask = tmp_path / "mask.tif"
    with rasterio.open(mask, 'w', driver='GTiff', height=1, width=1, count=1, dtype='uint8', crs='EPSG:2154', transform=from_origin(0,1,1,1)) as dst:
        dst.write(np.array([[1]], dtype='uint8'),1)
    cfg["boundary_mask"] = str(mask)
    gdf = run_thinning(cfg, env_df=None)
    assert len(gdf) == 1
    assert gdf.iloc[0].dataset == "RRN"


def test_mahalanobis_percentage_per_dataset(tmp_path):
    csv = tmp_path / "points.csv"
    df = pd.DataFrame(
        {
            "dd long": [0, 1, 2, 3, 4, 5],
            "dd lat": [0, 1, 2, 3, 4, 5],
            "year-int": [2022] * 6,
            "month-int": [1] * 6,
            "source": ["RRN", "RRN", "RRN", "GBIF", "GBIF", "GBIF"],
            "season": ["Winter"] * 6,
            "semester": ["2022-Winter"] * 6,
            "month": [1] * 6,
        }
    )
    df.to_csv(csv, index=False)
    cfg = {
        "presence_data": {
            "file": str(csv),
            "thinning": {
                "enable": True,
                "methods": [
                    {
                        "name": "mahalanobis",
                        "enable": True,
                        "params": {
                            "variables": ["v"],
                            "percentage": {"RRN": 0.33, "GBIF": 0.67},
                        },
                    }
                ],
                "save_thinned_points": str(tmp_path / "out.geojson"),
            },
            "year_filter": [2022, 2022],
        },
        "seasons": {"active": []},
        "boundary_mask": str(csv),
    }
    # create dummy mask raster
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
    cfg["boundary_mask"] = str(mask)

    env_df = pd.DataFrame({"v": np.arange(6)})
    np.random.seed(0)
    gdf = run_thinning(cfg, env_df=env_df)
    assert len(gdf[gdf.dataset == "RRN"]) == 1
    assert len(gdf[gdf.dataset == "GBIF"]) == 2


def test_grid_thin_basic():
    df = pd.DataFrame(
        {
            "x": [0, 1000, 2000, 3000],
            "y": [0, 0, 0, 0],
        }
    )
    thinned = grid_thin(df, 2)
    assert len(thinned) == 2
