import geopandas as gpd
from shapely.geometry import Point
import numpy as np
from scripts.cross_validation import environmental_block_split
from pathlib import Path
import pandas as pd

def test_environmental_block_split(tmp_path):
    gdf = gpd.GeoDataFrame(geometry=[Point(x,0) for x in range(6)], crs="EPSG:2154")
    # create dummy zarr stack with one variable
    import xarray as xr
    data = xr.DataArray(np.arange(6).reshape(3,2).astype(float), dims=["y","x"],
                        coords={"x": [0,1], "y": [0,1,2]}, name="var1")
    ds = xr.Dataset({"var1": data})
    stack = tmp_path/"stack.zarr"
    ds.to_zarr(stack)
    splits = list(environmental_block_split(gdf, stack, ["var1"], k=2, random_state=0))
    assert len(splits) == 2
    # ensure each fold has at least one test record
    assert all(len(test)>0 for _,test in splits)
