import numpy as np
import rasterio
from rasterio.transform import from_origin
from scripts.bias_grid import _bias_from_lc_urban, _bias_from_predefined
from pathlib import Path

def test_lc_urban_bias(tmp_path):
    # create synthetic LC Urban rasters
    meta = {
        "driver": "GTiff", "height": 2, "width": 2, "count": 1,
        "dtype": "float32", "crs": "EPSG:2154",
        "transform": from_origin(0,2,1,1)
    }
    arr = np.array([[0.0,1.0],[0.5,0.2]], dtype="float32")
    f = tmp_path/"2019_class11_URBAN.tif"
    with rasterio.open(f, "w", **meta) as dst:
        dst.write(arr,1)
    cfg={"bias_grid":{"output_file":str(tmp_path/"bias.tif"),"alpha":3},
        "predictors":{"target_resolution_m":1,
            "layers":{"landcover_fraction":{"pattern":str(tmp_path/"{year}_class{class}.tif"),"years":[2019]}}}}
    out=_bias_from_lc_urban(cfg)
    with rasterio.open(out) as src:
        data=src.read(1)
    assert data.min()>=0 and data.max()<=1


def test_predefined_bias(tmp_path):
    meta = {
        "driver": "GTiff", "height": 1, "width": 1, "count": 1,
        "dtype": "float32", "crs": "EPSG:2154",
        "transform": from_origin(0,1,1,1)
    }
    arr = np.array([[0.5]], dtype="float32")
    src = tmp_path/"bias_src.tif"
    with rasterio.open(src, "w", **meta) as dst:
        dst.write(arr,1)
    cfg = {"bias_grid": {"output_file": str(tmp_path/"bias_out.tif"), "file": str(src)}}
    out = _bias_from_predefined(cfg)
    with rasterio.open(out) as ds:
        data = ds.read(1)
    assert np.allclose(data, arr)
