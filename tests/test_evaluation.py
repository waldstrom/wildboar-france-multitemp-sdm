import numpy as np
import pandas as pd
import geopandas as gpd
from scripts.evaluation import evaluate

def test_evaluate_handles_nans(tmp_path):
    y_true = np.array([1, 0, 1, 0])
    y_score = np.array([0.9, np.nan, 0.8, 0.2])
    cfg = {
        "outputs": {"figures_dir": str(tmp_path)},
        "evaluation": {"classification_threshold": "auto"},
    }
    gdf = gpd.GeoDataFrame(
        pd.DataFrame({"x": range(4), "y": range(4)}),
        geometry=gpd.points_from_xy(range(4), range(4)),
        crs="EPSG:4326",
    )
    evaluate("tst", y_true, y_score, cfg, n_params=1, gdf_train=gdf)
    assert (tmp_path / "calibration_plot.png").exists()
    assert (tmp_path / "tst_train_nan.gpkg").exists()
