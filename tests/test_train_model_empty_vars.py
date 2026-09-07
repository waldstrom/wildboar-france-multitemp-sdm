import geopandas as gpd
import pytest
from shapely.geometry import Point

from scripts.maxent_training import train_model


def test_train_model_requires_predictors(tmp_path):
    pres = gpd.GeoDataFrame({'geometry': [Point(0, 0)]}, crs='EPSG:4326')
    bg = gpd.GeoDataFrame({'geometry': [Point(1, 1)]}, crs='EPSG:4326')
    cfg = {
        'experiment': {'name': 'test', 'vis_dir': str(tmp_path)},
        'outputs': {'maps_dir': str(tmp_path), 'figures_dir': str(tmp_path)},
        'maxent': {}
    }
    with pytest.raises(RuntimeError, match="No predictors were provided"):
        train_model(cfg, tmp_path, [], pres, bg)
