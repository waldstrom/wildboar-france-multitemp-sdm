import numpy as np
import geopandas as gpd
from shapely.geometry import Point
import rasterio
from rasterio.transform import Affine

from scripts.positive_control_utils import positive_control_da
from scripts.utils import extract_from_stack, drop_background_overlaps


def _write_boundary_mask(path):
    transform = Affine(1000, 0, 91000, 0, -1000, 6124000)
    data = np.ones((2, 2), dtype="float32")
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=data.shape[0],
        width=data.shape[1],
        count=1,
        dtype="float32",
        transform=transform,
        nodata=np.nan,
    ) as dst:
        dst.write(data, 1)


def test_positive_control_alignment(tmp_path):
    stack_dir = tmp_path / "stack"
    stack_dir.mkdir()
    boundary = stack_dir / "boundary_mask.tif"
    _write_boundary_mask(boundary)

    # Presence point placed in the upper-right quadrant of the first pixel. When
    # ``rowcol`` uses rounding this coordinate is nudged into the neighbouring
    # cell which breaks the perfect separability of the synthetic control layer.
    presence = gpd.GeoDataFrame(
        geometry=[Point(91999.0, 6123500.0)],
        crs="EPSG:2154",
    )

    da = positive_control_da(stack_dir, presence, name="positive_control")
    assert da.shape == (2, 2)

    values = extract_from_stack(stack_dir, presence, ["positive_control"])
    assert values.shape == (1, 1)
    assert values[0, 0] == 1.0

    background = gpd.GeoDataFrame(
        geometry=[Point(91001.0, 6122501.0)],
        crs="EPSG:2154",
    )
    bg_values = extract_from_stack(stack_dir, background, ["positive_control"])
    assert bg_values.shape == (1, 1)
    assert bg_values[0, 0] == 0.0


def test_background_overlap_removal(tmp_path):
    stack_dir = tmp_path / "stack"
    stack_dir.mkdir()
    boundary = stack_dir / "boundary_mask.tif"
    _write_boundary_mask(boundary)

    presence = gpd.GeoDataFrame(
        geometry=[Point(91999.0, 6123500.0)],
        crs="EPSG:2154",
    )

    background = gpd.GeoDataFrame(
        geometry=[
            Point(91950.0, 6123450.0),  # same cell as presence
            Point(92001.0, 6122500.0),  # different cell
        ],
        crs="EPSG:2154",
    )

    positive_control_da(stack_dir, presence, name="positive_control")

    raw_bg = extract_from_stack(stack_dir, background, ["positive_control"])
    assert raw_bg.shape == (2, 1)
    assert set(raw_bg.ravel()) == {0.0, 1.0}

    cleaned_bg = drop_background_overlaps(
        presence, background, context="test background"
    )
    assert len(cleaned_bg) == 1

    cleaned_values = extract_from_stack(stack_dir, cleaned_bg, ["positive_control"])
    assert cleaned_values.shape == (1, 1)
    assert cleaned_values[0, 0] == 0.0
