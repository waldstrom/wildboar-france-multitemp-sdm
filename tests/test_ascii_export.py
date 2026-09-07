from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from scripts import utils


def _fake_transform(xllcorner: float, yllcorner: float, cellsize: float, height: int):
    """Create a simple object mimicking ``rasterio.Affine``."""

    top_left_y = yllcorner + cellsize * height
    return SimpleNamespace(
        a=cellsize,
        b=0.0,
        c=xllcorner,
        d=0.0,
        e=-cellsize,
        f=top_left_y,
    )


def test_write_ascii_raster_includes_header_and_nodata(tmp_path):
    array = np.array(
        [
            [1.0, 2.0, 3.0, 4.0],
            [5.0, np.nan, 7.5, 8.0],
            [9.0, 10.0, 11.0, 12.0],
        ],
        dtype="float32",
    )
    transform = _fake_transform(91000.0, 6124000.0, 1000.0, array.shape[0])
    out_path = tmp_path / "grid.asc"

    utils._write_ascii_raster(array, transform, out_path)

    text = out_path.read_text()
    assert text.endswith("\n")
    lines = [line for line in text.splitlines() if line]

    assert lines[0] == "ncols        4"
    assert lines[1] == "nrows        3"
    assert lines[2].split()[0] == "xllcorner"
    assert lines[2].split()[-1] == "91000.000000000000"
    assert lines[3].split()[0] == "yllcorner"
    assert lines[3].split()[-1] == "6124000.000000000000"
    assert lines[4] == "cellsize     1000.000000000000"
    assert lines[5] == "NODATA_value  -9999"
    assert lines[6].startswith("1.000000 2.000000 3.000000 4.000000")
    assert "-9999.000000" in lines[7]


def test_ensure_consistent_grid_detects_shift():
    reference = utils._ensure_consistent_grid(
        None, 2, 2, _fake_transform(0.0, 0.0, 1000.0, 2)
    )

    with pytest.raises(ValueError):
        utils._ensure_consistent_grid(
            reference, 2, 2, _fake_transform(500.0, 0.0, 1000.0, 2)
        )

