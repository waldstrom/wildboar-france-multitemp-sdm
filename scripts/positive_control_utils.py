"""Utilities for generating positive-control rasters.

These helpers centralise the creation of the synthetic ``positive_control``
layer that is used by multiple evaluation workflows (multitemporal and
monotemporal variants).  The implementation relies on the default floor-based
indexing of :func:`rasterio.transform.rowcol` which matches the cell assignment
scheme of :func:`scripts.utils.deduplicate_cells` and
:func:`scripts.utils.extract_from_stack`.  Keeping all components on the same
indexing convention prevents background samples from inheriting the ``1`` values
reserved for presence pixels – a mismatch that would otherwise crater precision
and PR-AUC metrics during the sanity checks.
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.transform import rowcol
import xarray as xr

__all__ = ["positive_control_da", "write_positive_control_test_layer"]


def positive_control_da(
    stack_path: Path,
    pres_train: gpd.GeoDataFrame,
    *,
    name: str = "positive_control",
) -> xr.DataArray:
    """Create a positive-control raster aligned with ``pres_train`` points.

    The raster uses ``boundary_mask.tif`` within ``stack_path`` as spatial
    reference.  Each grid cell containing at least one presence record receives
    the value ``1`` while cells inside the study area without presences are set
    to ``0``.  All remaining pixels are marked as ``NaN``.  The raster is written
    to ``stack_path / f"{name}.tif"`` and returned for optional in-memory reuse.
    """

    reference = stack_path / "boundary_mask.tif"
    if not reference.exists():
        raise FileNotFoundError(
            f"Boundary mask not found in combined stack at {reference}"
        )

    with rasterio.open(reference) as src:
        mask_arr = src.read(1)
        transform = src.transform
        meta = src.meta.copy()

    valid = np.isfinite(mask_arr) & (mask_arr > 0)
    control = np.zeros_like(mask_arr, dtype="float32")
    control[~valid] = np.nan

    if "x" not in pres_train.columns:
        pres_train = pres_train.assign(x=pres_train.geometry.x)
    if "y" not in pres_train.columns:
        pres_train = pres_train.assign(y=pres_train.geometry.y)

    xs = pres_train["x"].to_numpy()
    ys = pres_train["y"].to_numpy()

    # Rely on the default ``math.floor`` behaviour.  Using ``round`` would shift
    # points that fall into the upper/right half of a pixel into the neighbouring
    # cell, breaking the perfect separability of the synthetic control layer.
    rows, cols = rowcol(transform, xs, ys)
    rows = np.asarray(rows, dtype="int64")
    cols = np.asarray(cols, dtype="int64")

    within = (
        (rows >= 0)
        & (rows < control.shape[0])
        & (cols >= 0)
        & (cols < control.shape[1])
    )
    control[rows[within], cols[within]] = 1.0

    meta.update(dtype="float32", count=1, nodata=np.nan)
    out_file = stack_path / f"{name}.tif"
    with rasterio.open(out_file, "w", **meta) as dst:
        dst.write(control, 1)

    transform = meta["transform"]
    y_coords = transform.f + (np.arange(control.shape[0]) + 0.5) * transform.e
    x_coords = transform.c + (np.arange(control.shape[1]) + 0.5) * transform.a

    return xr.DataArray(
        control,
        coords={"y": y_coords, "x": x_coords},
        dims=("y", "x"),
        name=name,
    )


def write_positive_control_test_layer(
    stack_path: Path,
    *,
    name: str = "positive_control",
) -> None:
    """Write a zero-valued positive-control raster for the test stack."""

    reference = stack_path / "boundary_mask.tif"
    if not reference.exists():
        raise FileNotFoundError(
            f"Boundary mask not found in test stack at {reference}"
        )

    with rasterio.open(reference) as src:
        mask_arr = src.read(1)
        meta = src.meta.copy()

    valid = np.isfinite(mask_arr) & (mask_arr > 0)
    control = np.zeros_like(mask_arr, dtype="float32")
    control[~valid] = np.nan

    meta.update(dtype="float32", count=1, nodata=np.nan)
    out_file = stack_path / f"{name}.tif"
    with rasterio.open(out_file, "w", **meta) as dst:
        dst.write(control, 1)
