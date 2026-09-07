# =============================================================================
# Project Title: Wild Boar Distribution Model for France
# Path: scripts/multitemp_preprocessing.py
# Purpose: Export 3x2 multitemporal predictor matrices as GeoTIFF layers.
# Process Step: Writes each predictor in a combined multitemporal stack to its
#               own GeoTIFF file prior to any filtering. These GeoTIFFs can be
#               used as inputs for subsequent feature filtering and model
#               training.
# Created by Adrian Meyer, Institute Geomatics, FHNW (adrian.meyer@fhnw.ch)
# Created: July 2025
# Version: v.0.1.0
# =============================================================================
"""Utility functions for multitemporal preprocessing."""
from __future__ import annotations

from pathlib import Path
import logging
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import xarray as xr
import rasterio
from rasterio.transform import from_origin

log = logging.getLogger(__name__)


def export_geotiffs(
    ds: xr.Dataset,
    out_dir: Path,
    nodata: float = np.nan,
    n_time_steps: int = 6,
    compress: str | None = "LZW",
    n_workers: int | None = None,
) -> None:
    """Export each data variable in ``ds`` to a multi-band GeoTIFF file.

    Each predictor layer is written to ``<variable>.tif`` with ``n_time_steps``
    bands. If the provided data array lacks an explicit time dimension it is
    duplicated so that the output still contains ``n_time_steps`` bands. This
    allows static predictors to be represented for all time steps without
    special handling upstream.

    Parameters
    ----------
    ds:
        ``xarray.Dataset`` containing predictor layers. Coordinates must be in
        EPSG:2154 and regular on both axes. The dataset should represent the
        combined 3x2 multitemporal matrix.
    out_dir:
        Destination directory. One ``<variable>.tif`` file is written for each
        data variable. Existing files are reused to avoid costly rewrites.
    nodata:
        Value written to disk for missing data (``NaN`` is preserved in memory).
    n_time_steps:
        Number of time steps to encode in the output GeoTIFF. Layers lacking a
        time dimension are repeated ``n_time_steps`` times.
    compress:
        Compression algorithm passed to :func:`rasterio.open`. ``None`` disables
        compression. ``"LZW"`` is recommended because stacks can be very large.
    n_workers:
        Number of worker threads used to write GeoTIFFs. Defaults to the number
        of available CPU cores but will not exceed the number of variables in
        ``ds``.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # GeoTIFF rasters expect the first row to represent the northernmost
    # coordinates.  Ensure the dataset follows this convention so that the
    # merged 3x2 matrix aligns with shifted point locations.  X coordinates
    # increase to the east while Y decreases to the south.
    ds = ds.sortby("x").sortby("y", ascending=False)

    x_coords = ds.x.values
    y_coords = ds.y.values
    if len(x_coords) < 2 or len(y_coords) < 2:
        raise ValueError("Dataset must have at least two cells per axis")
    res_x = float(abs(x_coords[1] - x_coords[0]))
    res_y = float(abs(y_coords[1] - y_coords[0]))
    transform = from_origin(float(x_coords.min()), float(y_coords.max()), res_x, res_y)

    if n_workers is None:
        n_workers = min(len(ds.data_vars), os.cpu_count() or 1)

    def _write(name: str, da: xr.DataArray) -> None:
        out_file = out_dir / f"{name}.tif"
        if out_file.exists():
            log.debug("GeoTIFF for %s already exists - reusing", name)
            return
        data = da.values
        if data.ndim == 2:
            data = np.repeat(data[np.newaxis, :, :], n_time_steps, axis=0)
        elif data.ndim == 3 and data.shape[0] == n_time_steps:
            pass
        else:
            raise ValueError(
                f"Unsupported data shape {data.shape} for variable {name}"
            )
        data = data.astype("float32")
        nodata_val = np.nan if np.isnan(nodata) else float(nodata)
        with rasterio.open(
            out_file,
            "w",
            driver="GTiff",
            height=data.shape[1],
            width=data.shape[2],
            count=data.shape[0],
            dtype="float32",
            crs="EPSG:2154",
            transform=transform,
            nodata=nodata_val,
            compress=compress,
        ) as dst:
            for i in range(data.shape[0]):
                dst.write(np.where(np.isnan(data[i]), nodata_val, data[i]), i + 1)
        log.debug("Wrote GeoTIFF for %s", name)

    with ThreadPoolExecutor(max_workers=n_workers) as exe:
        futures = [exe.submit(_write, name, da) for name, da in ds.data_vars.items()]
        for fut in futures:
            fut.result()
