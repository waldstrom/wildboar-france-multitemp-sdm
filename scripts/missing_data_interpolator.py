# =============================================================================
# Project Title: Wild Boar Distribution Model for France
# Path: scripts/missing_data_interpolator.py
# Purpose: Fill gaps in predictor rasters within the boundary mask.
# Process Step: Interpolates missing cells using bicubic interpolation and
#               writes results back to the original rasters while keeping
#               backups and visual diagnostics.
# Created by Adrian Meyer, Institute Geomatics, FHNW (adrian.meyer@fhnw.ch)
# Created: November 2025
# Version: v.0.1.0
# =============================================================================
"""Interpolate missing values in predictor rasters."""
from __future__ import annotations

import argparse
import logging
from datetime import datetime
from pathlib import Path
import shutil

import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio import warp
from rasterio.enums import Resampling
from scipy.interpolate import griddata

from scripts.utils import ensure_dir

log = logging.getLogger(__name__)

SUPPORTED_EXTS = {".tif", ".tiff", ".asc"}


def _resample_boundary(boundary_file: Path, tgt_meta: dict) -> np.ndarray:
    """Return boundary mask resampled to ``tgt_meta`` grid."""
    with rasterio.open(boundary_file) as src:
        bnd = src.read(1)
        dst_crs = tgt_meta.get("crs") or src.crs
        if (
            src.width != tgt_meta["width"]
            or src.height != tgt_meta["height"]
            or src.transform != tgt_meta["transform"]
            or src.crs != dst_crs
        ):
            out = np.empty((tgt_meta["height"], tgt_meta["width"]), dtype="float32")
            warp.reproject(
                bnd,
                out,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=tgt_meta["transform"],
                dst_crs=dst_crs,
                resampling=Resampling.nearest,
            )
            bnd = out
    return bnd == 1


def _bicubic_fill(data: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fill ``NaN`` values inside ``mask`` using bicubic interpolation."""
    missing = np.isnan(data) & mask
    if not np.any(missing):
        return data, missing

    rows, cols = np.indices(data.shape)
    valid = (~np.isnan(data)) & mask
    points = np.column_stack((cols[valid], rows[valid]))
    values = data[valid]
    targets = np.column_stack((cols[missing], rows[missing]))

    data[missing] = griddata(points, values, targets, method="cubic")
    remain = np.isnan(data) & mask
    if np.any(remain):
        data[remain] = griddata(
            points,
            values,
            np.column_stack((cols[remain], rows[remain])),
            method="nearest",
        )
    return data, missing


def _process_raster(
    raster_file: Path,
    boundary_file: Path,
    plot_root: Path,
) -> None:
    """Interpolate missing values for ``raster_file`` and save diagnostics."""
    with rasterio.open(raster_file) as src:
        meta = src.meta.copy()
        arr = src.read(1).astype("float32")
        nodata = src.nodata if src.nodata is not None else -9999
        arr[arr == nodata] = np.nan
    mask = _resample_boundary(boundary_file, meta)

    arr, imputed = _bicubic_fill(arr, mask)
    arr[~mask] = np.nan

    imputed_data = np.where(np.isnan(arr), nodata, arr).astype(meta["dtype"])
    meta.update(nodata=nodata)
    with rasterio.open(raster_file, "w", **meta) as dst:
        dst.write(imputed_data, 1)

    if np.any(imputed):
        rel_plot = plot_root / raster_file.with_suffix(".png").name
        ensure_dir(rel_plot.parent)
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.imshow(imputed, cmap="Reds")
        ax.set_title(f"Imputed cells: {raster_file.name}")
        ax.axis("off")
        fig.savefig(rel_plot, dpi=150, bbox_inches="tight")
        plt.close(fig)

    log.info("Processed %s - %d cells filled", raster_file, np.sum(imputed))


def main(data_dir: Path, boundary_file: Path, exps_dir: Path) -> None:
    """Fix missing predictor values in ``data_dir``."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_root = ensure_dir(exps_dir / f"feature_fix_{timestamp}")
    backup_root = ensure_dir(exp_root / "data")
    plot_root = ensure_dir(exp_root / "plots")

    raster_files = [
        p
        for p in data_dir.rglob("*")
        if p.suffix.lower() in SUPPORTED_EXTS and p.name != boundary_file.name
    ]
    for rf in raster_files:
        rel = rf.relative_to(data_dir)
        bkp = backup_root / rel
        ensure_dir(bkp.parent)
        shutil.copy2(rf, bkp)
        _process_raster(rf, boundary_file, plot_root / rel.parent)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data"), help="Root data directory")
    parser.add_argument(
        "--boundary",
        type=Path,
        default=Path("data/boundary_france.asc"),
        help="Boundary mask raster",
    )
    parser.add_argument("--exps", type=Path, default=Path("exps"), help="Experiments directory")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    main(args.data, args.boundary, args.exps)
