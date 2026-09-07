# =============================================================================
# Project Title: Wild Boar Distribution Model for France
# Path: scripts/background_sampling.py
# Purpose: Generate pseudo-absence background samples for model training.
# Process Step: Samples points within the study area using optional bias grid.
# Created by Adrian Meyer, Institute Geomatics, FHNW (adrian.meyer@fhnw.ch)
# Created: July 2025
# Version: v.0.1.0
# =============================================================================
"""Pseudo-absence (background) generation."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio

from scripts.utils import (
    ensure_dir,
    ensure_gdf_crs,
    filter_points_by_mask,
    get_processed_dir,
    load_mask,
    validate_crs,
)

log = logging.getLogger(__name__)


def _resolve_bias_grid(cfg: dict, requested: Path) -> Path:
    """Resolve a generated bias grid, then the configured local source grid."""
    if requested.exists():
        return requested

    configured = cfg.get("bias_grid", {}).get("file")
    fallback = (
        Path(configured)
        if configured
        else Path("data/bias-maps/Combined_All_Semesters.asc")
    )
    if fallback != requested:
        log.warning(
            "Bias grid %s not found; trying configured source %s",
            requested,
            fallback,
        )
    if not fallback.exists():
        raise FileNotFoundError(
            "No bias grid was found. Expected either "
            f"{requested} or {fallback}. See data/bias-maps/README.md."
        )
    return fallback


def sample_background(cfg: dict, bias_tif: Path | None = None) -> gpd.GeoDataFrame:
    """Sample background locations within the configured boundary mask."""
    print("    -> Sampling background points")
    proc_dir = ensure_dir(get_processed_dir())
    proc_file = proc_dir / "background_pts.gpkg"
    out_file = Path(cfg["background"]["output_file"])
    ensure_dir(out_file.parent)

    if proc_file.exists():
        log.info("Using cached background points from %s", proc_file)
        if not out_file.exists():
            shutil.copy2(proc_file, out_file)
        gdf = gpd.read_file(out_file)
        return ensure_gdf_crs(gdf, context=f"background cache {out_file}")

    n = cfg["background"]["n_points"]
    mask = load_mask(cfg["boundary_mask"])
    valid_rc = np.argwhere(mask)

    weights = None
    if bias_tif and cfg["background"]["use_bias_grid"]:
        path = _resolve_bias_grid(cfg, Path(bias_tif))
        with rasterio.open(path) as src:
            validate_crs(src.crs, context=f"bias grid {path}")
            full_weights = src.read(1)[mask]
            full_weights = np.maximum(full_weights, 0.1)
            weights = full_weights / full_weights.sum()

    def _sample(size: int, start_id: int) -> gpd.GeoDataFrame:
        idx = np.random.choice(len(valid_rc), size=size, replace=True, p=weights)
        rows, cols = valid_rc[idx].T
        xs = 91000 + (cols + np.random.rand(size)) * 1000
        ys = 6124000 + 1000 * 1000 - (rows + np.random.rand(size)) * 1000
        frame = pd.DataFrame(
            {
                "bg_id": np.arange(start_id, start_id + size),
                "source": "background",
            }
        )
        sampled = gpd.GeoDataFrame(
            frame,
            geometry=gpd.points_from_xy(xs, ys),
            crs="EPSG:2154",
        )
        sampled["x"] = sampled.geometry.x
        sampled["y"] = sampled.geometry.y
        return filter_points_by_mask(sampled, mask)

    gdf = gpd.GeoDataFrame(
        columns=["bg_id", "source", "geometry", "x", "y"],
        crs="EPSG:2154",
    )
    while len(gdf) < n:
        needed = n - len(gdf)
        new = _sample(needed, len(gdf))
        gdf = pd.concat([gdf, new], ignore_index=True)

    gdf = gdf.iloc[:n].reset_index(drop=True)
    gdf["bg_id"] = np.arange(len(gdf))
    gdf.to_file(out_file, driver="GPKG")
    if not proc_file.exists():
        shutil.copy2(out_file, proc_file)
    log.info("Background points (%d) written to %s", n, out_file)
    print("    -> Background saved", out_file)
    return gdf
