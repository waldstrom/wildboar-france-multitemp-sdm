# =============================================================================
# Project Title: Wild Boar Distribution Model for France
# Path: scripts/bias_grid.py
# Purpose: Create kernel-density bias grids to correct for survey effort.
# Process Step: Generates and saves bias rasters based on observation density.
# Created by Adrian Meyer, Institute Geomatics, FHNW (adrian.meyer@fhnw.ch)
# Created: July 2025
# Version: v.0.1.0
# =============================================================================
"""
Kernel-density bias grids - optional survey-effort correction.
"""
from __future__ import annotations
import logging
from pathlib import Path
import geopandas as gpd
import numpy as np
import rasterio
from rasterio import warp
from scipy.ndimage import gaussian_filter
from tqdm import tqdm
from scripts.utils import ensure_dir, validate_crs, ensure_gdf_crs, get_processed_dir
import shutil


def _bias_from_lc_urban(cfg: dict) -> Path:
    """Generate a bias raster by penalising high urban cover.

    The weight decays exponentially with the proportion of urban land-cover so
    that observations near densely populated areas receive lower sampling
    probability. The decay parameter ``alpha`` is configurable in ``config.yaml``.
    """
    out = Path(cfg["bias_grid"]["output_file"])
    ensure_dir(out.parent)
    proc_dir = ensure_dir(get_processed_dir())
    proc_file = proc_dir / "bias_grid.tif"
    if proc_file.exists():
        if not out.exists():
            shutil.copy2(proc_file, out)
        return out

    pattern_cfg = cfg["predictors"]["layers"]["landcover_fraction"]
    years = pattern_cfg.get("years", [])
    files = [
        Path(pattern_cfg["pattern"].format(year=y, **{"class": "11_URBAN"}))
        for y in years
    ]
    arrays = []
    meta = None
    for f in files:
        if not f.exists():
            continue
        with rasterio.open(f) as src:
            src_crs = validate_crs(src.crs, context=f"bias raster {f}")
            arr = src.read(1, masked=True).astype("float32")
            arrays.append(arr)
            if meta is None:
                meta = src.meta.copy()
                meta["crs"] = src_crs
    if not arrays:
        raise RuntimeError("No LC Urban rasters found for bias grid")
    urban_mean = np.mean(np.stack(arrays), axis=0)
    alpha = cfg["bias_grid"].get("alpha", 3)
    # Penalise proximity to urban areas using an exponential decay function
    bias = np.exp(-alpha * np.clip(urban_mean, 0, 1))
    bias = (bias - bias.min()) / (bias.max() - bias.min() + 1e-6)

    meta.update(dtype="float32", compress="LZW")

    tgt_res = cfg["predictors"]["target_resolution_m"]
    tgt_meta = {
        "driver": "GTiff",
        "height": 1000,
        "width": 1000,
        "count": 1,
        "dtype": "float32",
        "crs": "EPSG:2154",
        "transform": rasterio.Affine(
            tgt_res,
            0,
            91000.0,
            0,
            -tgt_res,
            6124000.0 + 1000 * tgt_res,
        ),
        "compress": "LZW",
    }

    if (
        meta.get("width") != tgt_meta["width"]
        or meta.get("height") != tgt_meta["height"]
        or meta.get("transform") != tgt_meta["transform"]
    ):
        aligned = np.empty((tgt_meta["height"], tgt_meta["width"]), dtype="float32")
        warp.reproject(
            bias,
            aligned,
            src_transform=meta["transform"],
            src_crs=validate_crs(meta.get("crs"), context="bias grid"),
            dst_transform=tgt_meta["transform"],
            dst_crs=tgt_meta["crs"],
            resampling=warp.Resampling.bilinear,
            dst_nodata=0,
        )
        bias = aligned
        meta = tgt_meta
    else:
        meta = {**tgt_meta}

    with rasterio.open(out, "w", **meta) as dst:
        dst.write(bias, 1)
    if not proc_file.exists():
        shutil.copy2(out, proc_file)
    return out

log = logging.getLogger(__name__)

# -----------------------------------------------------------------------------  
def _bias_from_occurrence_density(gdf: gpd.GeoDataFrame, cfg: dict) -> Path:
    """Kernel density smoothing of presence locations."""
    proc_dir = ensure_dir(get_processed_dir())
    proc_file = proc_dir / "bias_grid.tif"
    out = Path(cfg["bias_grid"]["output_file"])
    ensure_dir(out.parent)

    if proc_file.exists():
        log.info("Using cached bias grid from %s", proc_file)
        if not out.exists():
            shutil.copy2(proc_file, out)
        return out

    kernel_km = cfg["bias_grid"]["kernel_bandwidth_km"]
    res_km = cfg["bias_grid"]["resolution_km"]

    # Raster parameters - hard-coded 1 km grid
    width = height = 1000
    transform = rasterio.transform.from_origin(
        91000.0, 6124000.0 + 1000 * 1000, res_km * 1000, res_km * 1000
    )
    bias = np.zeros((height, width), dtype="float32")

    for geom in tqdm(gdf.geometry, desc="bias points"):
        col = int((geom.x - 91000) // (res_km * 1000))
        row = int((6124000 + 1000 * 1000 - geom.y) // (res_km * 1000))
        if 0 <= row < height and 0 <= col < width:
            bias[row, col] += 1.0

    sigma = kernel_km / res_km
    bias = gaussian_filter(bias, sigma=sigma)
    bias /= bias.max()

    meta = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": 1,
        "dtype": "float32",
        "crs": "EPSG:2154",
        "transform": transform,
        "compress": "LZW",
    }
    with rasterio.open(out, "w", **meta) as dst:
        dst.write(bias, 1)
    if not proc_file.exists():
        shutil.copy2(out, proc_file)
    log.info(
        "Bias grid written to %s (bandwidth %g km, sigma=%g px)", out, kernel_km, sigma
    )
    print("    -> Bias grid saved", out)
    return out


def _bias_from_predefined(cfg: dict) -> Path:
    """Use a precomputed bias raster located on disk."""
    out = Path(cfg["bias_grid"]["output_file"])
    ensure_dir(out.parent)
    src = cfg["bias_grid"].get("file")
    pattern = cfg["bias_grid"].get("pattern")
    if not src and pattern:
        seasons = cfg.get("seasons", {}).get("active", [])
        if len(seasons) == 1:
            src = pattern.format(season=seasons[0])
        else:
            src = pattern.format(season="All_Semesters")
    if not src:
        raise RuntimeError("No path specified for predefined bias grid")
    src = Path(src)
    if not src.exists():
        raise FileNotFoundError(src)
    if out.resolve() != src.resolve():
        shutil.copy2(src, out)
    return out


def create_bias_grid(gdf: gpd.GeoDataFrame, cfg: dict) -> Path:
    gdf = ensure_gdf_crs(gdf, context="bias input")
    if not cfg["bias_grid"]["enable"]:
        log.info("Bias grid disabled - skipping")
        return Path()

    strategy = cfg["bias_grid"].get("strategy", "occurrence_density")
    if strategy == "lc_urban":
        print("    -> Generating bias grid from LC Urban")
        out = _bias_from_lc_urban(cfg)
        log.info("Bias grid written to %s (strategy lc_urban)", out)
        return out
    if strategy == "predefined":
        print("    -> Using predefined bias grid")
        out = _bias_from_predefined(cfg)
        log.info("Bias grid copied from %s", out)
        return out
    # Default is kernel density of presence locations
    print("    -> Generating kernel-density bias grid")
    out = _bias_from_occurrence_density(gdf, cfg)
    log.info("Bias grid written to %s (strategy occurrence_density)", out)
    return out
