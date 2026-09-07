# =============================================================================
# Project Title: Wild Boar Distribution Model for France
# Path: scripts/preprocessing.py
# Purpose: Harmonise rasters by resampling them to the common 1 km grid.
# Process Step: Aligns all predictor layers according to parameters in the config.
# Created by Adrian Meyer, Institute Geomatics, FHNW (adrian.meyer@fhnw.ch)
# Created: July 2025
# Version: v.0.1.0
# =============================================================================
"""
Raster harmonisation - snaps *all* predictors to the 1 km grid defined in config.
"""
from __future__ import annotations
import logging
from pathlib import Path
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
import rasterio
from rasterio.enums import Resampling
from rasterio import warp
from tqdm import tqdm
import numpy as np
import xarray as xr
from scripts.utils import ensure_dir, validate_crs, get_processed_dir
import shutil

log = logging.getLogger(__name__)

# -----------------------------------------------------------------------------

def _load_boundary_mask(mask_file: Path, tgt_meta: dict) -> np.ndarray:
    """Load boundary mask and return as boolean array on target grid."""
    with rasterio.open(mask_file) as src:
        mask = src.read(1, masked=True)
        src_crs = validate_crs(src.crs, context=f"boundary mask {mask_file}")
        if (
            src.width != tgt_meta["width"]
            or src.height != tgt_meta["height"]
            or src.transform != tgt_meta["transform"]
        ):
            out = np.empty((tgt_meta["height"], tgt_meta["width"]), dtype=mask.dtype)
            warp.reproject(
                mask,
                out,
                src_transform=src.transform,
                src_crs=src_crs,
                dst_transform=tgt_meta["transform"],
                dst_crs=tgt_meta["crs"],
                resampling=Resampling.nearest,
            )
            mask = out
    return mask == 1

# -----------------------------------------------------------------------------  
def resample_raster(
    src_file: Path,
    target_meta: dict,
    resampling: str,
    valid_range: tuple[float, float] | None = None,
) -> xr.DataArray:
    """Resample ``src_file`` to ``target_meta`` using nearest neighbour.

    ``valid_range`` defines the acceptable value range. Any numbers outside this
    interval are treated as missing. ``-9999`` is always considered missing.
    """
    resample_algo = Resampling.nearest
    with rasterio.open(src_file) as src:
        data = src.read(1).astype("float32")
        nodata_val = src.nodata if src.nodata is not None else -9999
        data[data == nodata_val] = np.nan
        if valid_range is not None:
            low, high = valid_range
            data[(data < low) | (data > high)] = np.nan

        if (
            src.width == target_meta["width"]
            and src.height == target_meta["height"]
            and src.transform == target_meta["transform"]
        ):
            return xr.DataArray(data, dims=("y", "x"))

        out = np.empty((target_meta["height"], target_meta["width"]), dtype="float32")

        src_crs = validate_crs(src.crs, context=f"raster {src_file}")

        warp.reproject(
            np.where(np.isnan(data), target_meta["nodata"], data),
            out,
            src_transform=src.transform,
            src_crs=src_crs,
            dst_transform=target_meta["transform"],
            dst_crs=target_meta["crs"],
            resampling=resample_algo,
            src_nodata=target_meta["nodata"],
            dst_nodata=target_meta["nodata"],
        )
    out[out == target_meta["nodata"]] = np.nan
    return xr.DataArray(out, dims=("y", "x"))

# -----------------------------------------------------------------------------

def _resample_and_mask(
    args: tuple[Path, dict, str, tuple[float, float] | None, np.ndarray | None]
):
    """Helper for parallel raster resampling and masking."""
    src_file, tgt_meta, resampling, valid, mask_bool = args
    ds = resample_raster(src_file, tgt_meta, resampling, valid_range=valid)

    if "vssi" in src_file.stem.lower():
        ds = ds.interpolate_na(dim="y", method="nearest", fill_value="extrapolate")
        ds = ds.interpolate_na(dim="x", method="nearest", fill_value="extrapolate")

    if mask_bool is not None:
        ds = ds.where(mask_bool)
    return src_file.stem, ds

# -----------------------------------------------------------------------------

def _copy_file(args: tuple[Path, Path, Path]) -> None:
    """Copy a file from ``src_root`` to ``dest_root`` preserving subdirectories."""
    src_file, src_root, dest_root = args
    rel = src_file.relative_to(src_root)
    dest = dest_root / rel
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_file, dest)

# -----------------------------------------------------------------------------  
def build_stack(cfg: dict, use_cache: bool = True) -> Path:
    print("    -> Building predictor stack")
    predictors_cfg = cfg["predictors"]
    # Older configuration files used by the monotemporal utilities may not
    # include a dedicated ``outputs`` block.  ``cfg['monotemporal']['output_dir']``
    # is then the only hint about where results should be written.  To avoid
    # ``KeyError`` failures we gracefully fall back to that location (and finally
    # to a simple ``outputs`` directory).
    out_root = (
        cfg.get("outputs", {}).get("root")
        or cfg.get("monotemporal", {}).get("output_dir")
        or "outputs"
    )
    out_dir = ensure_dir(Path(out_root) / "stack")
    stack_path = out_dir / f"{cfg['experiment']['name']}_predictor_stack.zarr"

    proc_path = None
    if use_cache:
        proc_dir = ensure_dir(get_processed_dir() / "stack")
        proc_path = proc_dir / "predictor_stack.zarr"
        if proc_path.exists():
            log.info("Using cached predictor stack from %s", proc_path)
            if not stack_path.exists():
                from multiprocessing import Pool

                files = [f for f in proc_path.rglob("*") if f.is_file()]
                args_list = [(f, proc_path, stack_path) for f in files]
                with Pool() as pool:
                    list(
                        tqdm(
                            pool.imap_unordered(_copy_file, args_list),
                            total=len(files),
                            desc="copy",
                            leave=False,
                        )
                    )
            return stack_path

    if stack_path.exists():
        log.info("Predictor stack already exists - skipping (%s)", stack_path)
        return stack_path

    # 1. Derive target grid from first raster in list
    sample_ras = next(
        (p for p in Path().rglob("*.asc") if ".git" not in p.parts),
        None,
    )
    if sample_ras is None:
        raise FileNotFoundError(
            "No '.asc' raster files found outside '.git' directory"
        )
    with rasterio.open(sample_ras) as src:
        validate_crs(src.crs, context=f"sample raster {sample_ras}")
        tgt_meta = src.meta.copy()
        # ``target_resolution_m`` is required in most configs but some
        # experimental setups omit it. Default to 1000 m if missing to keep
        # backwards compatibility.
        tgt_res = predictors_cfg.get("target_resolution_m", 1000)
        if "target_resolution_m" not in predictors_cfg:
            log.warning(
                "'target_resolution_m' missing in config - defaulting to %sm",
                tgt_res,
            )
        tgt_nodata = predictors_cfg.get("nodata_value", -9999)
        if "nodata_value" not in predictors_cfg:
            log.warning(
                "'nodata_value' missing in config - defaulting to %s",
                tgt_nodata,
            )
        tgt_meta.update(
            {
                "driver": "GTiff",
                "crs": "EPSG:2154",
                "transform": rasterio.Affine(
                    tgt_res,
                    0,
                    91000.0,
                    0,
                    -tgt_res,
                    6124000.0 + 1000 * tgt_res,
                ),
                "height": 1000,
                "width": 1000,
                "nodata": tgt_nodata,
            }
        )
    log.info("Target grid established - %d x %d cells", tgt_meta["width"], tgt_meta["height"])

    mask_bool = None
    if cfg.get("boundary_mask"):
        mask_bool = _load_boundary_mask(Path(cfg["boundary_mask"]), tgt_meta)

    # 2. Iterate through groups and resample
    layers = {}
    if mask_bool is not None:
        layers["boundary_mask"] = xr.DataArray(mask_bool.astype("uint8"), dims=("y", "x"))

    n_workers = 32

    for group, spec in tqdm(predictors_cfg["layers"].items(), desc="groups"):
        categorical_hint = None
        if isinstance(spec, str):
            files = sorted(Path().glob(spec))
        elif isinstance(spec, list):
            files = [Path(f) for f in spec]
        elif isinstance(spec, dict):
            files = []
            categorical_hint = spec.get("categorical")

            if "pattern" in spec:
                pattern = spec["pattern"]
                fmt_dicts: list[dict] = []
                for y in spec.get("years", [None]):
                    for s in spec.get("seasons", [None]):
                        s_fmt = s.lower() if isinstance(s, str) else s
                        for m in spec.get("months", [None]):
                            m_fmt = f"{int(m):02d}" if isinstance(m, int) else str(m)
                            for idx in spec.get("indices", spec.get("classes", [None])):
                                lower = idx.lower() if isinstance(idx, str) else idx
                                fmt_dicts.append(
                                    {
                                        "year": y,
                                        "season": s_fmt,
                                        "season_lc": s_fmt,
                                        "month": m_fmt,
                                        "index": idx,
                                        "class": idx,
                                        "index_lc": lower,
                                        "class_lc": lower,
                                    }
                                )
                for fmt in fmt_dicts:
                    candidate_pat = pattern.format(**fmt)
                    matches = sorted(Path().glob(candidate_pat))
                    files.extend(matches)
            elif "files" in spec:
                files = [Path(f) for f in spec["files"]]
            else:
                raise KeyError(
                    f"Dict spec for {group} requires 'pattern' or 'files' key"
                )
        else:
            raise ValueError(f"Unsupported layer spec for {group}")

        log.info(
            "Resampling %s - %d raster(s) using %d worker(s)", group, len(files), n_workers
        )
        tasks = []
        for f in files:
            resampling = predictors_cfg["resampling_method"]["continuous"]
            if categorical_hint is True:
                resampling = predictors_cfg["resampling_method"]["categorical"]
            elif categorical_hint is None:
                stem = f.stem.lower()
                if (
                    "distance" not in stem
                    and any(
                        cat in stem
                        for cat in ["forest", "cover", "landcover", "class", "agri", "urban"]
                    )
                ):
                    resampling = predictors_cfg["resampling_method"]["categorical"]
            valid = (0.0, 1.0) if group == "landcover_fraction" else None
            tasks.append((f, tgt_meta, resampling, valid, mask_bool))

        with ProcessPoolExecutor(max_workers=n_workers) as exe:
            futures = [exe.submit(_resample_and_mask, t) for t in tasks]
            for fut in tqdm(as_completed(futures), total=len(futures), desc=group, leave=False):
                name, ds = fut.result()
                layers[name] = ds

    # 3. Stack & store
    transform = tgt_meta["transform"]
    width = tgt_meta["width"]
    height = tgt_meta["height"]
    x_coords = transform.c + transform.a * (np.arange(width) + 0.5)
    y_coords = transform.f + transform.e * (np.arange(height) + 0.5)
    stack = xr.Dataset(layers, coords={"y": y_coords, "x": x_coords})
    stack.to_zarr(stack_path, mode="w", consolidated=False)
    if use_cache and proc_path is not None and not proc_path.exists():
        shutil.copytree(stack_path, proc_path, dirs_exist_ok=True, copy_function=shutil.copy)
    log.info("Predictor stack written to %s", stack_path)
    print("    -> Stack saved", stack_path)
    return stack_path
