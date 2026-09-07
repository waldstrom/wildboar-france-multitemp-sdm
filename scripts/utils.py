# =============================================================================
# Project Title: Wild Boar Distribution Model for France
# Path: scripts/utils.py
# Purpose: Shared helper functions for file handling and configuration.
# Process Step: Provides reusable utilities for pipeline modules.
# Created by Adrian Meyer, Institute Geomatics, FHNW (adrian.meyer@fhnw.ch)
# Created: July 2025
# Version: v.0.1.0
# =============================================================================
"""
Utility helpers shared across modules.
"""
from __future__ import annotations
from pathlib import Path
from datetime import datetime
import yaml
import random
import numpy as np
import pandas as pd
import logging
import os
import tempfile
import zipfile
import shutil
import math
import warnings

log = logging.getLogger(__name__)

# -----------------------------------------------------------------------------  
def load_config(cfg_file: str | Path) -> dict:
    """Load YAML config and resolve simple Jinja-style references (${...})."""
    def _resolve_env(cfg: dict) -> dict:
        raw = yaml.safe_dump(cfg)
        for k, v in cfg.items():
            if isinstance(v, (str, int, float)):
                raw = raw.replace("${" + k + "}", str(v))
        return yaml.safe_load(raw)
    cfg = yaml.safe_load(Path(cfg_file).read_text())
    cfg = _resolve_env(cfg)
    log.info("Configuration loaded from %s", cfg_file)
    return cfg

# -----------------------------------------------------------------------------  
def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch  # type: ignore
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
    log.info("Global random seed set to %d", seed)

# -----------------------------------------------------------------------------  
def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p

# -----------------------------------------------------------------------------


def save_selected_predictors(stack_dir: str | Path, predictors: list[str]) -> Path:
    """Persist ``predictors`` next to ``stack_dir`` for later reuse."""

    stack_path = ensure_dir(stack_dir)
    out_file = stack_path / "selected_predictors.txt"
    text = "\n".join(predictors)
    if predictors:
        text += "\n"
    out_file.write_text(text, encoding="utf-8")
    return out_file


def load_selected_predictors(stack_dir: str | Path) -> list[str] | None:
    """Load predictor list saved via :func:`save_selected_predictors`."""

    stack_path = Path(stack_dir)
    in_file = stack_path / "selected_predictors.txt"
    if not in_file.exists():
        return None
    return [line.strip() for line in in_file.read_text(encoding="utf-8").splitlines() if line.strip()]

# -----------------------------------------------------------------------------

def timed_input(prompt: str, timeout: int = 10, default: str = "n") -> str:
    """Return user input or ``default`` after ``timeout`` seconds.

    The prompt is displayed and waits for input on ``stdin``. If no response is
    provided within ``timeout`` seconds the function returns ``default``.
    """
    import sys
    import select

    print(prompt, end="", flush=True)
    ready, _, _ = select.select([sys.stdin], [], [], timeout)
    if ready:
        return sys.stdin.readline().strip()
    print()  # move to next line after timeout
    return default

# -----------------------------------------------------------------------------

def sanitize_name(name: str) -> str:
    """Return ``name`` safe for use as a filename."""

    return name.replace("/", "_")

# -----------------------------------------------------------------------------

def is_debug_mode(cfg: dict | None = None) -> bool:
    """Return ``True`` if debug mode is enabled via config or environment."""

    if cfg and cfg.get("experiment", {}).get("debug_mode"):
        return True
    return os.environ.get("RUN_PIPELINE_DEBUG", "0") == "1"


def restrict_test_years(years: list[int], cfg: dict | None = None) -> list[int]:
    """Restrict ``years`` to a single debug year when debug mode is active."""

    if not years:
        return years
    if is_debug_mode(cfg):
        if 2021 in years:
            return [2021]
        return [years[0]]
    return years

# -----------------------------------------------------------------------------

def get_processed_dir() -> Path:
    """Return base directory for cached intermediate files."""

    base = Path(os.environ.get("PROCESSED_DATA_DIR", "processed_data"))
    base.mkdir(parents=True, exist_ok=True)
    return base

# -----------------------------------------------------------------------------

def load_mask(mask_file: str | Path) -> np.ndarray:
    """Load boundary mask as ``numpy`` array of booleans."""
    import rasterio

    with rasterio.open(mask_file) as src:
        data = src.read(1, masked=True)
    return data == 1


def filter_points_by_mask(df: pd.DataFrame, mask: np.ndarray) -> pd.DataFrame:
    """Keep only points falling inside mask (value 1)."""

    rows = ((6124000 + mask.shape[0] * 1000 - df["y"]) // 1000).astype(int)
    cols = ((df["x"] - 91000) // 1000).astype(int)
    valid = (
        (rows >= 0)
        & (rows < mask.shape[0])
        & (cols >= 0)
        & (cols < mask.shape[1])
        & (mask[rows, cols])
    )
    return df[valid].reset_index(drop=True)

# -----------------------------------------------------------------------------

def _grid_cell_indices(gdf):
    """Return ``(gdf, rows, cols)`` describing 1 km grid cell assignments."""

    if gdf.empty:
        return gdf.reset_index(drop=True), np.empty(0, dtype="int64"), np.empty(0, dtype="int64")

    if "x" not in gdf.columns:
        gdf = gdf.assign(x=gdf.geometry.x)
    if "y" not in gdf.columns:
        gdf = gdf.assign(y=gdf.geometry.y)

    rows = ((6124000 + 1000 * 1000 - gdf["y"]) // 1000).astype("int64")
    cols = ((gdf["x"] - 91000) // 1000).astype("int64")
    return gdf, rows.to_numpy(copy=False), cols.to_numpy(copy=False)


def deduplicate_cells(gdf):
    """Return at most one record per 1 km cell."""

    gdf, rows, cols = _grid_cell_indices(gdf)
    if rows.size == 0:
        return gdf

    gdf = gdf.assign(_r=rows, _c=cols)
    gdf = (
        gdf.drop_duplicates(subset=["_r", "_c"])  # type: ignore[arg-type]
        .drop(columns=["_r", "_c"])
        .reset_index(drop=True)
    )
    return gdf


def drop_background_overlaps(pres_gdf, bg_gdf, *, context: str = "background"):
    """Remove background points that fall inside presence grid cells."""

    pres_gdf, pres_rows, pres_cols = _grid_cell_indices(pres_gdf)
    bg_gdf, bg_rows, bg_cols = _grid_cell_indices(bg_gdf)

    if pres_rows.size == 0 or bg_rows.size == 0:
        return bg_gdf

    pres_cells = np.rec.fromarrays([pres_rows, pres_cols], names="r,c")
    bg_cells = np.rec.fromarrays([bg_rows, bg_cols], names="r,c")
    mask = ~np.isin(bg_cells, pres_cells)

    removed = int((~mask).sum())
    if removed:
        log.info(
            "Removed %d %s records overlapping presence cells", removed, context
        )

    return bg_gdf.loc[mask].reset_index(drop=True)

# -----------------------------------------------------------------------------

def count_model_params(model) -> int:
    """Return number of parameters for ``model`` for AIC/AICc.

    The function inspects common attribute names used by scikit-learn style
    estimators and elapid's MaxEnt implementation. If no known attributes are
    found, ``1`` is returned as a conservative fallback.
    """

    k = 0
    for attr in ["coef_", "feature_lambdas_", "lambdas_", "coefs_"]:
        vals = getattr(model, attr, None)
        if vals is not None:
            k += int(np.size(vals))
            break
    if hasattr(model, "intercept_"):
        k += int(np.size(getattr(model, "intercept_")))
    if k == 0:
        if hasattr(model, "estimator"):
            inner = getattr(model, "estimator")
            predictors = getattr(inner, "_predictors", None)
            if predictors is not None:
                total_nodes = 0
                try:
                    import numpy as _np

                    for tree in _np.asarray(predictors).ravel():
                        if hasattr(tree, "get_n_nodes"):
                            total_nodes += int(tree.get_n_nodes())
                except Exception:
                    total_nodes = 0
                if total_nodes > 0:
                    k = total_nodes
            if k == 0 and hasattr(inner, "n_features_in_"):
                k = int(getattr(inner, "n_features_in_"))
        elif hasattr(model, "n_features_in_"):
            k = int(getattr(model, "n_features_in_"))
    return k if k > 0 else 1

# -----------------------------------------------------------------------------


def _decompose_transform(transform) -> tuple[float, float, float, float, float, float]:
    """Return affine coefficients ``(a, b, c, d, e, f)`` for ``transform``."""

    if transform is None:
        raise ValueError("Raster transform is required for ASCII export")

    attrs = ("a", "b", "c", "d", "e", "f")
    if all(hasattr(transform, attr) for attr in attrs):
        return tuple(float(getattr(transform, attr)) for attr in attrs)

    # Fallback for GDAL-style tuples ``(c, a, b, f, d, e)``
    if isinstance(transform, (tuple, list)):
        if len(transform) >= 6:
            c, a, b, f, d, e = transform[:6]
            return (float(a), float(b), float(c), float(d), float(e), float(f))
        raise ValueError(
            "Affine transform tuples must contain at least six coefficients"
        )

    raise TypeError(
        f"Unsupported transform type for ASCII export: {type(transform)!r}"
    )


def _ascii_header(
    width: int, height: int, transform, nodata: float, *, atol: float = 1e-6
) -> list[str]:
    """Return ASCII header mirroring ``data/boundary_france.asc`` format."""

    a, b, c, d, e, f = _decompose_transform(transform)

    if not math.isclose(b, 0.0, abs_tol=atol) or not math.isclose(d, 0.0, abs_tol=atol):
        raise ValueError("Rotated rasters are not supported for ASCII export")

    pixel_width = math.hypot(a, b)
    pixel_height = math.hypot(d, e)
    if math.isclose(pixel_width, 0.0, abs_tol=atol):
        raise ValueError("Invalid transform with zero cellsize")

    if not math.isclose(pixel_width, pixel_height, abs_tol=atol):
        raise ValueError(
            f"ASCII export requires square pixels but received {pixel_width}x{pixel_height}"
        )

    cellsize = pixel_width

    # Determine the south-west corner from the affine transform
    x_candidates = [c, c + a * width]
    y_candidates = [f, f + e * height]
    xllcorner = min(x_candidates)
    yllcorner = min(y_candidates)

    nodata_str = f"{float(nodata):.15g}"

    return [
        f"ncols        {width}",
        f"nrows        {height}",
        f"xllcorner    {xllcorner:.15f}",
        f"yllcorner    {yllcorner:.15f}",
        f"cellsize     {cellsize:.15f}",
        f"NODATA_value  {nodata_str}",
    ]


def _write_ascii_raster(
    array: np.ndarray, transform, out_path: Path, nodata: float = -9999.0
) -> None:
    """Write ``array`` to ``out_path`` using ESRI ASCII grid format."""

    if array.ndim != 2:
        raise ValueError(
            f"ASCII export expects 2D arrays; received shape {array.shape}"
        )

    width = int(array.shape[1])
    height = int(array.shape[0])

    header = _ascii_header(width, height, transform, nodata)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    data = np.asarray(array, dtype="float32")
    invalid_mask = ~np.isfinite(data)
    if invalid_mask.any():
        data = data.copy()
        data[invalid_mask] = nodata

    with open(out_path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(header))
        fh.write("\n")
        np.savetxt(fh, data, fmt="%.6f", delimiter=" ")
        fh.write("\n")


def _ensure_consistent_grid(
    reference: dict | None,
    width: int,
    height: int,
    transform,
    *,
    atol: float = 1e-6,
) -> dict:
    """Validate that all rasters share the same grid definition."""

    a, b, c, d, e, f = _decompose_transform(transform)

    if reference is None:
        return {
            "width": int(width),
            "height": int(height),
            "a": a,
            "b": b,
            "c": c,
            "d": d,
            "e": e,
            "f": f,
            "transform": transform,
        }

    if int(width) != reference["width"] or int(height) != reference["height"]:
        raise ValueError(
            "Raster dimensions do not match the reference grid: "
            f"expected {reference['width']}x{reference['height']} but got {width}x{height}"
        )

    for key, value in (("a", a), ("b", b), ("c", c), ("d", d), ("e", e), ("f", f)):
        if not math.isclose(value, reference[key], abs_tol=atol):
            raise ValueError(
                "Raster georeferencing mismatch detected for coefficient "
                f"'{key}': expected {reference[key]} but got {value}"
            )

    return reference


def export_java_compatible_bundle(
    cfg: dict,
    season: str,
    test_year: int,
    stack_path: Path,
    vars_used: list[str],
    pres_train,
    bg_train,
    input_dir: Path,
    bias: "xr.DataArray | None" = None,
) -> Path:
    """Export predictors and samples in a format compatible with Java MaxEnt."""

    import rasterio

    if not stack_path.exists():
        raise FileNotFoundError(f"Predictor stack not found at {stack_path}")

    exp_cfg = cfg.setdefault("experiment", {})
    timestamp = exp_cfg.get("timestamp")
    if not timestamp:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        exp_cfg["timestamp"] = timestamp
    exp_name = exp_cfg.get("name", "experiment")
    season_safe = sanitize_name(season.lower())
    species_label = sanitize_name(f"{exp_name}_{timestamp}_test-{test_year}")

    tmp_root = Path(tempfile.mkdtemp(prefix="java-bundle-"))
    try:
        bundle_dir = tmp_root
        features_dir = bundle_dir / "features"
        features_dir.mkdir(parents=True, exist_ok=True)

        grid_meta: dict | None = None
        for var in vars_used:
            tif = stack_path / f"{var}.tif"
            if not tif.exists():
                raise FileNotFoundError(f"Predictor raster missing: {tif}")
            with rasterio.open(tif) as src:
                data = src.read().astype("float32")
                nodata_val = src.nodata
                if nodata_val is not None:
                    data[data == nodata_val] = np.nan
                if data.ndim == 3 and data.shape[0] > 1:
                    with np.errstate(invalid="ignore"):
                        data = np.nanmean(data, axis=0)
                else:
                    data = data[0] if data.ndim == 3 else data
                grid_meta = _ensure_consistent_grid(
                    grid_meta, src.width, src.height, src.transform
                )
                _write_ascii_raster(
                    data,
                    grid_meta["transform"],
                    features_dir / f"{var}.asc",
                )

        if grid_meta is None:
            raise RuntimeError("Failed to derive grid metadata for ASCII export")

        bias_written = False
        if bias is not None:
            bias_arr = np.asarray(bias.values, dtype="float32")
            height = grid_meta["height"]
            width = grid_meta["width"]
            if bias_arr.shape != (height, width):
                if bias_arr.size == height * width:
                    bias_arr = bias_arr.reshape(height, width)
                else:
                    raise ValueError(
                        "Bias raster dimensions do not match predictor grid"
                    )
            _write_ascii_raster(
                bias_arr, grid_meta["transform"], bundle_dir / "biasgrid.asc"
            )
            bias_written = True
        else:
            bias_tif = input_dir / "bias.tif"
            if bias_tif.exists():
                with rasterio.open(bias_tif) as src:
                    data = src.read(1).astype("float32")
                    nodata_val = src.nodata
                    if nodata_val is not None:
                        data[data == nodata_val] = np.nan
                    grid_meta = _ensure_consistent_grid(
                        grid_meta, src.width, src.height, src.transform
                    )
                    _write_ascii_raster(
                        data,
                        grid_meta["transform"],
                        bundle_dir / "biasgrid.asc",
                    )
                    bias_written = True

        pres_df = pd.DataFrame(
            {
                "species": [species_label] * len(pres_train),
                "longitude": pres_train.geometry.x,
                "latitude": pres_train.geometry.y,
            }
        )
        pres_df.to_csv(bundle_dir / "points.csv", index=False, float_format="%.6f")

        bg_df = pd.DataFrame(
            {
                "longitude": bg_train.geometry.x,
                "latitude": bg_train.geometry.y,
            }
        )
        bg_df.to_csv(
            bundle_dir / "background_points.csv", index=False, float_format="%.6f"
        )

        metadata = {
            "export_generated_at": datetime.utcnow().isoformat() + "Z",
            "season": season,
            "test_year": int(test_year),
            "experiment_name": exp_name,
            "experiment_timestamp": timestamp,
            "species_label": species_label,
            "n_presence_points": int(len(pres_train)),
            "n_background_points": int(len(bg_train)),
            "predictors": vars_used,
            "bias_included": bias_written,
            "maxent_parameters": cfg.get("maxent", {}),
        }
        (bundle_dir / "metadata.txt").write_text(
            yaml.safe_dump(metadata, sort_keys=False), encoding="utf-8"
        )

        zip_path = input_dir / f"java_input_{season_safe}_test-{test_year}.zip"
        if zip_path.exists():
            zip_path.unlink()
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for path in bundle_dir.rglob("*"):
                if path.is_file():
                    zf.write(path, arcname=path.relative_to(bundle_dir))

        log.info("Java-compatible bundle written to %s", zip_path)
        return zip_path
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)


# -----------------------------------------------------------------------------

def extract_from_stack(stack: Path | xr.Dataset, gdf, vars_used: list[str]) -> np.ndarray:
    """Extract raster values for points from a predictor stack.

    ``stack`` may either be the path to a stack on disk or an in-memory
    :class:`xarray.Dataset`.  Supporting datasets directly simplifies reuse of
    temporary stacks where writing to disk would be unnecessary overhead.

    Missing values are retained in the output array so that the caller can
    decide how to handle them. Any points falling outside the raster bounds are
    returned as ``NaN`` rows. The function derives the grid's origin and
    resolution from the stack itself which allows working with stacks that have
    been spatially shifted. For GeoTIFF based stacks the function supports
    multi-band rasters, collapsing them to a single layer by averaging across
    bands while ignoring ``NaN`` values.
    """
    import xarray as xr
    from zarr.errors import GroupNotFoundError
    import rasterio

    if any(v in {"x", "y"} for v in vars_used):
        raise ValueError("Coordinate predictors 'x' and 'y' are not supported")
    raster_vars = list(vars_used)

    if isinstance(stack, xr.Dataset):
        # Directly sample from the provided in-memory dataset
        ds = stack
        if raster_vars:
            ds_vars = ds[raster_vars]
            data = (
                np.column_stack([ds_vars[v].values.ravel() for v in raster_vars])
                .astype("float32")
            )
            data[data == -9999] = np.nan
        else:
            data = np.empty((ds.sizes["x"] * ds.sizes["y"], 0), dtype="float32")
        height = ds.sizes["y"]
        width = ds.sizes["x"]

        x_coords = ds.x.values
        y_coords = ds.y.values
        if len(x_coords) < 2 or len(y_coords) < 2:
            raise ValueError("Predictor stack must have at least two cells per axis")
        x0 = float(x_coords.min())
        y_max = float(y_coords.max())
        res_x = float(abs(x_coords[1] - x_coords[0]))
        res_y = float(abs(y_coords[1] - y_coords[0]))
    else:
        stack_path = Path(stack)
        if stack_path.is_dir() and any(stack_path.glob("*.tif")):
            # Geotiff based stack
            arrays = []
            meta = None
            for var in raster_vars:
                tif = stack_path / f"{var}.tif"
                with rasterio.open(tif) as src:
                    arr = src.read().astype("float32")
                    nodata = src.nodata if src.nodata is not None else -9999
                    arr[arr == nodata] = np.nan
                    if arr.shape[0] > 1:
                        if np.isnan(arr).all():
                            arr = arr[0]
                        else:
                            # ``np.nanmean`` emits a RuntimeWarning when all values
                            # along an axis are ``NaN``. This can happen for sparse
                            # rasters where some pixels contain no valid data across
                            # bands. Suppress this warning as such pixels will be set
                            # to ``NaN`` in the output anyway.
                            with warnings.catch_warnings():
                                warnings.filterwarnings(
                                    "ignore",
                                    message="Mean of empty slice",
                                    category=RuntimeWarning,
                                )
                                arr = np.nanmean(arr, axis=0)
                    else:
                        arr = arr[0]
                    arrays.append(arr.ravel())
                    if meta is None:
                        meta = {
                            "width": src.width,
                            "height": src.height,
                            "transform": src.transform,
                        }
            if meta is None:
                # No raster variables were requested, but we still need grid
                # metadata.  Grab it from any available layer.
                any_tif = next(stack_path.glob("*.tif"), None)
                if any_tif is None:
                    raise FileNotFoundError(f"No GeoTIFF layers found in {stack_path}")
                with rasterio.open(any_tif) as src:
                    meta = {
                        "width": src.width,
                        "height": src.height,
                        "transform": src.transform,
                    }
            data = (
                np.column_stack(arrays) if arrays else np.empty((meta["width"] * meta["height"], 0), dtype="float32")
            )
            height = meta["height"]
            width = meta["width"]
            transform = meta["transform"]
            x0 = float(transform.c)
            y_max = float(transform.f)
            res_x = float(transform.a)
            res_y = float(-transform.e)
            if res_x < 10 and res_y < 10:
                log.info("Scaling GeoTIFF coordinates by 1000 to convert km to m")
                x0 *= 1000
                y_max *= 1000
                res_x *= 1000
                res_y *= 1000
        else:
            # Zarr based stack (default behaviour)
            # Explicitly disable consolidated metadata lookup to avoid warnings
            try:
                ds = xr.open_zarr(stack_path, consolidated=False)
            except GroupNotFoundError as exc:
                raise FileNotFoundError(
                    f"Predictor stack not found at {stack_path}"
                ) from exc
            if raster_vars:
                ds_vars = ds[raster_vars]
                data = (
                    np.column_stack([ds_vars[v].values.ravel() for v in raster_vars])
                    .astype("float32")
                )
                data[data == -9999] = np.nan
            else:
                data = np.empty((ds.sizes["x"] * ds.sizes["y"], 0), dtype="float32")
            height = ds.sizes["y"]
            width = ds.sizes["x"]

            # Derive grid geometry dynamically from coordinates instead of assuming
            # fixed offsets.  This allows sampling from stacks that have been shifted
            # in space (e.g. the multitemporal 3x2 matrix) where the
            # minimum/maximum coordinates no longer match the national grid origin.
            x_coords = ds.x.values
            y_coords = ds.y.values
            if len(x_coords) < 2 or len(y_coords) < 2:
                raise ValueError("Predictor stack must have at least two cells per axis")
            x0 = float(x_coords.min())
            y_max = float(y_coords.max())
            res_x = float(abs(x_coords[1] - x_coords[0]))
            res_y = float(abs(y_coords[1] - y_coords[0]))

    if "x" not in gdf.columns:
        gdf["x"] = gdf.geometry.x
    if "y" not in gdf.columns:
        gdf["y"] = gdf.geometry.y

    # Drop records with missing coordinates to avoid indexing errors
    gdf["x"] = pd.to_numeric(gdf["x"], errors="coerce")
    gdf["y"] = pd.to_numeric(gdf["y"], errors="coerce")
    gdf = gdf.dropna(subset=["x", "y"]).reset_index(drop=True)

    rows = ((y_max - gdf["y"]) / res_y).astype(int)
    cols = ((gdf["x"] - x0) / res_x).astype(int)
    valid = (rows >= 0) & (rows < height) & (cols >= 0) & (cols < width)
    idx = rows * width + cols

    # Sample raster-based variables
    sample = np.full((len(gdf), len(raster_vars)), np.nan, dtype=data.dtype)
    if raster_vars:
        sample[valid] = data[idx[valid]]
    return sample

# -----------------------------------------------------------------------------

def validate_crs(crs, default: str = "EPSG:2154", context: str = "dataset"):
    """Return a CRS object.

    When the provided ``crs`` is missing or invalid, the ``default`` value is
    used instead. Warnings are only emitted when the fallback is not
    ``EPSG:2154`` which helps reduce noise when working with the project
    standard.
    """
    import rasterio
    default_crs = rasterio.crs.CRS.from_string(default)
    if crs is None:
        if default.lower() == "epsg:2154":
            log.info("%s missing CRS. Assuming %s", context, default)
        else:
            log.warning("%s missing CRS. Assuming %s", context, default)
        return default_crs
    try:
        crs_obj = rasterio.crs.CRS.from_user_input(crs)
    except Exception:
        if default.lower() == "epsg:2154":
            log.info("%s invalid CRS %s. Assuming %s", context, crs, default)
        else:
            log.warning("%s invalid CRS %s. Assuming %s", context, crs, default)
        return default_crs
    if crs_obj != default_crs:
        if default.lower() == "epsg:2154":
            log.info("%s CRS %s differs from expected %s", context, crs_obj, default)
        else:
            log.warning("%s CRS %s differs from expected %s", context, crs_obj, default)
    return crs_obj


def ensure_gdf_crs(gdf, default: str = "EPSG:2154", context: str = "vector"):
    """Ensure a GeoDataFrame uses the expected CRS.

    Missing or unexpected CRS definitions trigger a reprojection. Similar to
    :func:`validate_crs`, warnings are suppressed when the target CRS is the
    project default of ``EPSG:2154``.
    """
    import geopandas as gpd
    default_crs = default
    if gdf.crs is None:
        if default_crs.lower() == "epsg:2154":
            log.info("%s missing CRS. Assuming %s", context, default_crs)
        else:
            log.warning("%s missing CRS. Assuming %s", context, default_crs)
        gdf = gdf.set_crs(default_crs)
    elif gdf.crs.to_string() != default_crs:
        if default_crs.lower() == "epsg:2154":
            log.info("%s CRS %s differs from expected %s", context, gdf.crs, default_crs)
        else:
            log.warning("%s CRS %s differs from expected %s", context, gdf.crs, default_crs)
        gdf = gdf.to_crs(default_crs)
    return gdf
