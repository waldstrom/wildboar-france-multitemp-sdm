# =============================================================================
# Project Title: Wild Boar Distribution Model for France
# Path: scripts/multitemporal.py
# Purpose: Train seasonal MaxEnt models using multitemporal 3x2 data matrices.
# Process Step: Combines six seasonal stacks offset by ±1000km and evaluates
#               the seventh year. Performs temporal iteration so that each
#               season/year combination acts once as the test fold. Includes
#               spatial cross-validation and permutation based feature
#               importance. Produces annotated visualisations for every
#               training/test split and summarises evaluation metrics with
#               confidence intervals.
# Created by Adrian Meyer, Institute Geomatics, FHNW (adrian.meyer@fhnw.ch)
# Created: July 2025
# Version: v.0.1.0
# =============================================================================
"""Multitemporal training utility.

The script stacks seasonal predictor rasters from different years in a
3×2 spatial matrix. Presence and background points are shifted by
±1000 km (EPSG:2154) so that six copies of metropolitan France can be
presented to the model simultaneously. The remaining seventh year acts as
an independent test set. Iterating across all available years yields a
leave-one-year-out temporal cross-validation. A final model can be trained
on the six most recent years to provide a season specific prediction model.

Running the script requires a dedicated configuration file which defines
paths, seasons and optional switches such as whether the hunting bag
predictor should be included in the analysis.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import logging
import math
import copy
import shutil
import os
import re
import glob
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

import numpy as np
import pandas as pd
import geopandas as gpd
import warnings
import xarray as xr
from shapely.affinity import translate
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import rasterio
from rasterio import Affine
from statsmodels.stats.outliers_influence import variance_inflation_factor
import yaml
from tqdm.auto import tqdm
import pickle

warnings.filterwarnings("once", category=pd.errors.PerformanceWarning)

from scripts.utils import (
    load_config,
    ensure_dir,
    load_mask,
    extract_from_stack,
    deduplicate_cells,
    drop_background_overlaps,
    sanitize_name,
    count_model_params,
    timed_input,
    filter_points_by_mask,
    export_java_compatible_bundle,
    restrict_test_years,
    save_selected_predictors,
    load_selected_predictors,
)
from scripts.maxent_training import train_model
from scripts.evaluation import _compute_metrics, evaluate
from scripts.cross_validation import run_cv
from scripts.variable_importance import permutation_importance
from scripts.thinning import mahalanobis_thin, grid_thin
from scripts.multitemp_preprocessing import export_geotiffs
from scripts.visualization import (
    plot_cloglog_points,
    export_response_curves,
    compile_response_curves_pdf,
)

log = logging.getLogger(__name__)


_RUN_ID_CACHE: dict[tuple[str, str, str], str] = {}


def log_heading(title: str) -> None:
    """Log a section heading for easier debugging."""
    log.debug("%s\n%s\n%s", "=" * 60, title, "=" * 60)


def _load_shift_stack(args: tuple[dict, str, YearOffset, bool]) -> xr.Dataset:
    cfg, season, yo, rebuild = args
    ds, _ = _load_stack(cfg, season, yo.year, rebuild, validate=rebuild)
    return ds.assign_coords(x=ds.x + yo.dx, y=ds.y + yo.dy)

# Predictor rasters for all years are stacked once per season and reused
# across all test/train splits. This avoids repeatedly rebuilding identical
# stacks for each test-year combination.


# -----------------------------------------------------------------------------
@dataclass
class YearOffset:
    """Helper dataclass linking a training year with its spatial offset."""

    year: int
    dx: float
    dy: float


# -----------------------------------------------------------------------------
def _experiment_dir(cfg: dict) -> Path:
    """Return root directory for the current experiment.

    Historically experiment directories were created below ``exps/`` with the
    fixed prefix ``multitemp-``.  For greater flexibility we honour two optional
    configuration keys: ``experiment.dir_prefix`` controls the directory prefix
    (default ``multitemp``) and ``experiment.base_dir`` selects the parent
    directory (default ``exps``).  ``experiment.name`` continues to hold the
    experiment identifier while ``experiment.append_timestamp`` (default
    ``True`` for user supplied names) toggles whether an additional run specific
    timestamp is appended.  The timestamp ensures that consecutive runs started
    from the command line write to unique directories whereas programmatic
    callers that reuse the same configuration dictionary within a Python process
    share the same output folder.
    """

    exp_cfg = cfg.setdefault("experiment", {})
    exp_name = exp_cfg.get("name")
    auto_name = False
    if not exp_name:
        exp_name = datetime.now().strftime("%Y%m%d-%H%M%S")
        exp_cfg["name"] = exp_name
        auto_name = True

    base = Path(exp_cfg.get("base_dir", "exps"))
    base_key = os.path.abspath(base)
    prefix = exp_cfg.get("dir_prefix", "multitemp")

    timestamp = exp_cfg.get("timestamp")
    cache_key = (base_key, prefix, exp_name)
    if not timestamp:
        timestamp = _RUN_ID_CACHE.get(cache_key)
        if not timestamp:
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            _RUN_ID_CACHE[cache_key] = timestamp
        exp_cfg["timestamp"] = timestamp
    else:
        _RUN_ID_CACHE[cache_key] = timestamp

    append_ts = exp_cfg.get("append_timestamp")
    if append_ts is None:
        append_ts = not auto_name

    dir_name = f"{prefix}-{exp_name}"
    if append_ts and timestamp:
        dir_name = f"{dir_name}-{timestamp}"

    return base / dir_name


# -----------------------------------------------------------------------------
def _last_experiment_dir(cfg: dict) -> Path | None:
    """Return most recently modified previous experiment directory, if any."""

    current = _experiment_dir(cfg)
    root = current.parent
    exp_cfg = cfg.get("experiment", {})
    prefix = exp_cfg.get("dir_prefix", "multitemp")
    pattern = f"{prefix}-*"
    candidates = [d for d in root.glob(pattern) if d.is_dir() and d != current]
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


# -----------------------------------------------------------------------------
def _resolve_engines(cfg: dict) -> list[tuple[str, str]]:
    """Return ``[(label, trainer)]`` for engines configured in ``cfg``.

    The first element of each tuple represents the canonical engine label used
    for directory names and reporting (``"elapid"`` for MaxEnt, ``"gbm"`` for
    the gradient boosting model).  The second element is the identifier passed
    to the respective training routine.
    """

    raw = cfg.get("model", {}).get("engines")
    if not raw:
        return [("elapid", "maxent")]

    resolved: list[tuple[str, str]] = []
    seen: set[str] = set()
    for item in raw:
        key = str(item).lower()
        if key in {"elapid", "maxent"}:
            label, trainer = "elapid", "maxent"
        elif key == "gbm":
            label, trainer = "gbm", "gbm"
        else:
            raise ValueError(f"Unsupported modelling engine '{item}'")
        if label not in seen:
            resolved.append((label, trainer))
            seen.add(label)
    priority = {"gbm": 0, "elapid": 1}
    resolved.sort(key=lambda pair: priority.get(pair[0], len(priority)))
    return resolved


# -----------------------------------------------------------------------------
def _build_matrix(years: list[int], test_year: int, shift: float) -> list[YearOffset]:
    """Return ordered mapping of training years to spatial offsets."""

    offsets = [
        (0, 0),
        (1 * shift, 0),
        (2 * shift, 0),
        (0, -1 * shift),
        (1 * shift, -1 * shift),
        (2 * shift, -1 * shift),
    ]
    train_years = [y for y in years if y != test_year]
    train_years.sort()
    offsets = offsets[: len(train_years)]
    return [
        YearOffset(year=y, dx=dx, dy=dy) for y, (dx, dy) in zip(train_years, offsets)
    ]


# -----------------------------------------------------------------------------
def _expected_var_names(cfg: dict, year: int, season: str) -> set[str]:
    """Return expected predictor variable names for ``year`` and ``season``."""

    base_cfg = copy.deepcopy(cfg)
    base_cfg.pop("multitemporal", None)
    layers_cfg = base_cfg.get("predictors", {}).get("layers", {})
    for layer in layers_cfg.values():
        if isinstance(layer, dict):
            if "years" in layer:
                layer["years"] = [year]
            if "seasons" in layer:
                layer["seasons"] = [season]

    expected: set[str] = set()
    for spec in layers_cfg.values():
        files: list[Path] = []
        if isinstance(spec, str):
            files = [Path(f) for f in Path().glob(spec)]
        elif isinstance(spec, list):
            files = [Path(f) for f in spec]
        elif isinstance(spec, dict):
            if "files" in spec:
                files.extend(Path(f) for f in spec["files"])
            pattern = spec.get("pattern")
            if pattern:
                fmt_dicts: list[dict] = []
                for y in spec.get("years", [None]):
                    for s in spec.get("seasons", [None]):
                        for m in spec.get("months", [None]):
                            m_fmt = f"{int(m):02d}" if isinstance(m, int) else m
                            for idx in spec.get("indices", spec.get("classes", [None])):
                                lower = idx.lower() if isinstance(idx, str) else idx
                                fmt_dicts.append(
                                    {
                                        "year": y,
                                        "season": s,
                                        "season_lc": s,
                                        "month": m_fmt,
                                        "index": idx,
                                        "class": idx,
                                        "index_lc": lower,
                                        "class_lc": lower,
                                    }
                                )
                for fmt in fmt_dicts:
                    candidate = pattern.format(
                        **{k: v for k, v in fmt.items() if v is not None}
                    )
                    matches = [Path(p) for p in glob.glob(candidate)]
                    files.extend(matches)
        for f in files:
            if f.exists():
                expected.add(f.stem)
    return expected


# -----------------------------------------------------------------------------
def _load_presence(cfg: dict, season: str, year: int) -> gpd.GeoDataFrame:
    """Load presence points for a given season and year."""

    df = pd.read_csv(cfg["presence_data"]["file"], encoding="utf-8-sig")
    df.columns = (
        df.columns.str.strip().str.lower().str.replace(r"[\s-]+", "_", regex=True)
    )
    if "year" not in df.columns and "year_int" in df.columns:
        df = df.rename(columns={"year_int": "year"})
    year_col = cfg["presence_data"].get("year_column", "year").replace("-", "_")
    season_col = cfg["presence_data"].get("season_column", "season").replace("-", "_")
    lon_col = cfg["presence_data"].get("lon_column", "dd_long").replace("-", "_")
    lat_col = cfg["presence_data"].get("lat_column", "dd_lat").replace("-", "_")
    df[lon_col] = pd.to_numeric(df[lon_col], errors="coerce")
    df[lat_col] = pd.to_numeric(df[lat_col], errors="coerce")
    df = df.dropna(subset=[lon_col, lat_col]).reset_index(drop=True)
    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df[lon_col], df[lat_col]),
        crs=cfg["presence_data"]["crs"],
    )
    mask = (gdf[year_col] == year) & (gdf[season_col].str.lower() == season.lower())
    gdf = gdf.loc[mask].reset_index(drop=True)
    gdf["x"] = gdf.geometry.x
    gdf["y"] = gdf.geometry.y
    boundary = load_mask(cfg["boundary_mask"])
    df_valid = filter_points_by_mask(gdf.drop(columns="geometry"), boundary)
    gdf = gpd.GeoDataFrame(
        df_valid,
        geometry=gpd.points_from_xy(df_valid["x"], df_valid["y"]),
        crs=cfg["presence_data"]["crs"],
    )
    gdf["year"] = year
    gdf["season"] = season
    return gdf


# -----------------------------------------------------------------------------
def _sample_background(cfg: dict, n: int) -> gpd.GeoDataFrame:
    """Sample ``n`` background points within the study mask."""

    mask = load_mask(cfg["boundary_mask"])
    valid_rc = np.argwhere(mask)
    choice_idx = np.random.choice(len(valid_rc), size=n, replace=False)
    rows, cols = valid_rc[choice_idx].T
    xs = 91000 + (cols + np.random.rand(n)) * 1000
    ys = 6124000 + 1000 * 1000 - (rows + np.random.rand(n)) * 1000
    gdf = gpd.GeoDataFrame(
        pd.DataFrame({"bg_id": np.arange(n)}),
        geometry=gpd.points_from_xy(xs, ys),
        crs="EPSG:2154",
    )
    gdf["x"] = gdf.geometry.x
    gdf["y"] = gdf.geometry.y
    gdf["source"] = "background"
    return gdf


# -----------------------------------------------------------------------------
def _france_outline_from_mask(mask_file: str | Path) -> gpd.GeoSeries:
    """Convert boundary mask raster to a polygon outline."""

    import rasterio
    from rasterio import features
    from shapely.geometry import shape

    with rasterio.open(mask_file) as src:
        data = src.read(1)
        transform = src.transform
        crs = src.crs

    mask = data == 1
    polygons = [
        shape(geom)
        for geom, val in features.shapes(data, mask=mask, transform=transform)
        if val == 1
    ]
    outline = gpd.GeoSeries(polygons, crs=crs).unary_union
    return gpd.GeoSeries([outline], crs=crs)


# -----------------------------------------------------------------------------
def _sample_background_polygon(poly: gpd.GeoSeries, n: int) -> gpd.GeoDataFrame:
    """Sample ``n`` random background points within ``poly``."""

    from shapely.geometry import Point

    minx, miny, maxx, maxy = poly.total_bounds
    pts: list[tuple[float, float]] = []
    while len(pts) < n:
        x = np.random.uniform(minx, maxx)
        y = np.random.uniform(miny, maxy)
        if poly.contains(Point(x, y)).iloc[0]:
            pts.append((x, y))
    gdf = gpd.GeoDataFrame(
        pd.DataFrame({"bg_id": np.arange(n), "source": "background"}),
        geometry=gpd.points_from_xy(*zip(*pts)),
        crs="EPSG:2154",
    )
    gdf["x"] = gdf.geometry.x
    gdf["y"] = gdf.geometry.y
    return gdf


# -----------------------------------------------------------------------------
def _plot_outline(ax, geom, **kwargs) -> None:
    """Plot polygon or multipolygon outline on ``ax``."""

    if getattr(geom, "geom_type", "") == "Polygon":
        ax.plot(*geom.exterior.xy, **kwargs)
    else:
        for part in getattr(geom, "geoms", [geom]):
            ax.plot(*part.exterior.xy, **kwargs)


# -----------------------------------------------------------------------------
SATELLITE_INDEX_TOKENS = {
    "ndvi",
    "ndwi",
    "nbr",
    "ndbi",
    "ndmi",
    "ndsi",
    "evi",
    "evi2",
    "bsi",
    "vssi",
    "msavi",
    "savi",
}


def _is_satellite_index_suffix(name: str) -> bool:
    """Return ``True`` when ``name`` refers to a satellite index variable."""

    tokens = [tok for tok in re.split(r"[_\W]+", name.lower()) if tok]
    return any(tok in SATELLITE_INDEX_TOKENS for tok in tokens)


def _apply_feature_config(
    ds: xr.Dataset, cfg: dict, *, return_dropped: bool = False
) -> xr.Dataset | tuple[xr.Dataset, list[str]]:
    """Drop predictors marked as ``drop`` in feature configuration."""
    mt_cfg = cfg.get("multitemporal", {})
    if mt_cfg.get("ignore_featureconfig_drop", False):
        return ds

    fc_path = (
        mt_cfg.get("vif", {}).get("feature_config", "featureconfig.yaml")
    )
    if Path(fc_path).exists():
        try:
            feature_cfg = yaml.safe_load(Path(fc_path).read_text())
            drop = [
                k
                for k, v in feature_cfg.get("variables", {}).items()
                if str(v).lower() == "drop"
            ]
            drop = [v for v in drop if v in ds.data_vars]
            if drop:
                ds = ds.drop_vars(drop)
        except Exception as exc:  # pragma: no cover - config errors should not crash
            log.warning("Failed reading feature config %s: %s", fc_path, exc)
    if return_dropped:
        return ds, drop
    return ds


def _fill_missing_satellite_layers(
    ds_year: xr.Dataset,
    ds_master: xr.Dataset,
    dropped: list[str],
    year: int,
) -> xr.Dataset:
    """Replace dropped satellite layers with the average of remaining years."""

    if not dropped:
        return ds_year

    filled = 0
    prefix = f"{year}_"
    for name in dropped:
        if not name.startswith(prefix):
            continue
        suffix = name[len(prefix) :]
        if not suffix or not _is_satellite_index_suffix(suffix):
            continue
        pattern = re.compile(rf"\d{{4}}_{re.escape(suffix)}$")
        candidates = [
            ds_master[var]
            for var in ds_master.data_vars
            if pattern.match(var) and var != name
        ]
        if not candidates:
            continue
        stacked = xr.concat(candidates, dim="year_avg")
        mean_da = stacked.mean(dim="year_avg", skipna=True)
        mean_da.attrs = dict(candidates[0].attrs)
        ds_year[name] = mean_da
        filled += 1
    if filled:
        log.info(
            "Filled %d dropped satellite layer(s) for %s using remaining years",
            filled,
            year,
        )
    return ds_year


# -----------------------------------------------------------------------------
def fuse_yearly_layers(ds: xr.Dataset) -> xr.Dataset:
    """Combine variables with year tokens into single composite layers.

    Rasters that vary by year are often named using a ``<year>_`` prefix such
    as ``2018_CONIFERS``.  Other datasets encode the year elsewhere, e.g.
    ``tmax_2018_summer``.  When multiple years are shifted into the 3×2
    multitemporal matrix these layers should be fused so that subsequent
    processing operates on a single composite layer per variable.

    Parameters
    ----------
    ds:
        Dataset containing predictor layers.  Variables whose names begin with a
        four digit year followed by an underscore are grouped by the common
        suffix and merged using :func:`xarray.combine_by_coords`.

    Returns
    -------
    xarray.Dataset
        Dataset where yearly layers have been merged into composites and static
        layers remain unchanged.
    """

    fused: dict[str, xr.DataArray] = {}
    for name, da in ds.data_vars.items():
        match = re.match(r"(.+?)_(\d{4})_(.+)", name)
        if match:
            base = f"{match.group(1)}_{match.group(3)}"
            da_base = da.rename(base)
            if base in fused:
                fused[base] = fused[base].combine_first(da_base)
                fused[base].attrs.update(da_base.attrs)
            else:
                fused[base] = da_base
            continue
        match = re.match(r"^(\d{4})_(.+)$", name)
        if match:
            base = match.group(2)
            da_base = da.rename(base)
            if base in fused:
                fused[base] = fused[base].combine_first(da_base)
                fused[base].attrs.update(da_base.attrs)
            else:
                fused[base] = da_base
            continue
        match = re.match(r"(.+)_(\d{4})$", name)
        if match:
            base = match.group(1)
            da_base = da.rename(base)
            if base in fused:
                fused[base] = fused[base].combine_first(da_base)
                fused[base].attrs.update(da_base.attrs)
            else:
                fused[base] = da_base
        else:
            fused[name] = da
    return xr.Dataset(fused, coords=ds.coords)


# -----------------------------------------------------------------------------
def _vif_filter(ds: xr.Dataset, cfg: dict) -> tuple[xr.Dataset, list[str]]:
    """Keep least collinear satellite and bioclim variables."""

    params = cfg.get("multitemporal", {}).get("vif", {})
    if not params.get("enable", False):
        return ds, [v for v in ds.data_vars if v != "boundary_mask"]

    sample_frac = params.get("sample_fraction", 0.05)
    rng = np.random.default_rng(cfg["experiment"].get("random_seed", 0))

    # Ensure coordinate placeholders are not considered for VIF calculation
    ds_vars = ds.drop_vars(["boundary_mask", "x", "y"], errors="ignore")
    flat = ds_vars.stack(sample=("y", "x"))
    n_pixels = flat.sizes["sample"]
    min_samples = min(n_pixels, len(ds_vars.data_vars) + 1)
    n_samples = min(n_pixels, max(int(n_pixels * sample_frac), min_samples))
    log.info(
        "Sampling %d/%d pixels (%.1f%%) for VIF calculation",
        n_samples,
        n_pixels,
        n_samples / n_pixels * 100,
    )
    idx = rng.choice(n_pixels, size=n_samples, replace=False)
    t0 = time.perf_counter()
    df = (
        flat.isel(sample=idx)
        .to_dataframe()
        .reset_index(drop=True)
        .dropna()
    )
    log.debug(
        "Sample conversion to DataFrame completed in %.2fs (%d rows)",
        time.perf_counter() - t0,
        len(df),
    )
    # Exclude ERA5 anomaly variables
    all_vars = [v for v in df.columns if "anom" not in v.lower()]
    if df.empty or len(all_vars) < 2:
        log.warning(
            "Skipping VIF filter due to insufficient data (samples=%d, vars=%d)",
            len(df),
            len(all_vars),
        )
        return ds, [v for v in ds.data_vars if v != "boundary_mask"]
    values = df[all_vars].values
    vif_vals = {}
    for i, var in enumerate(tqdm(all_vars, desc="VIF", leave=False)):
        vif_vals[var] = variance_inflation_factor(values, i)

    sat_vars = [
        v
        for v in all_vars
        if any(
            s in v.lower()
            for s in [
                "ndvi",
                "evi",
                "nbr",
                "ndwi",
                "ndbi",
                "ndmi",
                "ndsi",
                "bsi",
                "vssi",
            ]
        )
    ]
    # Exclude bio09 and bio08 variables due to artefacts
    bio_vars = [
        v
        for v in all_vars
        if v.lower().startswith("bio")
        and not v.lower().startswith(("bio09", "bio08"))
    ]
    era_prefixes = [
        "2m_temperature",
        "soil_temperature",
        "snow_depth",
        "precip",
        "pet",
        "skin_t",
        "soil_moist",
        "ssrd",
        "surface_solar_radiation",
        "volumetric_soil_water_layer",
        "vpd",
        "tmax",
        "tmean",
    ]
    era_vars = [
        v for v in all_vars if any(v.lower().startswith(p) for p in era_prefixes)
    ]
    other_vars = [
        v for v in all_vars if v not in sat_vars and v not in bio_vars and v not in era_vars
    ]

    corr_cutoff = params.get("correlation_cutoff", 0.7)

    n_sat = min(params.get("n_satellite", 3), len(sat_vars))
    n_bio = min(params.get("n_bioclim", 3), len(bio_vars))
    n_era = min(params.get("n_era5", 3), len(era_vars))
    sat_sel = sorted(sat_vars, key=lambda v: vif_vals.get(v, np.inf))[:n_sat]
    bio_sel = sorted(bio_vars, key=lambda v: vif_vals.get(v, np.inf))[:n_bio]
    era_sel = sorted(era_vars, key=lambda v: vif_vals.get(v, np.inf))[:n_era]

    to_drop_bio = set()
    to_drop_era = set()
    for bio in bio_sel:
        for era in era_sel:
            corr = abs(df[bio].corr(df[era]))
            if corr >= corr_cutoff:
                if vif_vals.get(bio, np.inf) >= vif_vals.get(era, np.inf):
                    to_drop_bio.add(bio)
                else:
                    to_drop_era.add(era)
    bio_sel = [b for b in bio_sel if b not in to_drop_bio]
    era_sel = [e for e in era_sel if e not in to_drop_era]
    keep = list(dict.fromkeys(sat_sel + bio_sel + era_sel + other_vars))
    ds_filt = ds[keep]
    if "boundary_mask" in ds.data_vars:
        ds_filt["boundary_mask"] = ds["boundary_mask"]
    return ds_filt, keep


# -----------------------------------------------------------------------------
def _load_bias_raster(cfg: dict, season: str, year: int) -> xr.DataArray | None:
    """Load bias raster for a given season and year.

    The path is taken from ``cfg['bias_grid']['pattern']`` which may contain
    ``{season}`` and ``{year}`` placeholders.  If the pattern is missing the
    function returns ``None``.
    """

    bias_cfg = cfg.get("bias_grid", {})
    pattern = bias_cfg.get("pattern")
    if not pattern:
        return None
    path = Path(pattern.format(season=season, year=year))
    if not path.exists():
        log.warning("Bias raster %s not found", path)
        return None
    with rasterio.open(path) as src:
        arr = src.read(1, masked=True).astype("float32")
        transform = src.transform
    x = transform.c + transform.a * (np.arange(arr.shape[1]) + 0.5)
    y = transform.f + transform.e * (np.arange(arr.shape[0]) + 0.5)
    return xr.DataArray(arr, coords={"y": y, "x": x}, dims=("y", "x"), name="bias")


# -----------------------------------------------------------------------------
def _merge_bias_maps(
    cfg: dict, season: str, mapping: list[YearOffset]
) -> xr.DataArray | None:
    """Fuse bias rasters for all years into a 3×2 mosaic.

    Returns ``None`` if no bias rasters could be loaded.
    """

    arrays: list[xr.DataArray] = []
    n_workers = 32
    with ThreadPoolExecutor(max_workers=n_workers) as exe:
        for yo, da in zip(
            mapping, exe.map(lambda y: _load_bias_raster(cfg, season, y.year), mapping)
        ):
            if da is None:
                continue
            arrays.append(da.assign_coords(x=da.x + yo.dx, y=da.y + yo.dy))
    if not arrays:
        return None
    merged = xr.combine_by_coords(arrays, fill_value=0)
    merged = merged.sortby("x")
    merged = merged.sortby("y", ascending=False)
    # ``xarray.combine_by_coords`` may return a ``Dataset`` when multiple
    # data arrays with the same name are merged.  Downstream code expects a
    # single ``DataArray``, so collapse any spurious dataset dimension.
    if isinstance(merged, xr.Dataset):
        merged = merged.to_array(dim="layer").max("layer")
    merged.name = "bias"
    return merged


# -----------------------------------------------------------------------------
def _sample_background_bias(
    bias: xr.DataArray,
    n: int,
    strength: float = 1.0,
    mask: np.ndarray | xr.DataArray | None = None,
) -> gpd.GeoDataFrame:
    """Sample ``n`` background points weighted by ``bias`` raster.

    The ``strength`` parameter linearly interpolates between uniform sampling
    (``strength=0``) and fully bias-weighted sampling (``strength=1``).  If
    ``mask`` is given, sampling is restricted to cells where ``mask`` is
    ``True``.
    """

    arr = np.nan_to_num(bias.values, nan=0.0)
    if mask is None:
        mask_arr = np.ones_like(arr)
    else:
        mask_arr = np.array(mask, dtype=float)
        if mask_arr.shape != arr.shape:
            # ``arr`` may represent a 3x2 multitemporal matrix where the
            # national boundary mask needs to be repeated rather than
            # rescaled.  When the bias raster dimensions are integer
            # multiples of the mask we tile it accordingly; otherwise we
            # fall back to a nearest-neighbour resize.
            tile_y = arr.shape[0] // mask_arr.shape[0]
            tile_x = arr.shape[1] // mask_arr.shape[1]
            if (
                tile_y * mask_arr.shape[0] == arr.shape[0]
                and tile_x * mask_arr.shape[1] == arr.shape[1]
            ):
                mask_arr = np.tile(mask_arr, (tile_y, tile_x))
            else:
                from scipy.ndimage import zoom

                zoom_factors = (
                    arr.shape[0] / mask_arr.shape[0],
                    arr.shape[1] / mask_arr.shape[1],
                )
                mask_arr = zoom(mask_arr, zoom_factors, order=0)
    arr = arr * mask_arr
    arr = arr * float(strength) + (1.0 - float(strength)) * mask_arr
    total = arr.sum()
    if total <= 0:
        raise ValueError("Bias raster has no positive values")
    probs = arr.ravel() / total
    idx = np.random.choice(arr.size, size=n, replace=False, p=probs)
    rows = idx // arr.shape[1]
    cols = idx % arr.shape[1]
    xs = bias.x.values[cols]
    ys = bias.y.values[rows]
    gdf = gpd.GeoDataFrame(
        {"bg_id": np.arange(n), "source": "background"},
        geometry=gpd.points_from_xy(xs, ys),
        crs="EPSG:2154",
    )
    gdf["x"] = gdf.geometry.x
    gdf["y"] = gdf.geometry.y
    return gdf


# -----------------------------------------------------------------------------
def _visualise_bias(
    bias: xr.DataArray,
    mapping: list[YearOffset],
    bounds,
    season: str,
    test_year: int,
    out_dir: Path,
) -> None:
    """Save a visualisation of the bias mosaic with year annotations."""

    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(
        bias.values,
        extent=[float(bias.x.min()), float(bias.x.max()), float(bias.y.min()), float(bias.y.max())],
        origin="upper",
    )
    fig.colorbar(im, ax=ax, shrink=0.7, label="Bias")
    for yo in mapping:
        b = translate(bounds, xoff=yo.dx, yoff=yo.dy)
        _plot_outline(ax, b, color="grey", linewidth=0.5)
        minx, miny, maxx, maxy = b.bounds
        ax.text(minx, maxy, str(yo.year), ha="left", va="top", fontsize=8)
    ax.set_title(f"{season} bias mosaic – test {test_year}")
    fig.tight_layout()
    out_file = out_dir / f"bias_{test_year}.png"
    fig.savefig(out_file, dpi=150)
    plt.close(fig)


# -----------------------------------------------------------------------------
def _master_stack_path(cfg: dict, season: str) -> Path:
    """Return path for the combined stack covering all years of ``season``."""

    base_dir = cfg.get("multitemporal", {}).get("stack_base_dir")
    if base_dir:
        base = Path(base_dir)
    else:
        base = _experiment_dir(cfg) / "data-stack"
    return base / f"{season}_master.zarr"


# -----------------------------------------------------------------------------
def _build_master_stack(cfg: dict, season: str) -> xr.Dataset:
    """Create predictor stack for ``season`` containing all years."""

    import tempfile

    years = cfg["multitemporal"]["years"]
    path = _master_stack_path(cfg, season)
    log.info("Building predictor stack for %s (years %s)", season, years)

    base_cfg = copy.deepcopy(cfg)
    base_cfg.pop("multitemporal", None)
    base_cfg.setdefault("experiment", {})["name"] = f"{season}_all"

    layers = base_cfg.get("predictors", {}).get("layers", {})
    for layer in layers.values():
        if isinstance(layer, dict):
            if "years" in layer:
                layer["years"] = years
            if "seasons" in layer:
                layer["seasons"] = [season.lower()]

    tmp_root = Path(tempfile.mkdtemp())
    base_cfg.setdefault("outputs", {})["root"] = str(tmp_root)
    from scripts.preprocessing import build_stack

    tmp_stack = build_stack(base_cfg, use_cache=False)
    ensure_dir(path.parent)
    if path.exists():
        shutil.rmtree(path)
    shutil.copytree(tmp_stack, path, dirs_exist_ok=True, copy_function=shutil.copy)
    shutil.rmtree(tmp_root, ignore_errors=True)
    log.info("Predictor stack written to %s", path)

    ds = xr.open_zarr(path, consolidated=False)
    empty_vars = [v for v in ds.data_vars if np.isnan(ds[v]).all()]
    if empty_vars:
        log.warning(
            "Stack %s has empty predictor layers: %s",
            season,
            ", ".join(empty_vars),
        )
    mask_file = cfg.get("boundary_mask")
    if mask_file:
        mask = load_mask(mask_file)
        mask_da = xr.DataArray(mask, coords={"y": ds.y, "x": ds.x}, dims=("y", "x"))
        ds = ds.where(mask_da)
        ds.to_zarr(path, mode="w")

    return ds


# -----------------------------------------------------------------------------
def _load_stack(
    cfg: dict,
    season: str,
    year: int,
    rebuild: bool = False,
    validate: bool = True,
) -> tuple[xr.Dataset, Path]:
    """Open predictor stack for ``season``/``year`` and apply filters."""

    path = _master_stack_path(cfg, season)
    expected = _expected_var_names(cfg, year, season.lower())

    if rebuild and path.exists():
        shutil.rmtree(path)

    if not path.exists():
        log.debug("Predictor stack %s not found; rebuilding", path)
        _build_master_stack(cfg, season)

    ds_master = xr.open_zarr(path, consolidated=False)

    if validate and not expected.issubset(set(ds_master.data_vars)):
        missing = expected - set(ds_master.data_vars)
        log.warning(
            "Stack %s missing expected variable(s): %s", path, sorted(missing)
        )

    year_pattern = re.compile(r"(^|_)(\d{4})(_|$)")
    vars_year = []
    for v in ds_master.data_vars:
        match = year_pattern.search(v)
        if match:
            if match.group(2) == str(year):
                vars_year.append(v)
        else:
            vars_year.append(v)
    if "boundary_mask" in ds_master.data_vars and "boundary_mask" not in vars_year:
        vars_year.append("boundary_mask")
    ds = ds_master[vars_year]
    if not cfg["multitemporal"].get("include_hunting", True):
        drop = [v for v in ds.data_vars if "hunting" in v.lower()]
        if drop:
            ds = ds.drop_vars(drop)
    # Drop ERA5 anomaly variables
    drop_anom = [v for v in ds.data_vars if "anom" in v.lower()]
    if drop_anom:
        ds = ds.drop_vars(drop_anom)
    ds, dropped = _apply_feature_config(ds, cfg, return_dropped=True)
    if dropped:
        ds = _fill_missing_satellite_layers(ds, ds_master, dropped, year)
    return ds, path


# -----------------------------------------------------------------------------
def _merge_stacks(
    cfg: dict,
    season: str,
    mapping: list[YearOffset],
    out_dir: Path,
    apply_vif: bool = True,
    stack_name: str = "training_stack",
    rebuild_years: bool = False,
    fixed_vars: list[str] | None = None,
) -> tuple[Path, list[str]]:
    """Combine individual year stacks into a shifted 3x2 matrix.

    The combined stack is written to ``out_dir/stack_name`` as individual
    GeoTIFF layers (one per predictor) before any filtering is applied.
    Subsequent filtering (e.g. VIF) operates on the in-memory dataset but the
    GeoTIFFs remain available for inspection.
    """

    n_workers = 32
    out_stack = out_dir / stack_name
    zarr_path = out_stack.with_suffix(".zarr")
    if zarr_path.exists():
        log.info("Loading cached stack from %s", zarr_path)
        merged = xr.open_zarr(zarr_path)
    else:
        log.debug("Merging %d yearly stacks using %d worker(s)", len(mapping), n_workers)
        with ProcessPoolExecutor(max_workers=n_workers) as exe:
            stacks = list(
                exe.map(
                    _load_shift_stack,
                    [(cfg, season, yo, rebuild_years) for yo in mapping],
                )
            )
        merged = xr.combine_by_coords(stacks)
        merged = fuse_yearly_layers(merged)
        merged = merged.sortby("x").sortby("y", ascending=False)
        merged = _apply_feature_config(merged, cfg)
        # Drop coordinate placeholders that may have been introduced as data variables
        merged = merged.drop_vars([v for v in ("x", "y") if v in merged.data_vars])
        merged.to_zarr(zarr_path, mode="w")
        log.info("Merged stack saved to %s", zarr_path)

    # Ensure a consistent spatial orientation with the origin in the
    # upper-left corner when exporting to GeoTIFF.
    merged = merged.drop_vars([v for v in ("x", "y") if v in merged.data_vars], errors="ignore")
    merged = merged.sortby("x").sortby("y", ascending=False)
    export_geotiffs(
        merged,
        out_stack,
        cfg["predictors"].get("nodata_value", np.nan),
        compress=cfg["maxent"].get("geotiff_compression", "LZW"),
        n_workers=n_workers,
    )
    log.info("Merged predictor layers written to %s", out_stack)

    if fixed_vars is not None:
        merged = merged[fixed_vars]
        vars_used = [v for v in merged.data_vars if v not in {"boundary_mask", "x", "y"}]
    elif apply_vif:
        merged, vars_used = _vif_filter(merged, cfg)
        vars_used = [v for v in vars_used if v not in {"x", "y"}]
        merged = merged.drop_vars([v for v in ("x", "y") if v in merged.data_vars], errors="ignore")
    else:
        vars_used = [v for v in merged.data_vars if v not in {"boundary_mask", "x", "y"}]
    empty_vars = [v for v in merged.data_vars if np.isnan(merged[v]).all()]
    if empty_vars:
        log.warning(
            "Merged stack has empty predictor layers: %s",
            ", ".join(empty_vars),
        )
    try:
        save_selected_predictors(out_stack, vars_used)
    except Exception as exc:  # pragma: no cover - best effort logging
        log.warning("Failed to persist predictor list in %s: %s", out_stack, exc)
    return out_stack, vars_used


# -----------------------------------------------------------------------------
def _visualise_matrix(
    pres_list: list[gpd.GeoDataFrame],
    mapping: list[YearOffset],
    bounds,
    season: str,
    test_year: int,
    out_dir: Path,
) -> None:
    """Plot training matrix with annotations for every year copy."""

    fig, ax = plt.subplots(figsize=(8, 6))
    for gdf, yo in zip(pres_list, mapping):
        gdf.plot(ax=ax, markersize=2, label=str(yo.year))
        b = translate(bounds, xoff=yo.dx, yoff=yo.dy)
        _plot_outline(ax, b, color="grey", linewidth=0.5)
        minx, miny, maxx, maxy = b.bounds
        ax.text(minx, maxy, str(yo.year), ha="left", va="top", fontsize=8)
    ax.set_title(f"{season} training matrix – test {test_year}")
    ax.set_aspect("equal")
    ax.legend(loc="upper right", fontsize=6)
    out_file = out_dir / f"matrix_train_{test_year}.png"
    fig.savefig(out_file, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Training matrix figure saved to %s", out_file)


# -----------------------------------------------------------------------------
def _visualise_test(
    pres: gpd.GeoDataFrame, bounds, season: str, test_year: int, out_dir: Path
) -> None:
    """Plot test set presences."""

    fig, ax = plt.subplots(figsize=(6, 5))
    pres.plot(
        ax=ax,
        markersize=2,
        marker="o",
        facecolor="none",
        edgecolor="blue",
        linewidth=0.5,
    )
    _plot_outline(ax, bounds, color="grey", linewidth=0.5)
    ax.set_title(f"{season} test set {test_year}")
    ax.set_aspect("equal")
    out_file = out_dir / f"testset_{test_year}.png"
    fig.savefig(out_file, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Test set figure saved to %s", out_file)


# -----------------------------------------------------------------------------
def _visualise_points(
    gdf: gpd.GeoDataFrame,
    mapping: list[YearOffset],
    bounds,
    season: str,
    test_year: int,
    out_dir: Path,
    prefix: str,
) -> None:
    """Plot shifted points (presence/background) across the 3x2 matrix."""

    fig, ax = plt.subplots(figsize=(8, 6))
    if prefix == "background":
        edge, size = "0.3", 0.5
    elif prefix == "train":
        edge, size = "red", 2
    else:
        edge, size = "green", 2
    gdf.plot(
        ax=ax,
        markersize=size,
        marker="o",
        facecolor="none",
        edgecolor=edge,
        linewidth=0.5,
    )
    for yo in mapping:
        b = translate(bounds, xoff=yo.dx, yoff=yo.dy)
        _plot_outline(ax, b, color="grey", linewidth=0.5)
        minx, miny, maxx, maxy = b.bounds
        ax.text(minx, maxy, str(yo.year), ha="left", va="top", fontsize=8)
    ax.set_aspect("equal")
    ax.set_title(f"{season} {prefix} points – test {test_year}")
    out_file = out_dir / f"{season}_{prefix}_points.png"
    fig.savefig(out_file, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("%s points figure saved to %s", prefix.capitalize(), out_file)


# -----------------------------------------------------------------------------
def _visualise_features(
    stack_path: Path,
    vars_used: list[str],
    season: str,
    test_year: int,
    out_dir: Path,
) -> None:
    """Plot all predictor layers as small multiples."""

    n = len(vars_used)
    ncols = 3
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3, nrows * 3))
    for i, var in enumerate(vars_used):
        ax = axes.flat[i]
        tif = stack_path / f"{var}.tif"
        with rasterio.open(tif) as src:
            img = src.read(1).astype("float32")
            nodata = src.nodata if src.nodata is not None else -9999
            img[img == nodata] = np.nan
        ax.imshow(img, cmap="viridis")
        ax.set_title(var, fontsize=6)
        ax.set_axis_off()
    for j in range(i + 1, nrows * ncols):
        axes.flat[j].set_axis_off()
    fig.suptitle(f"{season} predictors – test {test_year}")
    fig.tight_layout()
    out_file = out_dir / f"{season}_predictors.png"
    fig.savefig(out_file, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Predictor figure saved to %s", out_file)


# -----------------------------------------------------------------------------
def _predict_raster(
    model,
    stack_path: Path,
    vars_used: list[str],
    cfg: dict,
    suffix: str = "test",
) -> Path:
    """Predict suitability for ``stack_path`` and save cloglog raster."""

    arrays = []
    meta = None
    for var in vars_used + ["boundary_mask"]:
        tif = stack_path / f"{var}.tif"
        with rasterio.open(tif) as src:
            arr = src.read(1).astype("float32")
            if meta is None:
                meta = src.meta.copy()
            arrays.append(arr)
    boundary_arr = arrays[-1]
    boundary = np.isfinite(boundary_arr) & (boundary_arr > 0)
    grid = np.stack(arrays[:-1], axis=0).reshape(len(vars_used), -1).T
    for i in range(grid.shape[1]):
        col = grid[:, i]
        m = np.nanmean(col)
        col[np.isnan(col)] = m
        grid[:, i] = col

    preds = np.full(boundary.size, np.nan, dtype="float32")
    idx = np.flatnonzero(boundary.ravel())
    batch_size = 100000
    for start in range(0, idx.size, batch_size):
        sub = idx[start:start + batch_size]
        preds[sub] = model.predict(grid[sub])
    suitability = preds.reshape(boundary.shape)
    suitability[~boundary] = np.nan

    meta.update(
        {
            "height": boundary.shape[0],
            "width": boundary.shape[1],
            "count": 1,
            "dtype": "float32",
            "nodata": np.nan,
        }
    )

    # Adjust transform by half a pixel north-west to correct a historical
    # half-cell (≈500 m) offset between cloglog outputs and input layers.
    meta["transform"] = meta["transform"] * Affine.translation(-0.5, -0.5)

    out_dir = Path(cfg["outputs"]["maps_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_name = sanitize_name(cfg["experiment"]["name"])
    out_file = out_dir / f"{safe_name}_{suffix}_cloglog.tif"
    with rasterio.open(out_file, "w", **meta) as dst:
        dst.write(suitability.astype("float32"), 1)
    log.info("%s raster saved to %s", suffix.capitalize(), out_file)
    return out_file


# -----------------------------------------------------------------------------
def _confidence_interval(vals: np.ndarray) -> float:
    """95% confidence interval half width."""
    if len(vals) == 0:
        return float("nan")
    return 1.96 * np.nanstd(vals, ddof=1) / math.sqrt(len(vals))


# -----------------------------------------------------------------------------
def process_season(cfg: dict, season: str, rebuild: bool) -> None:
    """Run multitemporal modelling for a single season."""

    years = cfg["multitemporal"]["years"]
    shift = cfg["multitemporal"]["shift_km"] * 1000
    exp_dir = ensure_dir(_experiment_dir(cfg))
    out_root = ensure_dir(exp_dir / "output" / season)
    stack_dir = ensure_dir(exp_dir / "data-stack")
    log_heading(f"Processing season {season}")

    run_test_year = cfg.get("run_test_year", "all")
    if run_test_year != "all":
        run_test_year = int(run_test_year)
        if run_test_year not in years:
            raise ValueError(f"test year {run_test_year} not in configured years")
        test_years = [run_test_year]
    else:
        test_years = years
    test_years = restrict_test_years(list(test_years), cfg)

    master_path = _master_stack_path(cfg, season)
    if rebuild or not master_path.exists():
        _build_master_stack(cfg, season)

    # Boundary polygon used for visualisation and background sampling
    fr_poly = _france_outline_from_mask(cfg["boundary_mask"])
    bounds = fr_poly.iloc[0]
    mask = load_mask(cfg["boundary_mask"])

    # Load presences for all years; thinning will be applied later on the
    # combined training dataset so each copy of France is processed only once.
    pres_raw: dict[int, gpd.GeoDataFrame] = {
        y: _load_presence(cfg, season, y) for y in years
    }
    thin_cfg = cfg.get("presence_data", {}).get("thinning", {})
    vars_fixed: list[str] | None = None

    metrics_records = []
    engine_specs = _resolve_engines(cfg)
    for test_year in tqdm(test_years, desc=f"{season} CV", unit="year"):
        log_heading(f"{season} - test year {test_year}")
        log.info("Processing test year %d", test_year)
        mapping = _build_matrix(years, test_year, shift)
        run_dir = ensure_dir(out_root / f"test_{test_year}")
        input_dir = ensure_dir(exp_dir / f"input-data-test-{test_year}")

        # create run-specific configuration to avoid filename collisions
        cfg_run = copy.deepcopy(cfg)
        mt_cfg = cfg_run.setdefault("multitemporal", {})
        if "stack_base_dir" not in mt_cfg:
            mt_cfg["stack_base_dir"] = str(stack_dir)
        cfg_run["outputs"]["root"] = str(run_dir)
        cfg_run["outputs"]["logs_dir"] = str(run_dir / "logs")
        cfg_run["outputs"]["model_dir"] = str(run_dir / "models")
        cfg_run["outputs"]["maps_dir"] = str(run_dir / "maps")
        cfg_run["outputs"]["figures_dir"] = str(run_dir / "figures")
        cfg_run["experiment"]["vis_dir"] = str(run_dir / "visualizations")
        cfg_run["experiment"]["fi_dir"] = str(run_dir / "feature_importance")
        cfg_run["experiment"]["stats_dir"] = str(run_dir / "stats")
        for p in [
            cfg_run["outputs"]["logs_dir"],
            cfg_run["outputs"]["model_dir"],
            cfg_run["outputs"]["maps_dir"],
            cfg_run["outputs"]["figures_dir"],
            cfg_run["experiment"]["vis_dir"],
            cfg_run["experiment"]["fi_dir"],
            cfg_run["experiment"]["stats_dir"],
        ]:
            ensure_dir(p)

        vis_dir = Path(cfg_run["experiment"]["vis_dir"])
        log_dir = Path(cfg_run["outputs"]["logs_dir"])
        log_file = log_dir / "run.log"
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.INFO)
        fh.setFormatter(
            logging.Formatter(
                "%(asctime)s [%(levelname)s] %(name)s > %(message)s"
            )
        )
        logging.getLogger().addHandler(fh)

        bias_da = _merge_bias_maps(cfg_run, season, mapping)
        if bias_da is not None:
            export_geotiffs(
                xr.Dataset({"bias": bias_da}),
                input_dir,
                cfg_run["predictors"].get("nodata_value", np.nan),
                n_time_steps=1,
                compress=cfg_run["maxent"].get("geotiff_compression", "LZW"),
            )
            _visualise_bias(bias_da, mapping, bounds, season, test_year, vis_dir)

        # Build combined predictor stack for this training rotation.  The first
        # iteration applies VIF filtering to derive a fixed set of variables
        # which is reused in all subsequent rotations.
        stack_name = f"{season}_stack"
        stack_path = input_dir / stack_name
        if rebuild or not stack_path.exists():
            log_heading("Building combined predictor stack")
            stack_path, vars_used = _merge_stacks(
                cfg_run,
                season,
                mapping,
                input_dir,
                stack_name=stack_name,
                rebuild_years=False,
                fixed_vars=vars_fixed,
                apply_vif=vars_fixed is None,
            )
            if vars_fixed is None:
                vars_fixed = vars_used
        else:
            vars_used = load_selected_predictors(stack_path)
            if not vars_used:
                vars_used = [
                    f.stem
                    for f in stack_path.glob("*.tif")
                    if f.stem != "boundary_mask"
                ]
                log.warning(
                    "Selected predictor list missing in %s; falling back to all layers",
                    stack_path,
                )
            if vars_fixed is None:
                vars_fixed = vars_used
        log.info("Using predictors: %s", ", ".join(vars_used))

        # Training data preparation – shift presences and thin once across the
        # entire matrix.
        log_heading("Preparing training data")
        pres_list = []
        log.info("Shifting presence points across %d training years", len(mapping))
        for yo in tqdm(mapping, desc="Shift presences", leave=False):
            gdf = pres_raw[yo.year].copy()
            gdf.geometry = gdf.geometry.translate(xoff=yo.dx, yoff=yo.dy)
            gdf["x"] = gdf.geometry.x
            gdf["y"] = gdf.geometry.y
            pres_list.append(gdf)
        pres_train = gpd.GeoDataFrame(
            pd.concat(pres_list, ignore_index=True), crs="EPSG:2154"
        )
        pres_train = deduplicate_cells(pres_train)
        grid_km = thin_cfg.get("grid_size_km")
        if grid_km:
            pres_train = grid_thin(pres_train, grid_km)
            log.info(
                "Grid thinning %.0f km retained %d records",
                grid_km,
                len(pres_train),
            )
        log.info("Finished shifting presence points (%d records)", len(pres_train))

        if thin_cfg.get("enable", False):
            env = extract_from_stack(stack_path, pres_train, vars_used)
            perc_map = thin_cfg.get("percentages", {}).get(season, {})
            frames: list[pd.DataFrame] = []
            if perc_map:
                for src, year_map in perc_map.items():
                    mask_src = pres_train["source"].astype(str) == src
                    g_src = pres_train.loc[mask_src].reset_index(drop=True)
                    env_src = env[mask_src]
                    if isinstance(year_map, dict):
                        for yr, pct in year_map.items():
                            mask_year = g_src["year"] == int(yr)
                            g_sub = g_src.loc[mask_year].reset_index(drop=True)
                            env_sub = env_src[mask_year]
                            if len(g_sub) == 0:
                                log.warning(
                                    "No presence records for %s %s; skipping thinning",
                                    src,
                                    yr,
                                )
                                continue
                            if len(g_sub) <= 3:
                                log.warning(
                                    "Too few records for %s %s (%d); thinning skipped",
                                    src,
                                    yr,
                                    len(g_sub),
                                )
                                frames.append(g_sub)
                                continue
                            if pct < 1.0:
                                tgt = max(1, int(len(g_sub) * pct))
                                log.info(
                                    "Mahalanobis thinning %s %s: %d -> %d points",
                                    src,
                                    yr,
                                    len(g_sub),
                                    tgt,
                                )
                                frames.append(mahalanobis_thin(g_sub, tgt, env_sub))
                            else:
                                frames.append(g_sub)
                    else:
                        pct = year_map
                        if len(g_src) == 0:
                            log.warning(
                                "No presence records for %s; skipping thinning", src
                            )
                            continue
                        if len(g_src) <= 3:
                            log.warning(
                                "Too few records for %s (%d); thinning skipped",
                                src,
                                len(g_src),
                            )
                            frames.append(g_src)
                            continue
                        if pct < 1.0:
                            tgt = max(1, int(len(g_src) * pct))
                            log.info(
                                "Mahalanobis thinning %s: %d -> %d points",
                                src,
                                len(g_src),
                                tgt,
                            )
                            frames.append(mahalanobis_thin(g_src, tgt, env_src))
                        else:
                            frames.append(g_src)
                remaining = pres_train.loc[~pres_train["source"].isin(perc_map.keys())]
                if not remaining.empty:
                    frames.append(remaining)
                pres_train = gpd.GeoDataFrame(
                    pd.concat(frames, ignore_index=True), crs="EPSG:2154"
                )
            else:
                pct = thin_cfg.get("percentage", 1.0)
                if pct < 1.0:
                    tgt = max(1, int(len(pres_train) * pct))
                    pres_train = gpd.GeoDataFrame(
                        mahalanobis_thin(pres_train, tgt, env), crs="EPSG:2154"
                    )
        pres_train = deduplicate_cells(pres_train)

        # Background sampling
        bg_per_copy = cfg["multitemporal"].get("background_per_year", 10000)
        bias_strength = cfg_run.get("background", {}).get("bias_strength", 0.4)
        if (
            cfg_run.get("background", {}).get("use_bias_grid", False)
            and bias_da is not None
            and bias_strength > 0
        ):
            log.info(
                "Sampling background points using bias raster (%d per copy)",
                bg_per_copy,
            )
            bg_train = _sample_background_bias(
                bias_da, bg_per_copy * len(mapping), bias_strength, mask
            )
            bg_train = deduplicate_cells(bg_train)
        else:
            log.info(
                "Sampling background points for %d matrix copies", len(mapping)
            )
            bg_list = []
            for i, yo in enumerate(tqdm(mapping, desc="Sample background", leave=False)):
                shifted = fr_poly.translate(xoff=yo.dx, yoff=yo.dy)
                g = _sample_background_polygon(shifted, bg_per_copy)
                g["bg_id"] = np.arange(len(g)) + i * bg_per_copy
                bg_list.append(g)
            bg_train = gpd.GeoDataFrame(
                pd.concat(bg_list, ignore_index=True), crs="EPSG:2154"
            )
            bg_train = deduplicate_cells(bg_train)
        bg_train = drop_background_overlaps(
            pres_train, bg_train, context="training background"
        )

        log.info("Background points prepared: %d", len(bg_train))
        log.debug("Training presences: %d", len(pres_train))
        log.debug("Training background points: %d", len(bg_train))

        # Split off validation subset
        val_frac = cfg["multitemporal"].get("validation_fraction", 0.1)
        rng = np.random.default_rng(cfg["experiment"].get("random_seed", 0))
        pres_val = pres_train.sample(frac=val_frac, random_state=rng.integers(1e9))
        bg_val = bg_train.sample(frac=val_frac, random_state=rng.integers(1e9))
        pres_train = pres_train.drop(pres_val.index).reset_index(drop=True)
        bg_train = bg_train.drop(bg_val.index).reset_index(drop=True)
        log.debug("Validation presences: %d", len(pres_val))
        log.debug("Validation background points: %d", len(bg_val))

        # Prepare test data before model training
        pres_test = _load_presence(cfg, season, test_year)
        bias_test = _merge_bias_maps(
            cfg_run, season, [YearOffset(test_year, 0.0, 0.0)]
        )
        if (
            cfg_run.get("background", {}).get("use_bias_grid", False)
            and bias_test is not None
            and bias_strength > 0
        ):
            bg_test = _sample_background_bias(bias_test, bg_per_copy, bias_strength, mask)
        else:
            bg_test = _sample_background(cfg, bg_per_copy)
        bg_test = deduplicate_cells(bg_test)

        test_stack, _ = _load_stack(
            cfg_run, season, test_year, rebuild, validate=rebuild
        )
        # harmonise yearly prefixed variable names with the merged stack
        # returned by ``_merge_stacks``.  Training variables are stored
        # without their ``<year>_`` prefix (e.g. ``2017_class1`` -> ``class1``)
        # so the test stack must follow the same convention to avoid being
        # flagged as missing predictors.
        test_stack = fuse_yearly_layers(test_stack)
        missing = [v for v in vars_used if v not in test_stack.data_vars]
        if missing:
            log.warning(
                "Missing predictors for test year %d: %s",
                test_year,
                ", ".join(missing),
            )
            vars_used = [v for v in vars_used if v not in missing]
            if vars_fixed is not None:
                vars_fixed = [v for v in vars_fixed if v in vars_used]
        export_vars = vars_used + (
            ["boundary_mask"] if "boundary_mask" in test_stack.data_vars else []
        )
        test_stack = test_stack.sortby(["y", "x"])[export_vars]
        test_stack_path = run_dir / "test_stack"
        export_geotiffs(
            test_stack,
            test_stack_path,
            cfg_run["predictors"].get("nodata_value", np.nan),
            compress=cfg_run["maxent"].get("geotiff_compression", "LZW"),
            n_workers=32,
        )
        log.info("Test stack written to %s", test_stack_path)

        env_test = extract_from_stack(test_stack_path, pres_test, vars_used)
        if thin_cfg.get("enable", False):
            if "percentages" in thin_cfg:
                season_map = thin_cfg.get("percentages", {}).get(season, {})
                frames = []
                for src, year_map in season_map.items():
                    g_src = pres_test[pres_test["source"] == src]
                    if len(g_src) == 0:
                        continue
                    pct = (
                        year_map.get(test_year, 1.0)
                        if isinstance(year_map, dict)
                        else year_map
                    )
                    if pct < 1.0 and len(g_src) > 3:
                        tgt = max(1, int(len(g_src) * pct))
                        env_src = env_test[pres_test["source"] == src]
                        frames.append(mahalanobis_thin(g_src, tgt, env_src))
                    else:
                        frames.append(g_src)
                remaining = pres_test.loc[
                    ~pres_test["source"].isin(season_map.keys())
                ]
                if not remaining.empty:
                    frames.append(remaining)
                pres_test = gpd.GeoDataFrame(
                    pd.concat(frames, ignore_index=True), crs="EPSG:2154"
                )
            else:
                pct = thin_cfg.get("percentage", 1.0)
                if pct < 1.0 and len(pres_test) > 3:
                    tgt = max(1, int(len(pres_test) * pct))
                    pres_test = gpd.GeoDataFrame(
                        mahalanobis_thin(pres_test, tgt, env_test), crs="EPSG:2154"
                    )
        pres_test = deduplicate_cells(pres_test)
        bg_test = drop_background_overlaps(
            pres_test, bg_test, context="test background"
        )

        # Persist thinned presence points for reuse
        pts_dir = ensure_dir(run_dir / "points")
        pres_train.to_file(pts_dir / "train.gpkg", driver="GPKG")
        pres_val.to_file(pts_dir / "val.gpkg", driver="GPKG")
        pres_test.to_file(pts_dir / "test.gpkg", driver="GPKG")
        env_test = extract_from_stack(test_stack_path, pres_test, vars_used)

        X_pres = env_test
        X_bg = extract_from_stack(test_stack_path, bg_test, vars_used)

        # Persist training inputs for debugging and reproducibility
        pres_file = input_dir / f"{season}_train_points.gpkg"
        bg_file = input_dir / f"{season}_background_points.gpkg"
        pres_train.to_file(pres_file, driver="GPKG")
        bg_train.to_file(bg_file, driver="GPKG")
        env_pres = extract_from_stack(stack_path, pres_train, vars_used)
        df_pres = pres_train.drop(columns="geometry").copy()
        pres_env_df = pd.DataFrame(env_pres, columns=vars_used)
        df_pres = pd.concat([df_pres.reset_index(drop=True), pres_env_df], axis=1)
        df_pres.to_csv(input_dir / f"{season}_train_features.csv", index=False)

        env_bg = extract_from_stack(stack_path, bg_train, vars_used)
        df_bg = bg_train.drop(columns="geometry").copy()
        bg_env_df = pd.DataFrame(env_bg, columns=vars_used)
        df_bg = pd.concat([df_bg.reset_index(drop=True), bg_env_df], axis=1)
        df_bg.to_csv(input_dir / f"{season}_background_features.csv", index=False)
        log.debug("Saved training data to %s", input_dir)

        env_pres_val = extract_from_stack(stack_path, pres_val, vars_used)
        env_bg_val = extract_from_stack(stack_path, bg_val, vars_used)

        _visualise_points(pres_train, mapping, bounds, season, test_year, vis_dir, prefix="train")
        _visualise_points(bg_train, mapping, bounds, season, test_year, vis_dir, prefix="background")
        _visualise_features(stack_path, vars_used, season, test_year, vis_dir)

        if cfg_run.get("export_java_compatible_input_data", True):
            export_java_compatible_bundle(
                cfg_run,
                season,
                test_year,
                stack_path,
                vars_used,
                pres_train,
                bg_train,
                input_dir,
                bias=bias_da,
            )

        _visualise_matrix(pres_list, mapping, bounds, season, test_year, vis_dir)

        for engine_label, trainer in engine_specs:
            engine_dir = ensure_dir(run_dir / engine_label)
            cfg_engine = copy.deepcopy(cfg_run)
            cfg_engine.setdefault("experiment", {})
            cfg_engine.setdefault("outputs", {})
            cfg_engine["experiment"]["name"] = (
                f"{cfg_run['experiment']['name']}_{engine_label}"
            )
            cfg_engine["outputs"]["root"] = str(engine_dir)
            cfg_engine["outputs"]["logs_dir"] = str(engine_dir / "logs")
            cfg_engine["outputs"]["model_dir"] = str(engine_dir / "models")
            cfg_engine["outputs"]["maps_dir"] = str(engine_dir / "maps")
            cfg_engine["outputs"]["figures_dir"] = str(engine_dir / "figures")
            cfg_engine["experiment"]["vis_dir"] = str(engine_dir / "visualizations")
            cfg_engine["experiment"]["fi_dir"] = str(engine_dir / "feature_importance")
            cfg_engine["experiment"]["stats_dir"] = str(engine_dir / "stats")
            cfg_engine.setdefault("model", {})["engines"] = [engine_label]
            for path in [
                cfg_engine["outputs"]["logs_dir"],
                cfg_engine["outputs"]["model_dir"],
                cfg_engine["outputs"]["maps_dir"],
                cfg_engine["outputs"]["figures_dir"],
                cfg_engine["experiment"]["vis_dir"],
                cfg_engine["experiment"]["fi_dir"],
                cfg_engine["experiment"]["stats_dir"],
            ]:
                ensure_dir(path)

            os.environ["LOGS_DIR"] = cfg_engine["outputs"]["logs_dir"]

            if trainer == "maxent":
                model, _, vars_engine = train_model(
                    cfg_engine, stack_path, list(vars_used), pres_train, bg_train
                )
            elif trainer == "gbm":
                from scripts.gbm_training import train_model as train_gbm_model

                model, _, vars_engine = train_gbm_model(
                    cfg_engine, stack_path, list(vars_used), pres_train, bg_train
                )
            else:
                raise ValueError(f"Unknown trainer '{trainer}'")

            vars_engine = list(vars_engine)
            response_dir = Path(cfg_engine["experiment"]["vis_dir"]) / "response_curves"
            export_response_curves(
                model,
                pd.concat([pres_env_df, bg_env_df], ignore_index=True)[vars_engine],
                vars_engine,
                response_dir,
            )
            compile_response_curves_pdf(
                response_dir, response_dir / "response_curves.pdf"
            )

            if (
                cfg_engine.get("evaluation", {})
                .get("cross_validation", {})
                .get("enable", False)
            ) and trainer == "maxent":
                run_cv(
                    cfg_engine,
                    pres_gdf=pres_train,
                    bg_gdf=bg_train,
                    stack_path=stack_path,
                    vars_used=vars_engine,
                )

            engine_vis = Path(cfg_engine["experiment"]["vis_dir"])
            _visualise_test(pres_test, bounds, season, test_year, engine_vis)
            train_tif = _predict_raster(
                model, stack_path, vars_engine, cfg_engine, suffix="train"
            )
            test_tif = _predict_raster(
                model, test_stack_path, vars_engine, cfg_engine, suffix="test"
            )
            plot_cloglog_points(
                train_tif,
                pres_train,
                bounds,
                mapping,
                engine_vis / f"train_cloglog_train_points_{test_year}.png",
                f"{season} train points – test {test_year}",
                edgecolor="white",
            )
            plot_cloglog_points(
                train_tif,
                pres_val,
                bounds,
                mapping,
                engine_vis / f"train_cloglog_val_points_{test_year}.png",
                f"{season} val points – test {test_year}",
                edgecolor="green",
            )
            plot_cloglog_points(
                test_tif,
                pres_test,
                bounds,
                [YearOffset(test_year, 0, 0)],
                engine_vis / f"test_cloglog_test_points_{test_year}.png",
                f"{season} test points – test {test_year}",
                edgecolor="blue",
                label_years=False,
            )

            def _extract(vars_list: list[str], stack: Path, pres, bg):
                env_pres_local = extract_from_stack(stack, pres, vars_list)
                env_bg_local = extract_from_stack(stack, bg, vars_list)
                return env_pres_local, env_bg_local

            env_pres_engine, env_bg_engine = _extract(
                vars_engine, stack_path, pres_train, bg_train
            )
            env_pres_val_engine, env_bg_val_engine = _extract(
                vars_engine, stack_path, pres_val, bg_val
            )
            X_pres_engine, X_bg_engine = _extract(
                vars_engine, test_stack_path, pres_test, bg_test
            )

            y_true_train = np.concatenate(
                [np.ones(len(env_pres_engine)), np.zeros(len(env_bg_engine))]
            )
            y_score_train = np.concatenate(
                [model.predict(env_pres_engine), model.predict(env_bg_engine)]
            )
            y_true_val = np.concatenate(
                [np.ones(len(env_pres_val_engine)), np.zeros(len(env_bg_val_engine))]
            )
            y_score_val = np.concatenate(
                [
                    model.predict(env_pres_val_engine),
                    model.predict(env_bg_val_engine),
                ]
            )
            y_true_test = np.concatenate(
                [np.ones(len(X_pres_engine)), np.zeros(len(X_bg_engine))]
            )
            y_score_test = np.concatenate(
                [model.predict(X_pres_engine), model.predict(X_bg_engine)]
            )

            train_gdf = gpd.GeoDataFrame(
                pd.concat([pres_train, bg_train], ignore_index=True), crs=pres_train.crs
            )
            val_gdf = gpd.GeoDataFrame(
                pd.concat([pres_val, bg_val], ignore_index=True), crs=pres_val.crs
            )
            test_gdf = gpd.GeoDataFrame(
                pd.concat([pres_test, bg_test], ignore_index=True), crs=pres_test.crs
            )

            n_params = count_model_params(model)
            evaluate(
                f"{season}_test_{test_year}_{engine_label}",
                y_true_train,
                y_score_train,
                cfg_engine,
                y_true_test,
                y_score_test,
                y_true_val,
                y_score_val,
                n_params=n_params,
                gdf_train=train_gdf,
                gdf_test=test_gdf,
                gdf_val=val_gdf,
            )

            fig_dir = Path(cfg_engine["outputs"]["figures_dir"])
            safe_name = sanitize_name(f"{season}_test_{test_year}_{engine_label}")
            for name in [
                f"{safe_name}_threshold_auc.png",
                f"{safe_name}_threshold_cbi.png",
            ]:
                src = fig_dir / name
                if src.exists():
                    shutil.copy(src, engine_vis / name)

            thr = cfg_engine["evaluation"].get("classification_threshold", 0.5)
            train_metrics = _compute_metrics(
                y_true_train, y_score_train, threshold=thr, n_params=n_params
            )
            val_metrics = _compute_metrics(
                y_true_val, y_score_val, threshold=thr, n_params=n_params
            )
            test_metrics = _compute_metrics(
                y_true_test, y_score_test, threshold=thr, n_params=n_params
            )

            metrics_records.append(
                {
                    "season": season,
                    "test_year": test_year,
                    "split": "train",
                    "engine": engine_label,
                    **train_metrics,
                }
            )
            metrics_records.append(
                {
                    "season": season,
                    "test_year": test_year,
                    "split": "validation",
                    "engine": engine_label,
                    **val_metrics,
                }
            )
            metrics_records.append(
                {
                    "season": season,
                    "test_year": test_year,
                    "split": "test",
                    "engine": engine_label,
                    **test_metrics,
                }
            )

            model_dir = Path(cfg_engine["outputs"]["model_dir"])
            model_file = (
                model_dir / f"{sanitize_name(cfg_engine['experiment']['name'])}_model.pkl"
            )
            with open(model_file, "wb") as f:
                pickle.dump(model, f)
            log.info("Model weights saved to %s", model_file)

            if (
                cfg_engine.get("variable_importance", {})
                .get("permutation", {})
                .get("enable", False)
            ):
                permutation_importance(
                    cfg_engine,
                    model,
                    stack_path,
                    vars_engine,
                    pres_train,
                    bg_train,
                    engine_dir / "importance",
                )

        logging.getLogger().removeHandler(fh)
        fh.close()

    metrics_df = pd.DataFrame(metrics_records)
    metrics_path = out_root / "temporal_cv_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    log.info("Temporal CV metrics written to %s", metrics_path)
    summary_records: list[dict] = []
    for (engine_label, split), grp in metrics_df.groupby(["engine", "split"]):
        summary = {"engine": engine_label, "split": split}
        metrics_to_summarise = [
            "auc",
            "cbi",
            "cbi_pearson",
            "cbi_spearman",
            "omission",
            "aicc",
            "pr_auc",
            "precision",
            "recall",
            "f1",
        ]
        for metric in metrics_to_summarise:
            if metric not in grp:
                continue
            vals = grp[metric].to_numpy()
            summary[f"{metric}_mean"] = np.nanmean(vals)
            summary[f"{metric}_ci95"] = _confidence_interval(vals)
        summary_records.append(summary)
    summary_path = out_root / "temporal_cv_summary.csv"
    pd.DataFrame(summary_records).to_csv(summary_path, index=False)
    log.info("Temporal CV summary written to %s", summary_path)


# -----------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="Run multitemporal seasonal models")
    ap.add_argument("--config", required=True, help="Path to configuration YAML")
    ap.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level (DEBUG, INFO, WARNING, ERROR)",
    )
    args = ap.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))
    cfg = load_config(args.config)

    seasons = cfg["multitemporal"]["seasons"]
    years = cfg["multitemporal"]["years"]
    exp_dir = ensure_dir(_experiment_dir(cfg))
    merged_exist = all(
        (exp_dir / f"input-data-test-{y}" / f"{season}_stack").exists()
        for y in years
        for season in seasons
    )
    if merged_exist:
        rebuild = (
            input("Rebuild merged predictor layers? (y/n): ")
            .strip()
            .lower()
            .startswith("y")
        )
    else:
        prev = _last_experiment_dir(cfg)
        if prev and any(prev.glob("input-data-test-*")):
            reuse = timed_input(
                f"Reuse merged predictor layers from previous experiment '{prev.name}'? (y/n): ",
                timeout=10,
                default="n",
            ).lower().startswith("y")
            if reuse:
                for d in prev.glob("input-data-test-*"):
                    shutil.copytree(d, exp_dir / d.name, dirs_exist_ok=True)
                log.info("Reusing merged predictor layers from %s", prev)
                rebuild = False
            else:
                rebuild = True
        else:
            rebuild = True

    for season in tqdm(cfg["multitemporal"]["seasons"], desc="Seasons", unit="season"):
        log.info("Starting season %s", season)
        process_season(cfg, season, rebuild)


if __name__ == "__main__":
    main()
