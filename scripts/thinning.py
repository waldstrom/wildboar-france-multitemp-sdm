# =============================================================================
# Project Title: Wild Boar Distribution Model for France
# Path: scripts/thinning.py
# Purpose: Apply spatial thinning based on Mahalanobis distance and gridded filters.
# Process Step: Reduces sampling bias in presence data prior to model fitting.
# Created by Adrian Meyer, Institute Geomatics, FHNW (adrian.meyer@fhnw.ch)
# Created: July 2025
# Version: v.0.1.0
# =============================================================================
"""
Spatial thinning - Mahalanobis distance maximisation and grid-based filter.
"""
from __future__ import annotations
import logging
import os
import shutil
import json
import hashlib
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.spatial import distance
from tqdm import tqdm
from tqdm.auto import trange

from scripts.utils import ensure_dir, ensure_gdf_crs, get_processed_dir

log = logging.getLogger(__name__)


def _meta_path(path: Path) -> Path:
    """Return sidecar JSON path storing the cache signature for ``path``."""

    return Path(f"{path}.meta.json")


def _read_signature(meta_path: Path) -> str | None:
    """Return cache signature stored in ``meta_path``, if available."""

    try:
        data = json.loads(meta_path.read_text())
    except FileNotFoundError:
        return None
    except json.JSONDecodeError:
        log.warning("Failed to parse thinning cache metadata from %s", meta_path)
        return None
    return data.get("signature")


def _write_signature(meta_path: Path, signature: str) -> None:
    """Persist ``signature`` to ``meta_path`` as JSON."""

    ensure_dir(meta_path.parent)
    meta_path.write_text(json.dumps({"signature": signature}))


def thinning_signature(cfg: dict) -> str:
    """Return a deterministic cache signature for the thinning configuration."""

    presence_cfg = cfg.get("presence_data", {})
    thinning_cfg = presence_cfg.get("thinning", {})
    presence_core = {k: v for k, v in presence_cfg.items() if k != "thinning"}
    thinning_core = {k: v for k, v in thinning_cfg.items() if k != "save_thinned_points"}
    payload = {
        "presence": presence_core,
        "thinning": thinning_core,
        "seasons": cfg.get("seasons", {}).get("active"),
    }
    serialised = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha1(serialised.encode("utf-8")).hexdigest()

# -----------------------------------------------------------------------------  
def mahalanobis_thin(df: pd.DataFrame, target_n: int, env_matrix: np.ndarray) -> pd.DataFrame:
    """Greedy farthest point sampling in Mahalanobis space.

    The distance matrix is calculated in parallel and progress is reported
    through ``tqdm``. The matrix is whitened first so we can rely on the
    optimised Euclidean implementation in ``scipy``.
    """

    n = len(df)
    env_matrix = env_matrix.astype(np.float32, copy=False)

    # Replace missing values with column means to avoid singular covariance
    # matrices caused by NaN columns. Columns that are entirely NaN are
    # replaced with zeros which effectively removes them from the distance
    # calculation while keeping dimensionality consistent.
    if np.isnan(env_matrix).any():
        col_means = np.nanmean(env_matrix, axis=0)
        col_means = np.where(np.isnan(col_means), 0, col_means)
        inds = np.where(np.isnan(env_matrix))
        env_matrix[inds] = col_means[inds[1]]

    # Instead of removing constant variables, keep the full matrix so the
    # dimensionality matches the predictor stack used for modelling. Constant
    # columns do not contribute to the distance calculation but may render the
    # covariance matrix singular.  Adding a small value to the diagonal
    # regularises the covariance matrix sufficiently so all variables can be
    # retained.
    cov = np.cov(env_matrix.T)
    cov += np.eye(cov.shape[0]) * 1e-6

    # Whiten the matrix so Euclidean distance equals Mahalanobis distance.
    # Use an eigen-decomposition which is numerically stable even for nearly
    # singular covariance matrices. Eigenvalues that are zero or negative after
    # regularisation are set to a small positive value.
    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvals[eigvals <= 0] = 1e-6
    inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
    whitened = env_matrix @ inv_sqrt

    def _row_dist(i: int) -> np.ndarray:
        return distance.cdist(whitened[i : i + 1], whitened, "euclidean")[0]

    log.info("Computing %dx%d distance matrix", n, n)
    dist_matrix = np.empty((n, n), dtype=np.float32)
    with ThreadPoolExecutor(max_workers=os.cpu_count() or 1) as ex:
        futures = {ex.submit(_row_dist, i): i for i in range(n)}
        for fut in tqdm(
            as_completed(futures),
            total=n,
            desc="distance matrix",
            dynamic_ncols=True,
        ):
            dist_matrix[futures[fut]] = fut.result()

    first = np.random.choice(n)
    selected = [first]
    min_d = dist_matrix[first].copy()
    min_d[first] = -np.inf
    for _ in trange(
        1,
        min(target_n, n),
        desc="mahalanobis thinning",
        dynamic_ncols=True,
    ):
        next_idx = int(np.argmax(min_d))
        selected.append(next_idx)
        min_d = np.minimum(min_d, dist_matrix[next_idx])
        min_d[selected] = -np.inf
    return df.iloc[selected]

# -----------------------------------------------------------------------------  
def grid_thin(df: pd.DataFrame, grid_size_km: int) -> pd.DataFrame:
    """Sample at most one random record per ``grid_size_km`` cell."""

    df = df.copy()
    df["gx"] = (df["x"] // (grid_size_km * 1000)).astype(int)
    df["gy"] = (df["y"] // (grid_size_km * 1000)).astype(int)
    # Randomly select a single observation per grid cell to reduce spatial
    # autocorrelation before the environmental thinning step.
    thinned = (
        df.groupby(["gx", "gy"], group_keys=False)
        .apply(lambda x: x.sample(n=1, random_state=np.random.default_rng().integers(0, 1e6)))
        .drop(columns=["gx", "gy"])
    )
    return thinned

# -----------------------------------------------------------------------------  
def run_thinning(cfg: dict, env_df: pd.DataFrame | None) -> gpd.GeoDataFrame:
    print("    -> Thinning presence records")
    signature = thinning_signature(cfg)
    proc_dir = ensure_dir(get_processed_dir())
    proc_file = proc_dir / f"thinned_presence_{signature}.geojson"
    proc_meta = _meta_path(proc_file)

    out_file = Path(cfg["presence_data"]["thinning"]["save_thinned_points"])
    out_meta = _meta_path(out_file)
    ensure_dir(out_file.parent)

    cached_out_sig = _read_signature(out_meta)
    if out_file.exists() and cached_out_sig != signature:
        log.info(
            "Discarding thinned presence file %s (signature %s does not match %s)",
            out_file,
            cached_out_sig,
            signature,
        )
        out_file.unlink(missing_ok=True)
        out_meta.unlink(missing_ok=True)
        cached_out_sig = None

    cached_proc_sig = _read_signature(proc_meta)
    if proc_file.exists() and cached_proc_sig != signature:
        log.info(
            "Discarding cached thinned points %s (signature %s does not match %s)",
            proc_file,
            cached_proc_sig,
            signature,
        )
        proc_file.unlink(missing_ok=True)
        proc_meta.unlink(missing_ok=True)
        cached_proc_sig = None

    def _load_existing(path: Path, context: str) -> gpd.GeoDataFrame:
        gdf_local = gpd.read_file(path)
        gdf_local = ensure_gdf_crs(gdf_local, context=context)
        from scripts.utils import load_mask, filter_points_by_mask

        mask_local = load_mask(cfg["boundary_mask"])
        df_temp = filter_points_by_mask(gdf_local.drop(columns="geometry"), mask_local)
        return gpd.GeoDataFrame(
            df_temp,
            geometry=gpd.points_from_xy(df_temp["x"], df_temp["y"]),
            crs="EPSG:2154",
        )

    if proc_file.exists():
        log.info("Using cached thinned points from %s", proc_file)
        shutil.copy2(proc_file, out_file)
        _write_signature(out_meta, signature)
        _write_signature(proc_meta, signature)
        return _load_existing(proc_file, context=f"thinned cache {proc_file}")

    if out_file.exists():
        log.info("Reusing existing thinned points from %s", out_file)
        shutil.copy2(out_file, proc_file)
        _write_signature(out_meta, signature)
        _write_signature(proc_meta, signature)
        return _load_existing(out_file, context="thinned file")

    data_file = cfg["presence_data"]["file"]
    df = pd.read_csv(data_file, encoding="utf-8-sig")
    df = df.rename(
        columns={
            "dd long": "x",
            "dd lat": "y",
            "year-int": "year",
            "month-int": "month_int",
            "source": "dataset",
        }
    )
    df["x"] = pd.to_numeric(df["x"], errors="coerce")
    df["y"] = pd.to_numeric(df["y"], errors="coerce")
    if cfg.get("seasons", {}).get("active"):
        df = df[df["semester"].isin(cfg["seasons"]["active"])]
    else:
        df = df[df["year"].between(*cfg["presence_data"]["year_filter"])]
    if cfg["presence_data"].get("source_filter"):
        df = df[df["dataset"].isin(cfg["presence_data"]["source_filter"])]
    # Remove records with missing coordinates to avoid downstream errors
    df = df.dropna(subset=["x", "y"]).reset_index(drop=True)
    from scripts.utils import load_mask, filter_points_by_mask
    mask = load_mask(cfg["boundary_mask"])
    df = filter_points_by_mask(df, mask)

    if cfg["presence_data"]["thinning"]["enable"]:
        log.info("Using combined spatial and Mahalanobis thinning")
        for method in tqdm(
            cfg["presence_data"]["thinning"]["methods"],
            desc="thinning methods",
            dynamic_ncols=True,
        ):
            if not method.get("enable", True):
                log.info("Skipping %s thinning (disabled)", method.get("name"))
                continue
            if method["name"] == "mahalanobis":
                if env_df is None:
                    raise ValueError("env_df must be provided for mahalanobis thinning")
                vars_used = method["params"].get("variables") or env_df.columns
                perc = method["params"].get("percentage")
                if isinstance(perc, dict):
                    frames = []
                    env_frames = [] if env_df is not None else None
                    remaining_mask = ~df["dataset"].isin(perc.keys())
                    for ds_name, pct in perc.items():
                        mask = df["dataset"] == ds_name
                        if not mask.any():
                            continue
                        df_sub = df[mask].reset_index(drop=True)
                        env_sub = None
                        if env_df is not None:
                            env_sub = env_df.loc[mask].reset_index(drop=True)
                        env_mat = (
                            env_sub[vars_used].to_numpy() if env_sub is not None else None
                        )
                        target_n = max(1, int(len(df_sub) * pct))
                        log.info(
                            "Mahalanobis thinning %s using %d variables - target %d points",
                            ds_name,
                            len(vars_used),
                            target_n,
                        )
                        thinned = mahalanobis_thin(df_sub, target_n, env_mat)
                        frames.append(thinned)
                        if env_sub is not None:
                            env_frames.append(
                                env_sub.loc[thinned.index].reset_index(drop=True)
                            )
                    if remaining_mask.any():
                        frames.append(df.loc[remaining_mask].reset_index(drop=True))
                        if env_df is not None:
                            env_frames.append(
                                env_df.loc[remaining_mask].reset_index(drop=True)
                            )
                    df = pd.concat(frames, ignore_index=True)
                    if env_frames is not None:
                        env_df = pd.concat(env_frames, ignore_index=True)
                else:
                    env_mat = env_df[vars_used].to_numpy()
                    if "percentage" in method["params"]:
                        target_n = max(1, int(len(df) * perc))
                    else:
                        target_n = method["params"]["target_n"]
                    log.info(
                        "Mahalanobis thinning using %d variables - target %d points",
                        len(vars_used),
                        target_n,
                    )
                    df = mahalanobis_thin(df, target_n, env_mat)
                    if env_df is not None:
                        env_df = env_df.loc[df.index].reset_index(drop=True)
                df = df.reset_index(drop=True)
            elif method["name"] == "grid":
                df = grid_thin(df, method["params"]["grid_size_km"])
                log.info("Grid thinning applied - %d points retained", len(df))
                if env_df is not None:
                    env_df = env_df.loc[df.index].reset_index(drop=True)
                df = df.reset_index(drop=True)
            else:
                raise ValueError(f"Unknown thinning method {method['name']}")

    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df["x"], df["y"]), crs="EPSG:2154")
    gdf.to_file(out_file, driver="GeoJSON")
    _write_signature(out_meta, signature)
    shutil.copy2(out_file, proc_file)
    _write_signature(proc_meta, signature)
    log.info("Thinned presence points written to %s", out_file)
    print("    -> Thinned points saved", out_file)
    return gdf
