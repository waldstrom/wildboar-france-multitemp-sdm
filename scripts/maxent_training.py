# =============================================================================
# Project Title: Wild Boar Distribution Model for France
# Path: scripts/maxent_training.py
# Purpose: Train an elapid MaxEnt model and output rasters and diagnostics.
# Process Step: Handles model fitting using environmental stacks and presence data.
# Created by Adrian Meyer, Institute Geomatics, FHNW (adrian.meyer@fhnw.ch)
# Created: July 2025
# Version: v.0.1.0
# =============================================================================
"""
Train an elapid MaxEnt model and save cloglog raster + diagnostics. The default
configuration restricts feature types to linear, quadratic and hinge and uses a
regularization multiplier of ``2.0`` to favour generalizable response curves as
recommended by Phillips & Dudík (2008), Elith et al. (2011) and
Radosavljevic & Anderson (2014).
"""
from __future__ import annotations
import os
import logging
from pathlib import Path
import inspect
from concurrent.futures import ThreadPoolExecutor, as_completed
import geopandas as gpd
import numpy as np
import rasterio
from rasterio import Affine
from elapid import (
    MaxentModel,
    distance_weights,
    stack_geodataframes,
)
from sklearn.metrics import roc_curve
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scripts.utils import ensure_dir, sanitize_name
from scripts.visualization import plot_dropped_points
from tqdm.auto import tqdm
try:
    import psutil  # type: ignore
except Exception:  # pragma: no cover - psutil is optional
    psutil = None

# Default to all available CPUs unless overridden in the config.
_DEFAULT_THREADS = os.cpu_count() or 1
for _ev in [
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
]:
    os.environ[_ev] = str(_DEFAULT_THREADS)

log = logging.getLogger(__name__)

# -----------------------------------------------------------------------------  
def train_model(
    cfg: dict,
    stack_path: Path,
    vars_used: list[str],
    pres_gdf: gpd.GeoDataFrame,
    bg_gdf: gpd.GeoDataFrame,
) -> tuple[MaxentModel, Path]:
    print("    -> Training MaxEnt model")
    threads = cfg.setdefault("maxent", {}).get("threads", _DEFAULT_THREADS)
    if threads == -1:
        threads = os.cpu_count() or 1
    cfg["maxent"]["threads"] = threads
    for _ev in [
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "BLIS_NUM_THREADS",
    ]:
        os.environ[_ev] = str(threads)
    from scripts.utils import extract_from_stack
    import xarray as xr
    
    def _check_memory(array: np.ndarray, factor: int = 2) -> None:
        """Raise RuntimeError if available memory is insufficient."""
        if psutil is None:
            return
        required = array.nbytes * factor
        available = psutil.virtual_memory().available
        if required > available:
            raise RuntimeError(
                f"Insufficient system memory: ~{required/1e9:.2f}GB required "
                f"but only {available/1e9:.2f}GB available."
            )

    def _log_predictor_stats(matrix: np.ndarray, names: list[str]) -> None:
        for idx, name in enumerate(names):
            col = matrix[:, idx]
            n_missing = int(np.isnan(col).sum())
            unique_vals = np.unique(col[~np.isnan(col)])
            if n_missing == len(col):
                log.error("Predictor %s has only missing values", name)
            elif len(unique_vals) <= 1:
                val = unique_vals[0] if len(unique_vals) == 1 else "NaN"
                log.warning("Predictor %s is constant (%s)", name, val)
            log.debug(
                "Predictor %s - missing: %d unique: %d", name, n_missing, len(unique_vals)
            )

    def _safe_distance_weights(gdf: gpd.GeoDataFrame, label: str) -> np.ndarray:
        """Return distance-based weights without exhausting system memory."""
        n = len(gdf)
        # Heuristic: full pairwise distances require ~n^2*8 bytes
        if psutil is not None:
            required = n * n * 8
            available = psutil.virtual_memory().available
            if required > available:
                log.warning(
                    "Skipping distance weights for %s (%d records): ~%.2fGB required, %.2fGB available",
                    label,
                    n,
                    required / 1e9,
                    available / 1e9,
                )
                return np.ones(n)
        # Limit neighbors for large datasets to avoid dense distance matrices
        n_neighbors = -1 if n < 10000 else min(64, n - 1)
        try:
            return distance_weights(gdf, n_neighbors=n_neighbors)
        except MemoryError:
            log.warning(
                "distance_weights failed for %s with %d records; using uniform weights",
                label,
                n,
            )
            return np.ones(n)

    if any(v in {"x", "y"} for v in vars_used):
        log.info(
            "Dropping coordinate predictors: %s",
            ", ".join(v for v in vars_used if v in {"x", "y"}),
        )
        vars_used = [v for v in vars_used if v not in {"x", "y"}]

    log.info("Training with predictors: %s", ", ".join(vars_used))
    if not vars_used:
        raise RuntimeError(
            "No predictors were provided; 'vars_used' is empty."
        )
    log.info("Preparing training data")
    merged = stack_geodataframes(pres_gdf, bg_gdf, add_class_label=True)
    X = extract_from_stack(stack_path, merged, vars_used)
    log.debug("Initial training matrix shape: %s", X.shape)
    _log_predictor_stats(X, vars_used)
    mask_valid = ~np.any(np.isnan(X), axis=1)
    if not np.any(mask_valid):
        missing_counts = np.isnan(X).sum(axis=0)
        bad_vars = [name for name, cnt in zip(vars_used, missing_counts) if cnt == X.shape[0]]
        log.error(
            "All training records contain missing values. Empty predictors: %s",
            ", ".join(bad_vars),
        )
        raise RuntimeError(
            "No valid training samples remain after filtering missing values.",
        )
    if not np.all(mask_valid):
        dropped = int(np.sum(~mask_valid))
        log.info("Dropping %d records with missing predictor values", dropped)
        missing_counts = np.isnan(X[~mask_valid]).sum(axis=0)
        for name, cnt in zip(vars_used, missing_counts):
            if cnt > 0:
                log.debug("%s caused %d drops", name, int(cnt))
        dropped_gdf = merged.loc[~mask_valid]
        vis_dir = Path(cfg["experiment"]["vis_dir"])
        vis_file = vis_dir / "step6_dropped_points.png"
        gpkg_file = vis_dir / "step6_dropped_points.gpkg"
        try:
            plot_dropped_points(dropped_gdf, vis_file)
            log.info("Plot of dropped points saved to %s", vis_file)
            dropped_gdf.to_file(gpkg_file, driver="GPKG")
            log.info("Dropped points exported to %s", gpkg_file)
        except Exception as exc:
            log.warning("Failed to plot/export dropped records: %s", exc)
        merged = merged.iloc[mask_valid].reset_index(drop=True)
        X = X[mask_valid]
    _check_memory(X)
    y = merged["class"].to_numpy()
    log.info(
        "Training matrix prepared: %d samples, %d predictors", X.shape[0], X.shape[1]
    )
    if X.shape[1] == 0:
        raise RuntimeError(
            "No predictor columns were extracted. Check 'vars_used' and stack contents."
        )

    # Drop predictors with zero variance which would cause elapid to fail
    variances = np.nanvar(X, axis=0)
    keep_cols = variances > 0
    if not np.any(keep_cols):
        raise RuntimeError(
            "All predictors have zero variance. Check extracted values."
        )
    if not np.all(keep_cols):
        dropped = [v for v, k in zip(vars_used, keep_cols) if not k]
        log.info(
            "Dropping %d constant predictors: %s", len(dropped), ", ".join(dropped)
        )
        X = X[:, keep_cols]
        vars_used = [v for v, k in zip(vars_used, keep_cols) if k]
    log.debug("Training matrix shape after cleaning: %s", X.shape)

    _log_predictor_stats(X, vars_used)

    if len(vars_used) == 0:
        log.error("No predictors remain after dropping constant columns")
        raise RuntimeError(
            "No predictors remain after dropping constant columns. Check extracted values and variable selection."
        )

    # Down-weight clustered occurrence records to further correct for
    # sampling bias using distance-based weights (Phillips & Dudík, 2008).
    w_pres = _safe_distance_weights(pres_gdf, "presences")
    w_bg = _safe_distance_weights(bg_gdf, "background")
    sample_weight = np.concatenate([w_pres, w_bg])
    if not np.all(mask_valid):
        sample_weight = sample_weight[mask_valid]

    ft = list(cfg["maxent"].get("feature_types", []))
    if "product_features" in cfg["maxent"]:
        if cfg["maxent"]["product_features"] and "product" not in ft:
            ft.append("product")
        elif not cfg["maxent"]["product_features"]:
            ft = [f for f in ft if f != "product"]

    n_hinge = cfg["maxent"].get("n_hinge_features")
    if n_hinge == "auto":
        n_hinge = None
    maxent_kwargs = {
        "feature_types": ft or None,
        "beta_multiplier": cfg["maxent"]["regularization_multiplier"],
        "n_cpus": cfg["maxent"]["threads"],
        "transform": cfg["maxent"]["output_type"],
        "n_lambdas": cfg["maxent"].get("max_iterations", 1000),
        "convergence_tolerance": cfg["maxent"].get("convergence_tolerance", 1e-5),
    }
    if n_hinge is not None:
        maxent_kwargs["n_hinge_features"] = n_hinge
    beta_lqp = cfg["maxent"].get("beta_lqp")
    if beta_lqp is not None and "beta_lqp" in inspect.signature(MaxentModel.__init__).parameters:
        maxent_kwargs["beta_lqp"] = beta_lqp

    maxent = MaxentModel(**maxent_kwargs)
    log.info(
        "Fitting MaxEnt - %d presences, %d background, %d vars",
        len(pres_gdf),
        len(bg_gdf),
        len(vars_used),
    )
    fit_kwargs = {"sample_weight": sample_weight, "labels": vars_used}
    if "verbose" in inspect.signature(maxent.fit).parameters:
        fit_kwargs["verbose"] = True
    try:
        maxent.fit(X, y, **fit_kwargs)
    except MemoryError as exc:
        log.error("MaxEnt training failed due to insufficient memory: %s", exc)
        raise RuntimeError(
            "MaxEnt training aborted because available memory was insufficient."
        ) from exc
    except ValueError as exc:
        log.error("MaxEnt training failed: %s", exc)
        log.warning("Retrying with only linear features")
        maxent_kwargs["feature_types"] = ["linear"]
        maxent = MaxentModel(**maxent_kwargs)
        fit_kwargs = {"sample_weight": sample_weight, "labels": vars_used}
        if "verbose" in inspect.signature(maxent.fit).parameters:
            fit_kwargs["verbose"] = True
        try:
            maxent.fit(X, y, **fit_kwargs)
        except Exception as exc2:
            log.error("Fallback training also failed: %s", exc2)
            raise RuntimeError(
                "MaxEnt training failed even after retrying with linear features"
            ) from exc2

    # Compute ROC curve on training data if not provided by elapid
    if not hasattr(maxent, "roc_curve_") or maxent.roc_curve_ is None:
        try:
            y_pred_train = maxent.predict(X)
            mask = ~np.isnan(y_pred_train)
            if np.sum(mask) < 2 or len(np.unique(y[mask])) < 2:
                raise ValueError("Insufficient data for ROC curve")
            fpr, tpr, _ = roc_curve(y[mask], y_pred_train[mask])
            maxent.roc_curve_ = (fpr, tpr)
            log.debug("ROC curve computed after training")
        except Exception as exc:
            log.warning("Failed to compute ROC curve: %s", exc)
            maxent.roc_curve_ = None


    # Predict raster over the full stack using row-wise batching to reduce memory
    # footprint. Disable consolidated metadata lookup to suppress warnings.
    log.info("Generating prediction raster")
    n_workers = cfg["maxent"].get("threads", _DEFAULT_THREADS)
    cpu_count = os.cpu_count() or 1
    if n_workers == -1 or n_workers < 1:
        n_workers = cpu_count
    if n_workers > cpu_count:
        log.warning(
            "Requested %d threads but only %d CPUs available; reducing to %d",
            n_workers,
            cpu_count,
            cpu_count,
        )
        n_workers = cpu_count

    # Limit the internal thread usage of numerical libraries so the combined
    # threads from all workers do not exceed the available CPU cores.
    blas_threads = max(1, cpu_count // n_workers)
    if blas_threads < int(os.environ.get("OMP_NUM_THREADS", blas_threads)):
        log.info("Adjusting internal threads to %d per worker", blas_threads)
    for _ev in [
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "BLIS_NUM_THREADS",
    ]:
        os.environ[_ev] = str(blas_threads)

    cfg["maxent"]["threads"] = n_workers

    chunk_size = int(1e5)
    row_chunk = cfg["maxent"].get("prediction_row_chunk", 512)

    def _predict_block(grid_block: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Predict suitability for a flattened grid block."""
        preds_block = np.full(mask.size, np.nan, dtype="float32")
        valid = np.flatnonzero(mask.ravel())
        if valid.size == 0:
            return preds_block.reshape(mask.shape)

        def _predict_chunk(start: int) -> tuple[np.ndarray, np.ndarray]:
            end = start + chunk_size
            idx = valid[start:end]
            return idx, maxent.predict(grid_block[idx])

        if n_workers == 1:
            for start in range(0, valid.size, chunk_size):
                idx, res = _predict_chunk(start)
                preds_block[idx] = res
        else:
            with ThreadPoolExecutor(max_workers=n_workers) as ex:
                futures = [
                    ex.submit(_predict_chunk, start)
                    for start in range(0, valid.size, chunk_size)
                ]
                for fut in as_completed(futures):
                    idx, res = fut.result()
                    preds_block[idx] = res
        return preds_block.reshape(mask.shape)

    if stack_path.is_dir() and any(stack_path.glob("*.tif")):
        srcs = {
            var: rasterio.open(stack_path / f"{var}.tif")
            for var in vars_used + ["boundary_mask"]
        }
        src_meta = srcs["boundary_mask"].meta.copy()
        height = src_meta["height"]
        width = src_meta["width"]
        transform = src_meta["transform"]
        suitability = np.full((height, width), np.nan, dtype="float32")
        for row_start in tqdm(
            range(0, height, row_chunk),
            desc="Predict raster",
            unit="row",
            leave=False,
        ):
            row_stop = min(row_start + row_chunk, height)
            window = rasterio.windows.Window(0, row_start, width, row_stop - row_start)
            arrays = [srcs[v].read(1, window=window).astype("float32") for v in vars_used]
            mask = srcs["boundary_mask"].read(1, window=window).astype(bool)
            grid_block = np.column_stack([a.ravel() for a in arrays])
            for i in range(grid_block.shape[1]):
                col = grid_block[:, i]
                m = np.nanmean(col)
                col[np.isnan(col)] = m
                grid_block[:, i] = col
            pred_block = _predict_block(grid_block, mask)
            suitability[row_start:row_stop, :] = pred_block
        for src in srcs.values():
            src.close()
    else:
        ds = xr.open_zarr(stack_path, consolidated=False)
        height = ds.sizes["y"]
        width = ds.sizes["x"]
        suitability = np.full((height, width), np.nan, dtype="float32")
        for row_start in tqdm(
            range(0, height, row_chunk),
            desc="Predict raster",
            unit="row",
            leave=False,
        ):
            row_stop = min(row_start + row_chunk, height)
            arrays = [
                ds[v].isel(y=slice(row_start, row_stop)).values.astype("float32")
                for v in vars_used
            ]
            mask = (
                ds["boundary_mask"].isel(y=slice(row_start, row_stop)).values.astype(bool)
            )
            grid_block = np.column_stack([a.ravel() for a in arrays])
            for i in range(grid_block.shape[1]):
                col = grid_block[:, i]
                m = np.nanmean(col)
                col[np.isnan(col)] = m
                grid_block[:, i] = col
            pred_block = _predict_block(grid_block, mask)
            suitability[row_start:row_stop, :] = pred_block
        ds.close()
    safe_name = sanitize_name(cfg["experiment"]["name"])
    tif_out = Path(cfg["outputs"]["maps_dir"]) / f"{safe_name}_cloglog.tif"
    ensure_dir(tif_out.parent)
    if stack_path.is_dir() and any(stack_path.glob("*.tif")):
        transform = src_meta["transform"]
        crs = src_meta.get("crs", "EPSG:2154")
    else:
        x_coords = ds.x.values
        y_coords = ds.y.values
        res_x = float(abs(x_coords[1] - x_coords[0]))
        res_y = float(abs(y_coords[1] - y_coords[0]))
        transform = rasterio.transform.from_origin(
            float(x_coords.min()), float(y_coords.max()), res_x, res_y
        )
        crs = "EPSG:2154"

    # ``rasterio`` uses the upper-left corner of the upper-left pixel for the
    # affine transform.  Historically our preprocessing defined the grid using
    # cell centres which resulted in a half-pixel (≈500 m) shift of generated
    # cloglog rasters.  Compensate for this by nudging the transform half a pixel
    # north-west so the raster aligns with the input layers.
    transform = transform * Affine.translation(-0.5, -0.5)

    meta = {
        "driver": "GTiff",
        "height": suitability.shape[0],
        "width": suitability.shape[1],
        "count": 1,
        "dtype": "float32",
        "crs": crs,
        "transform": transform,
        "nodata": np.nan,
        "compress": cfg["maxent"].get("geotiff_compression", "deflate"),
    }
    with rasterio.open(tif_out, "w", **meta) as dst:
        dst.write(suitability.astype("float32"), 1)

        # AUC curve
        fig, ax = plt.subplots()
        if hasattr(maxent, "roc_curve_") and maxent.roc_curve_ is not None:
            ax.plot(*maxent.roc_curve_)
        else:
            log.warning("ROC curve data missing; skipping plot")
        ax.set_xlabel("1 - Specificity")
        ax.set_ylabel("Sensitivity")
        fig_dir = Path(cfg["outputs"]["figures_dir"])
        ensure_dir(fig_dir)
        auc_file = fig_dir / f"{safe_name}_auc.png"
        fig.savefig(auc_file, dpi=150)
        plt.close(fig)
        log.info("AUC curve saved to %s", auc_file)

        # Compute and display evaluation metrics on the training data
        try:
            from scripts.evaluation import _compute_metrics  # type: ignore

            y_score_train = maxent.predict(X)
            metrics = _compute_metrics(y, y_score_train, threshold="auto")
            log.info(
                "Training metrics - AUC %.3f CBI %.3f", metrics["auc"], metrics["cbi"]
            )
            print(
                f"    -> Train AUC {metrics['auc']:.3f} CBI {metrics['cbi']:.3f}"
            )
        except Exception as exc:  # pragma: no cover - evaluation is best-effort
            log.warning("Failed to compute training metrics: %s", exc)

        # Report gain development if the model exposes it
        gain_hist = None
        for attr in (
            "gain_history_",
            "training_log_",
            "gains_",
            "gain_trajectory_",
        ):
            if hasattr(maxent, attr):
                gain_hist = np.asarray(getattr(maxent, attr)).ravel()
                break
        if gain_hist is not None and gain_hist.size > 0:
            step = max(1, gain_hist.size // 10)
            for idx in range(0, gain_hist.size, step):
                log.info("Gain after iter %d: %.6f", idx + 1, gain_hist[idx])
                print(f"    -> Gain iter {idx + 1}: {gain_hist[idx]:.6f}")

        # Print summary of hyper-parameters used
        hyper_params = {
            "feature_types": maxent.feature_types,
            "regularization_multiplier": cfg["maxent"]["regularization_multiplier"],
            "output_type": cfg["maxent"]["output_type"],
            "max_iterations": cfg["maxent"].get("max_iterations", 1000),
            "convergence_tolerance": cfg["maxent"].get("convergence_tolerance", 1e-5),
            "threads": cfg["maxent"]["threads"],
        }
        log.info("Hyperparameters: %s", hyper_params)
        print(
            "    -> Hyperparameters: "
            + ", ".join(f"{k}={v}" for k, v in hyper_params.items())
        )

        log.info("Model training done - cloglog map saved to %s", tif_out)
        print("    -> Model saved", tif_out)
        return maxent, tif_out, vars_used
