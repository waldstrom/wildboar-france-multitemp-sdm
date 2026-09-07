# =============================================================================
# Project Title: Wild Boar Distribution Model for France
# Path: scripts/gbm_training.py
# Purpose: Train a gradient boosting SDM and export suitability maps.
# Process Step: Mirrors ``maxent_training`` but relies on LightGBM to provide a
#               tree-based alternative to the elapid MaxEnt implementation.
# Created by Adrian Meyer, Institute Geomatics, FHNW (adrian.meyer@fhnw.ch)
# Created: July 2025
# Version: v.0.2.0
# =============================================================================
"""Gradient boosting species distribution model training."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from dataclasses import dataclass
from typing import Literal

import geopandas as gpd
import numpy as np
import rasterio
from rasterio import Affine
import lightgbm as lgb
from sklearn.metrics import roc_curve
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from elapid import stack_geodataframes

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:  # pragma: no cover - psutil is optional
    import psutil  # type: ignore
except Exception:  # pragma: no cover - graceful degradation
    psutil = None

from scripts.utils import ensure_dir, sanitize_name, extract_from_stack
from scripts.visualization import plot_dropped_points

log = logging.getLogger(__name__)


def _sigmoid(values: np.ndarray) -> np.ndarray:
    """Numerically stable logistic function."""

    arr = np.asarray(values, dtype=float)
    with np.errstate(over="ignore"):
        out = 1.0 / (1.0 + np.exp(-arr))
    return out


def _summarise_scores(scores: np.ndarray) -> dict[str, float]:
    """Return descriptive statistics for the given score array."""

    arr = np.asarray(scores, dtype=float)
    if arr.size == 0:
        return {}
    return {
        "min": float(np.nanmin(arr)),
        "p05": float(np.nanpercentile(arr, 5)),
        "median": float(np.nanmedian(arr)),
        "p95": float(np.nanpercentile(arr, 95)),
        "max": float(np.nanmax(arr)),
        "mean": float(np.nanmean(arr)),
    }


def _log_distribution(label: str, scores: np.ndarray) -> None:
    """Log score distribution for traceability."""

    stats = _summarise_scores(scores)
    if not stats:
        log.info("%s probability distribution unavailable (no samples)", label)
        return
    log.info(
        "%s probability distribution - min %.3f | p05 %.3f | median %.3f | p95 %.3f | max %.3f | mean %.3f",
        label,
        stats["min"],
        stats["p05"],
        stats["median"],
        stats["p95"],
        stats["max"],
        stats["mean"],
    )


def _apply_calibration(calibration: dict[str, float] | None, raw_scores: np.ndarray) -> np.ndarray:
    """Apply Platt-style calibration parameters to raw LightGBM scores."""

    if not calibration:
        return _sigmoid(raw_scores)
    if calibration.get("type") != "platt":
        raise ValueError(f"Unsupported calibration type '{calibration.get('type')}'")
    logits = calibration["coef"] * raw_scores + calibration["intercept"]
    return _sigmoid(logits)


def _fit_calibration(
    method: Literal["platt"],
    raw_scores: np.ndarray,
    y_true: np.ndarray,
    sample_weight: np.ndarray | None,
    regularization: float = 1e6,
) -> dict[str, float] | None:
    """Return calibration parameters for the requested method."""

    if raw_scores.size == 0:
        return None
    method = method.lower()
    if method != "platt":
        raise ValueError(f"Unsupported calibration method '{method}'")
    lr = LogisticRegression(
        solver="lbfgs",
        C=float(regularization),
        fit_intercept=True,
        max_iter=1000,
    )
    lr.fit(raw_scores.reshape(-1, 1), y_true, sample_weight=sample_weight)
    coef = float(lr.coef_[0, 0])
    intercept = float(lr.intercept_[0])
    return {"type": "platt", "coef": coef, "intercept": intercept}


@dataclass
class GBMModel:
    """Thin wrapper exposing a LightGBM booster with calibration support."""

    booster: lgb.Booster
    roc_curve_: tuple[np.ndarray, np.ndarray] | None = None
    calibration_: dict[str, float] | None = None

    def predict(self, X: np.ndarray, *, raw_score: bool = False) -> np.ndarray:
        data = np.asarray(X, dtype=float)
        booster = object.__getattribute__(self, "booster")
        best_iter = booster.best_iteration or booster.current_iteration()
        if raw_score:
            return booster.predict(
                data,
                num_iteration=best_iter or booster.num_trees(),
                raw_score=True,
            )

        if self.calibration_ is not None:
            raw = booster.predict(
                data,
                num_iteration=best_iter or booster.num_trees(),
                raw_score=True,
            )
            return _apply_calibration(self.calibration_, raw)

        return booster.predict(data, num_iteration=best_iter or booster.num_trees())

    def set_calibration(self, calibration: dict[str, float] | None) -> None:
        object.__setattr__(self, "calibration_", calibration)

    def __getattr__(self, item):  # pragma: no cover - passthrough convenience
        booster = object.__getattribute__(self, "booster")
        try:
            return getattr(booster, item)
        except AttributeError as exc:  # pragma: no cover - mirror AttributeError
            raise AttributeError(f"{type(self).__name__} has no attribute {item!r}") from exc

    def __getstate__(self) -> dict:
        booster = object.__getattribute__(self, "booster")
        return {
            "booster_str": booster.model_to_string(),
            "roc_curve_": self.roc_curve_,
            "calibration_": self.calibration_,
        }

    def __setstate__(self, state: dict) -> None:
        booster = lgb.Booster(model_str=state["booster_str"])
        object.__setattr__(self, "booster", booster)
        object.__setattr__(self, "roc_curve_", state.get("roc_curve_"))
        object.__setattr__(self, "calibration_", state.get("calibration_"))


def _fill_missing_with_mean(column: np.ndarray, fallback: float = 0.0) -> None:
    """Replace ``NaN`` entries with the column mean or a fallback value."""

    with np.errstate(invalid="ignore", divide="ignore"):
        mean_val = np.nanmean(column)
    if np.isnan(mean_val):
        mean_val = fallback
    mask = np.isnan(column)
    if np.any(mask):
        column[mask] = mean_val


def _safe_distance_weights(gdf: gpd.GeoDataFrame, label: str) -> np.ndarray:
    """Return distance-based weights without exhausting system memory."""

    from elapid import distance_weights

    n = len(gdf)
    if n == 0:
        return np.ones(0)
    required = n * n * 8
    if psutil is not None:
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
    else:
        available = getattr(os, "sysconf", lambda x: 0)("SC_PAGE_SIZE") * getattr(
            os, "sysconf", lambda x: 0
        )("SC_AVPHYS_PAGES")
        if available and required > available:
            log.warning(
                "Skipping distance weights for %s (%d records): ~%.2fGB required",
                label,
                n,
                required / 1e9,
            )
            return np.ones(n)
    try:
        n_neighbors = -1 if n < 10000 else min(64, n - 1)
        return distance_weights(gdf, n_neighbors=n_neighbors)
    except MemoryError:
        log.warning(
            "distance_weights failed for %s with %d records; using uniform weights",
            label,
            n,
        )
        return np.ones(n)


def _prepare_training_matrix(
    cfg: dict,
    stack_path: Path,
    vars_used: list[str],
    pres_gdf: gpd.GeoDataFrame,
    bg_gdf: gpd.GeoDataFrame,
) -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray, np.ndarray]:
    """Return ``(X, y, vars_kept, weights, merged_gdf)`` for GBM training."""

    vars_clean = [v for v in vars_used if v not in {"x", "y"}]
    if not vars_clean:
        raise RuntimeError("No predictors were provided; 'vars_used' is empty.")

    merged = stack_geodataframes(pres_gdf, bg_gdf, add_class_label=True)
    X = extract_from_stack(stack_path, merged, vars_clean)
    mask_valid = ~np.any(np.isnan(X), axis=1)
    if not np.all(mask_valid):
        dropped = int(np.sum(~mask_valid))
        log.info("Dropping %d records with missing predictor values", dropped)
        dropped_gdf = merged.loc[~mask_valid]
        vis_dir = Path(cfg["experiment"]["vis_dir"])
        vis_file = vis_dir / "step6_dropped_points.png"
        gpkg_file = vis_dir / "step6_dropped_points.gpkg"
        try:
            plot_dropped_points(dropped_gdf, vis_file)
            dropped_gdf.to_file(gpkg_file, driver="GPKG")
        except Exception as exc:  # pragma: no cover - best effort only
            log.warning("Failed to export dropped records: %s", exc)
        merged = merged.loc[mask_valid].reset_index(drop=True)
        X = X[mask_valid]

    variances = np.nanvar(X, axis=0)
    keep_cols = variances > 0
    if not np.any(keep_cols):
        raise RuntimeError("All predictors have zero variance. Check extracted values.")
    if not np.all(keep_cols):
        removed = [v for v, k in zip(vars_clean, keep_cols) if not k]
        log.info("Dropping %d constant predictors: %s", len(removed), ", ".join(removed))
        X = X[:, keep_cols]
        vars_clean = [v for v, k in zip(vars_clean, keep_cols) if k]

    w_pres = _safe_distance_weights(pres_gdf, "presences")
    w_bg = _safe_distance_weights(bg_gdf, "background")
    weights = np.concatenate([w_pres, w_bg])
    weights = weights[mask_valid]
    y = merged["class"].to_numpy()

    return X, y, vars_clean, weights, merged


def _gbm_params(cfg: dict) -> dict:
    params = cfg.setdefault("gbm", {})
    defaults = {
        "boosting_type": "gbdt",
        "objective": "binary",
        "metric": ["auc"],
        "learning_rate": 0.05,
        "num_leaves": 63,
        "max_depth": -1,
        "min_data_in_leaf": 40,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.7,
        "bagging_freq": 1,
        "lambda_l1": 0.0,
        "lambda_l2": 1.0,
        "min_gain_to_split": 0.01,
        "num_boost_round": 2000,
        "early_stopping_rounds": 100,
        "validation_fraction": 0.1,
        "verbose_eval": 100,
    }
    for key, value in defaults.items():
        params.setdefault(key, value)
    return params


def train_model(
    cfg: dict,
    stack_path: Path,
    vars_used: list[str],
    pres_gdf: gpd.GeoDataFrame,
    bg_gdf: gpd.GeoDataFrame,
) -> tuple[GBMModel, Path, list[str]]:
    """Train gradient boosting classifier and export raster prediction."""

    print("    -> Training GBM model")
    log.info("Training GBM model with %d predictors", len(vars_used))

    X, y, vars_clean, weights, merged = _prepare_training_matrix(
        cfg, stack_path, list(vars_used), pres_gdf, bg_gdf
    )

    model, history = fit_lightgbm(cfg, X, y, vars_clean, weights)

    calibration_cfg = cfg.get("gbm", {}).get("calibration", {})
    calibration = None
    if calibration_cfg.get("enable", False):
        dataset_choice = str(calibration_cfg.get("dataset", "validation")).lower()
        candidate = history.get("valid") if dataset_choice.startswith("val") else None
        if candidate is None:
            candidate = history.get("train")
            dataset_choice = "training"
        if candidate is not None:
            calibration = _fit_calibration(
                method=str(calibration_cfg.get("method", "platt")),
                raw_scores=candidate["raw_scores"],
                y_true=candidate["y"],
                sample_weight=candidate.get("weights"),
                regularization=float(calibration_cfg.get("regularization", 1e6)),
            )
            if calibration is not None:
                model.set_calibration(calibration)
                log.info(
                    "Applied %s calibration using %s data",
                    calibration.get("type", "platt"),
                    dataset_choice,
                )
            else:
                log.warning("Calibration fitting returned no parameters; skipping")
        else:
            log.warning("Calibration requested but no suitable dataset was available")

    preds = model.predict(X)
    try:
        model.roc_curve_ = roc_curve(y, preds)[:2]
    except Exception as exc:  # pragma: no cover - defensive
        log.warning("Failed to compute ROC curve for GBM: %s", exc)
        model.roc_curve_ = None

    # Inspect score distribution for comparability with MaxEnt outputs
    train_context = history.get("train")
    if train_context is not None:
        raw_train = train_context.get("raw_scores")
        if raw_train is not None:
            _log_distribution("GBM train (raw sigmoid)", _sigmoid(raw_train))
        train_probs = model.predict(train_context["X"])
        mask_pres = train_context["y"] == 1
        _log_distribution("GBM train presences", train_probs[mask_pres])
        _log_distribution("GBM train background", train_probs[~mask_pres])

    valid_context = history.get("valid")
    if valid_context is not None and valid_context.get("X") is not None:
        valid_probs = model.predict(valid_context["X"])
        mask_pres = valid_context["y"] == 1
        _log_distribution("GBM validation presences", valid_probs[mask_pres])
        _log_distribution("GBM validation background", valid_probs[~mask_pres])

    safe_name = sanitize_name(cfg["experiment"]["name"])
    tif_out = Path(cfg["outputs"]["maps_dir"]) / f"{safe_name}_gbm_probability.tif"
    ensure_dir(tif_out.parent)

    def _predict_block(grid_block: np.ndarray, mask: np.ndarray) -> np.ndarray:
        scores = np.full(mask.size, np.nan, dtype="float32")
        valid = np.flatnonzero(mask.ravel())
        if valid.size == 0:
            return scores.reshape(mask.shape)
        batch = 100000
        for start in range(0, valid.size, batch):
            idx = valid[start : start + batch]
            scores[idx] = model.predict(grid_block[idx])
        return scores.reshape(mask.shape)

    if stack_path.is_dir() and any(stack_path.glob("*.tif")):
        srcs = {
            var: rasterio.open(stack_path / f"{var}.tif")
            for var in vars_clean + ["boundary_mask"]
        }
        meta = srcs["boundary_mask"].meta.copy()
        suitability = np.full((meta["height"], meta["width"]), np.nan, dtype="float32")
        row_chunk = int(cfg.get("gbm", {}).get("prediction_row_chunk", 512))
        for row_start in range(0, meta["height"], row_chunk):
            row_stop = min(row_start + row_chunk, meta["height"])
            window = rasterio.windows.Window(0, row_start, meta["width"], row_stop - row_start)
            arrays = [
                srcs[v].read(1, window=window).astype("float32") for v in vars_clean
            ]
            mask = srcs["boundary_mask"].read(1, window=window).astype(bool)
            grid_block = np.column_stack([a.ravel() for a in arrays])
            for i in range(grid_block.shape[1]):
                col = grid_block[:, i]
                _fill_missing_with_mean(col)
                grid_block[:, i] = col
            suitability[row_start:row_stop, :] = _predict_block(grid_block, mask)
        for src in srcs.values():
            src.close()
        transform = meta["transform"]
        crs = meta.get("crs", "EPSG:2154")
    else:
        import xarray as xr

        ds = xr.open_zarr(stack_path, consolidated=False)
        row_chunk = int(cfg.get("gbm", {}).get("prediction_row_chunk", 512))
        suitability = np.full((ds.sizes["y"], ds.sizes["x"]), np.nan, dtype="float32")
        for row_start in range(0, ds.sizes["y"], row_chunk):
            row_stop = min(row_start + row_chunk, ds.sizes["y"])
            arrays = [
                ds[v].isel(y=slice(row_start, row_stop)).values.astype("float32")
                for v in vars_clean
            ]
            mask = ds["boundary_mask"].isel(y=slice(row_start, row_stop)).values.astype(bool)
            grid_block = np.column_stack([a.ravel() for a in arrays])
            for i in range(grid_block.shape[1]):
                col = grid_block[:, i]
                _fill_missing_with_mean(col)
                grid_block[:, i] = col
            suitability[row_start:row_stop, :] = _predict_block(grid_block, mask)
        transform = rasterio.transform.from_origin(
            float(ds.x.values.min()),
            float(ds.y.values.max()),
            float(abs(ds.x.values[1] - ds.x.values[0])),
            float(abs(ds.y.values[1] - ds.y.values[0])),
        )
        crs = "EPSG:2154"
        ds.close()

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
        "compress": cfg.get("gbm", {}).get("geotiff_compression", "deflate"),
    }
    with rasterio.open(tif_out, "w", **meta) as dst:
        dst.write(suitability.astype("float32"), 1)

    fig_dir = Path(cfg["outputs"]["figures_dir"])
    ensure_dir(fig_dir)
    auc_file = fig_dir / f"{safe_name}_auc.png"
    if model.roc_curve_ is not None:
        fpr, tpr = model.roc_curve_
        fig, ax = plt.subplots()
        ax.plot(fpr, tpr)
        ax.set_xlabel("1 - Specificity")
        ax.set_ylabel("Sensitivity")
        fig.savefig(auc_file, dpi=150)
        plt.close(fig)

    try:
        from scripts.evaluation import _compute_metrics  # type: ignore

        metrics = _compute_metrics(y, preds, threshold="auto")
        log.info(
            "GBM aggregated training metrics - AUC %.3f CBI %.3f",
            metrics.get("auc", float("nan")),
            metrics.get("cbi", float("nan")),
        )
        print(
            "    -> Train AUC "
            f"{metrics.get('auc', float('nan')):.3f} CBI {metrics.get('cbi', float('nan')):.3f}"
        )

        if train_context is not None:
            metrics_train = _compute_metrics(
                train_context["y"],
                model.predict(train_context["X"]),
                threshold="auto",
            )
            log.info(
                "GBM split metrics (train) - AUC %.3f CBI %.3f",
                metrics_train.get("auc", float("nan")),
                metrics_train.get("cbi", float("nan")),
            )

        if valid_context is not None:
            metrics_valid = _compute_metrics(
                valid_context["y"],
                model.predict(valid_context["X"]),
                threshold="auto",
            )
            log.info(
                "GBM split metrics (validation) - AUC %.3f CBI %.3f",
                metrics_valid.get("auc", float("nan")),
                metrics_valid.get("cbi", float("nan")),
            )
    except Exception as exc:  # pragma: no cover - evaluation best effort
        log.warning("Failed to compute GBM training metrics: %s", exc)

    return model, tif_out, vars_clean


def fit_lightgbm(
    cfg: dict,
    X: np.ndarray,
    y: np.ndarray,
    vars_clean: list[str],
    weights: np.ndarray,
) -> GBMModel:
    """Fit a LightGBM model and return a ``GBMModel`` wrapper."""

    params = _gbm_params(cfg)
    random_state = cfg["experiment"].get("random_seed", 0)
    val_frac = float(params.get("validation_fraction", 0.0))

    train_context = {
        "X": X,
        "y": y,
        "weights": weights,
    }
    valid_context: dict | None = None

    if val_frac > 0 and val_frac < 1:
        (
            X_train,
            X_valid,
            y_train,
            y_valid,
            w_train,
            w_valid,
        ) = train_test_split(
            X,
            y,
            weights,
            test_size=val_frac,
            stratify=y,
            random_state=random_state,
        )
        train_context = {"X": X_train, "y": y_train, "weights": w_train}
        valid_context = {"X": X_valid, "y": y_valid, "weights": w_valid}
        train_set = lgb.Dataset(
            X_train,
            label=y_train,
            weight=w_train,
            feature_name=vars_clean,
            free_raw_data=False,
        )
        valid_set = lgb.Dataset(
            X_valid,
            label=y_valid,
            weight=w_valid,
            feature_name=vars_clean,
            free_raw_data=False,
        )
        valid_sets = [train_set, valid_set]
        valid_names = ["train", "valid"]
    else:
        train_set = lgb.Dataset(
            X,
            label=y,
            weight=weights,
            feature_name=vars_clean,
            free_raw_data=False,
        )
        valid_sets = [train_set]
        valid_names = ["train"]

    train_params = {
        "boosting_type": params["boosting_type"],
        "objective": params["objective"],
        "metric": params.get("metric", ["auc"]),
        "learning_rate": params["learning_rate"],
        "num_leaves": params["num_leaves"],
        "max_depth": params["max_depth"],
        "min_data_in_leaf": params["min_data_in_leaf"],
        "feature_fraction": params["feature_fraction"],
        "bagging_fraction": params["bagging_fraction"],
        "bagging_freq": params["bagging_freq"],
        "lambda_l1": params["lambda_l1"],
        "lambda_l2": params["lambda_l2"],
        "min_gain_to_split": params["min_gain_to_split"],
        "verbose": -1,
        "num_threads": params.get("num_threads", -1),
        "feature_pre_filter": False,
    }

    evals_result: dict[str, dict[str, list[float]]] = {}
    callbacks = [lgb.record_evaluation(evals_result)]
    verbose_eval = params.get("verbose_eval")
    if verbose_eval:
        callbacks.append(lgb.log_evaluation(verbose_eval))

    if val_frac > 0 and val_frac < 1:
        stopping_rounds = params.get("early_stopping_rounds", 100)
        if stopping_rounds:
            callbacks.append(
                lgb.early_stopping(
                    stopping_rounds=stopping_rounds,
                    first_metric_only=params.get("first_metric_only", False),
                )
            )

    booster = lgb.train(
        train_params,
        train_set,
        num_boost_round=params["num_boost_round"],
        valid_sets=valid_sets,
        valid_names=valid_names,
        callbacks=callbacks,
    )

    best_iter = booster.best_iteration or booster.current_iteration() or booster.num_trees()
    train_context["raw_scores"] = booster.predict(
        train_context["X"], num_iteration=best_iter, raw_score=True
    )
    if valid_context is not None:
        valid_context["raw_scores"] = booster.predict(
            valid_context["X"], num_iteration=best_iter, raw_score=True
        )

    return GBMModel(booster=booster), {"train": train_context, "valid": valid_context}
