# =============================================================================
# Project Title: Wild Boar Distribution Model for France
# Path: scripts/multitemporal_optuna.py
# Purpose: Optuna study over multitemporal thinning, VIF pruning and MaxEnt params.
# Created by Adrian Meyer, Institute Geomatics, FHNW (adrian.meyer@fhnw.ch)
# Created: July 2025
# Version: v.0.1.0
# =============================================================================
"""Optuna wrapper for multitemporal experiments."""
from __future__ import annotations

import argparse
import copy
import logging
from pathlib import Path
from datetime import datetime
import yaml

import geopandas as gpd
import numpy as np
import optuna
import pandas as pd
import xarray as xr
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
from optuna.storages import RDBStorage

from scripts.utils import (
    load_config,
    ensure_dir,
    extract_from_stack,
    count_model_params,
)
from scripts.multitemporal import (
    YearOffset,
    _build_matrix,
    _merge_stacks,
    _load_presence,
    _load_stack,
    _france_outline_from_mask,
    _sample_background_polygon,
    _vif_filter,
    fuse_yearly_layers,
    _experiment_dir,
    _merge_bias_maps,
    _sample_background_bias,
    _visualise_bias,
)
from scripts.thinning import mahalanobis_thin
from scripts.maxent_training import train_model
from scripts.evaluation import _compute_metrics

log = logging.getLogger(__name__)


# -----------------------------------------------------------------------------

def _prepare_stack(cfg: dict, season: str, mapping: list[YearOffset]) -> Path:
    """Build combined stack once for reuse in all trials.

    Returns the path to the unfiltered stack saved as a Zarr dataset which is
    used as the baseline for variable selection within the Optuna objective.
    The directory defined by ``multitemporal.output_dir`` already encodes the
    season and timestamp for this run, so we no longer create nested seasonal
    sub-directories here.
    """
    base_dir = ensure_dir(Path(cfg["multitemporal"]["output_dir"]) / "optuna_base")

    # Ensure master stack exists before parallel loading to avoid repeated
    # rebuilds. Loading one year sequentially triggers creation if necessary.
    ds_tmp, _ = _load_stack(cfg, season, mapping[0].year)
    ds_tmp.close()

    stack_path, _ = _merge_stacks(cfg, season, mapping, base_dir, apply_vif=False)
    return stack_path.with_suffix(".zarr")


# -----------------------------------------------------------------------------

def _config_pct(cfg: dict, season: str, year: int) -> float:
    thin_cfg = cfg.get("presence_data", {}).get("thinning", {})
    season_map = thin_cfg.get("percentages", {}).get(season, {})
    vals: list[float] = []
    for src_map in season_map.values():
        if isinstance(src_map, dict):
            val = src_map.get(year)
            if val is None:
                val = src_map.get(str(year))
            if val is not None:
                vals.append(float(val))
        else:
            vals.append(float(src_map))
    return min(vals) if vals else thin_cfg.get("percentage", 1.0)


def _precompute(
    cfg: dict,
    season: str,
    mapping: list[YearOffset],
    base_stack: Path,
    pres_raw: dict[int, gpd.GeoDataFrame],
    env_cache: dict[int, np.ndarray],
    pres_test: gpd.GeoDataFrame,
    test_year: int,
    fr_poly: gpd.GeoSeries,
    bias_train: xr.DataArray | None,
    bias_test: xr.DataArray | None,
) -> dict:
    bg_per_copy = cfg["multitemporal"].get("background_per_year", 10000)
    thin_cfg = cfg.get("presence_data", {}).get("thinning", {})
    thin_mode = "equalize_counts" if thin_cfg.get("equalize_counts") else "percentages"

    min_cnt = None
    if thin_mode == "equalize_counts":
        min_cnt = min(len(pres_raw[yo.year]) for yo in mapping)

    pres_list = []
    for yo in mapping:
        gdf = pres_raw[yo.year]
        env = env_cache[yo.year]
        if thin_mode == "equalize_counts":
            tgt = min(len(gdf), max(1, int(min_cnt)))
        else:
            pct_loc = _config_pct(cfg, season, yo.year)
            tgt = min(len(gdf), max(1, int(len(gdf) * pct_loc)))
        if tgt >= len(gdf):
            thin = gdf.copy()
        else:
            thin = gpd.GeoDataFrame(mahalanobis_thin(gdf, tgt, env), crs="EPSG:2154")
        thin.geometry = thin.geometry.translate(xoff=yo.dx, yoff=yo.dy)
        thin["x"] = thin.geometry.x
        thin["y"] = thin.geometry.y
        pres_list.append(thin)
    pres_train = gpd.GeoDataFrame(pd.concat(pres_list, ignore_index=True), crs="EPSG:2154")

    bias_strength = cfg.get("background", {}).get("bias_strength", 0.4)
    use_bias = (
        cfg.get("background", {}).get("use_bias_grid", False) and bias_strength > 0
    )
    if use_bias and bias_train is not None:
        bg_train = _sample_background_bias(
            bias_train, bg_per_copy * len(mapping), bias_strength
        )
    else:
        bg_list = []
        for i, yo in enumerate(mapping):
            shifted = fr_poly.translate(xoff=yo.dx, yoff=yo.dy)
            g = _sample_background_polygon(shifted, bg_per_copy)
            g["bg_id"] = np.arange(len(g)) + i * bg_per_copy
            bg_list.append(g)
        bg_train = gpd.GeoDataFrame(pd.concat(bg_list, ignore_index=True), crs="EPSG:2154")

    val_frac = cfg["multitemporal"].get("validation_fraction", 0.1)
    rng = np.random.default_rng(cfg["experiment"].get("random_seed", 0))
    pres_val = pres_train.sample(frac=val_frac, random_state=rng.integers(1e9))
    bg_val = bg_train.sample(frac=val_frac, random_state=rng.integers(1e9))
    pres_train = pres_train.drop(pres_val.index).reset_index(drop=True)
    bg_train = bg_train.drop(bg_val.index).reset_index(drop=True)

    ds_base = xr.open_zarr(base_stack, consolidated=False)
    ds_pruned, vars_used = _vif_filter(ds_base, cfg)
    ds_base.close()

    if not cfg["multitemporal"].get("include_hunting", True):
        drop = [v for v in ds_pruned.data_vars if "hunting" in v.lower()]
        if drop:
            ds_pruned = ds_pruned.drop_vars(drop)
            vars_used = [v for v in vars_used if v not in drop]

    ds_test, _ = _load_stack(cfg, season, test_year)
    ds_test = fuse_yearly_layers(ds_test)
    missing = [v for v in vars_used if v not in ds_test.data_vars]
    if missing:
        vars_used = [v for v in vars_used if v in ds_test.data_vars]
        select_vars = vars_used
        if "boundary_mask" in ds_pruned.data_vars:
            select_vars = vars_used + ["boundary_mask"]
        ds_pruned = ds_pruned[select_vars]

    stack_path = base_stack.parent / "stack_pruned.zarr"
    ds_pruned.to_zarr(stack_path, mode="w")

    X_pres_train = extract_from_stack(stack_path, pres_train, vars_used)
    X_bg_train = extract_from_stack(stack_path, bg_train, vars_used)
    X_pres_val = extract_from_stack(stack_path, pres_val, vars_used)
    X_bg_val = extract_from_stack(stack_path, bg_val, vars_used)
    y_true_train = np.concatenate(
        [np.ones(len(X_pres_train)), np.zeros(len(X_bg_train))]
    )
    y_true_val = np.concatenate(
        [np.ones(len(X_pres_val)), np.zeros(len(X_bg_val))]
    )

    if use_bias and bias_test is not None:
        bg_test = _sample_background_bias(bias_test, bg_per_copy, bias_strength)
    else:
        bg_test = _sample_background_polygon(fr_poly, bg_per_copy)
    X_pres_test = extract_from_stack(ds_test, pres_test, vars_used)
    X_bg_test = extract_from_stack(ds_test, bg_test, vars_used)
    y_true_test = np.concatenate(
        [np.ones(len(X_pres_test)), np.zeros(len(X_bg_test))]
    )
    ds_test.close()

    return {
        "stack_path": stack_path,
        "vars_used": vars_used,
        "pres_train": pres_train,
        "bg_train": bg_train,
        "pres_val": pres_val,
        "bg_val": bg_val,
        "pres_test": pres_test,
        "bg_test": bg_test,
        "X_pres_train": X_pres_train,
        "X_bg_train": X_bg_train,
        "X_pres_val": X_pres_val,
        "X_bg_val": X_bg_val,
        "X_pres_test": X_pres_test,
        "X_bg_test": X_bg_test,
        "y_true_train": y_true_train,
        "y_true_val": y_true_val,
        "y_true_test": y_true_test,
        "test_year": test_year,
    }


def _objective(
    trial: optuna.Trial,
    cfg: dict,
    season: str,
    data: dict,
) -> float:
    cfg_trial = copy.deepcopy(cfg)
    ss = cfg_trial.get("optuna", {}).get("search_space", {})
    reg_space = ss.get("regularization_multiplier", {})
    cfg_trial["maxent"]["regularization_multiplier"] = trial.suggest_float(
        "reg_mu",
        reg_space.get("low", 0.5),
        reg_space.get("high", 4.0),
        step=reg_space.get("step"),
    )

    trial_dir = ensure_dir(
        Path(cfg["multitemporal"]["output_dir"]) / f"trial_{trial.number:03d}"
    )
    cfg_trial.setdefault("experiment", {})
    cfg_trial.setdefault("outputs", {})
    cfg_trial["experiment"].setdefault(
        "name", cfg.get("experiment", {}).get("name", "")
    )
    cfg_trial["outputs"]["root"] = str(trial_dir)
    cfg_trial["outputs"]["logs_dir"] = str(trial_dir / "logs")
    cfg_trial["outputs"]["model_dir"] = str(trial_dir / "models")
    cfg_trial["outputs"]["maps_dir"] = str(trial_dir / "maps")
    cfg_trial["outputs"]["figures_dir"] = str(trial_dir / "figures")
    cfg_trial["experiment"]["vis_dir"] = str(trial_dir / "visualizations")
    cfg_trial["experiment"]["fi_dir"] = str(trial_dir / "feature_importance")
    cfg_trial["experiment"]["stats_dir"] = str(trial_dir / "stats")
    for p in [
        cfg_trial["outputs"]["logs_dir"],
        cfg_trial["outputs"]["model_dir"],
        cfg_trial["outputs"]["maps_dir"],
        cfg_trial["outputs"]["figures_dir"],
        cfg_trial["experiment"]["vis_dir"],
        cfg_trial["experiment"]["fi_dir"],
        cfg_trial["experiment"]["stats_dir"],
    ]:
        ensure_dir(p)

    with open(trial_dir / "params.txt", "w") as fh:
        yaml.safe_dump(trial.params, fh)

    log_file = trial_dir / "trial.log"
    root_logger = logging.getLogger()
    file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(name)s > %(message)s")
    )
    root_logger.addHandler(file_handler)
    log.info("Starting trial %d", trial.number)
    try:
        stack_path = Path(data["stack_path"]).resolve()
        model, tif_out, vars_used = train_model(
            cfg_trial,
            stack_path,
            data["vars_used"],
            data["pres_train"],
            data["bg_train"],
        )

        X_pres_train = data["X_pres_train"]
        X_bg_train = data["X_bg_train"]
        X_pres_val = data["X_pres_val"]
        X_bg_val = data["X_bg_val"]
        X_pres_test = data["X_pres_test"]
        X_bg_test = data["X_bg_test"]

        y_true_train = data["y_true_train"]
        y_true_val = data["y_true_val"]
        y_true_test = data["y_true_test"]

        y_score_train = np.concatenate(
            [model.predict(X_pres_train), model.predict(X_bg_train)]
        )
        y_score_val = np.concatenate(
            [model.predict(X_pres_val), model.predict(X_bg_val)]
        )
        y_score_test = np.concatenate(
            [model.predict(X_pres_test), model.predict(X_bg_test)]
        )
        n_params = count_model_params(model)

        train_metrics = _compute_metrics(
            y_true_train,
            y_score_train,
            threshold=cfg_trial.get("evaluation", {}).get("classification_threshold", 0.5),
            n_params=n_params,
        )
        val_metrics = _compute_metrics(
            y_true_val,
            y_score_val,
            threshold=cfg_trial.get("evaluation", {}).get("classification_threshold", 0.5),
            n_params=n_params,
        )
        test_metrics = _compute_metrics(
            y_true_test,
            y_score_test,
            threshold=cfg_trial.get("evaluation", {}).get("classification_threshold", 0.5),
            n_params=n_params,
        )

        if trial.number == 0:
            from scripts.visualization import (
                plot_points,
                plot_variables,
                plot_raster,
                plot_variable_histograms,
                export_response_curves,
            )
            from scripts.utils import sanitize_name
            import rasterio

            vis_dir = Path(cfg_trial["experiment"]["vis_dir"])

            try:
                plot_variables(stack_path, vars_used, vis_dir / "predictor_maps.png")
            except Exception as exc:
                log.warning("Failed to plot predictors: %s", exc)

            try:
                plot_points(
                    data["pres_train"],
                    vis_dir / "thinned_points.png",
                    "Thinned presences",
                )
                plot_points(
                    data["pres_test"],
                    vis_dir / "test_points.png",
                    "Test presences",
                )
                plot_points(
                    data["bg_train"],
                    vis_dir / "background_points.png",
                    "Background samples",
                    markersize=0.5,
                )
            except Exception as exc:
                log.warning("Failed to plot points: %s", exc)

            try:
                df_pres = pd.DataFrame(X_pres_train, columns=vars_used)
                df_bg = pd.DataFrame(X_bg_train, columns=vars_used)
                plot_variable_histograms(
                    df_pres,
                    df_bg,
                    vars_used,
                    vis_dir / "variable_hist.png",
                )
                export_response_curves(
                    model,
                    pd.concat([df_pres, df_bg], ignore_index=True),
                    vars_used,
                    vis_dir / "response_curves",
                )
            except Exception as exc:
                log.warning("Failed to plot variable stats: %s", exc)

            try:
                plot_raster(tif_out, vis_dir / "train_cloglog_map.png", "Train cloglog")
            except Exception as exc:
                log.warning("Failed to plot training map: %s", exc)

            try:
                ds_t, _ = _load_stack(cfg_trial, season, data["test_year"])
                ds_t = fuse_yearly_layers(ds_t)
                arrays = [ds_t[v].values for v in vars_used]
                boundary = (
                    ds_t["boundary_mask"].values.astype(bool)
                    if "boundary_mask" in ds_t
                    else np.ones(arrays[0].shape, dtype=bool)
                )
                grid = np.stack([arr.ravel() for arr in arrays], axis=1)
                preds = np.full(boundary.size, np.nan, dtype="float32")
                valid_idx = np.where(boundary.ravel())[0]
                chunk = int(1e5)
                for start in range(0, len(valid_idx), chunk):
                    idx = valid_idx[start : start + chunk]
                    preds[idx] = model.predict(grid[idx])
                suitability = preds.reshape(boundary.shape)
                suitability[~boundary] = np.nan
                safe = sanitize_name(cfg_trial["experiment"]["name"])
                tif_test = (
                    Path(cfg_trial["outputs"]["maps_dir"]) / f"{safe}_test_cloglog.tif"
                )
                meta = {
                    "driver": "GTiff",
                    "height": suitability.shape[0],
                    "width": suitability.shape[1],
                    "count": 1,
                    "dtype": "float32",
                    "crs": "EPSG:2154",
                    "transform": rasterio.transform.from_origin(
                        91000.0,
                        6124000.0 + suitability.shape[0] * 1000,
                        1000,
                        1000,
                    ),
                    "nodata": np.nan,
                    "compress": cfg_trial["maxent"].get(
                        "geotiff_compression", "deflate"
                    ),
                }
                with rasterio.open(tif_test, "w", **meta) as dst:
                    dst.write(suitability.astype("float32"), 1)
                plot_raster(
                    tif_test, vis_dir / "test_cloglog_map.png", "Test cloglog"
                )
                ds_t.close()
            except Exception as exc:
                log.warning("Failed to create test cloglog map: %s", exc)

        for k, v in train_metrics.items():
            if not isinstance(v, np.ndarray):
                trial.set_user_attr(f"train_{k}", float(v))
        for k, v in val_metrics.items():
            if not isinstance(v, np.ndarray):
                trial.set_user_attr(f"val_{k}", float(v))
        for k, v in test_metrics.items():
            if not isinstance(v, np.ndarray):
                trial.set_user_attr(f"test_{k}", float(v))

        stats_dir = Path(cfg_trial["experiment"]["stats_dir"])
        metrics = sorted(set(train_metrics) | set(val_metrics) | set(test_metrics))

        def _to_float(v: float | np.ndarray | None) -> float:
            if isinstance(v, np.ndarray):
                return float(v.ravel()[0]) if v.size == 1 else float("nan")
            try:
                return float(v) if v is not None else float("nan")
            except (TypeError, ValueError):
                return float("nan")

        df = pd.DataFrame(
            {
                "metric": metrics,
                "train": [_to_float(train_metrics.get(m)) for m in metrics],
                "validation": [_to_float(val_metrics.get(m)) for m in metrics],
                "test": [_to_float(test_metrics.get(m)) for m in metrics],
            }
        )
        df.to_csv(stats_dir / "metrics.csv", index=False)
        with open(stats_dir / "predictors.txt", "w") as fh:
            fh.write("\n".join(vars_used))

        trial.set_user_attr("cloglog_map", str(tif_out))
        trial.set_user_attr("predictors", vars_used)

        metric = (
            cfg_trial.get("optuna", {}).get("objective_metric", "boyce").lower()
        )
        if metric == "boyce":
            result = float(test_metrics.get("cbi", np.nan))
        elif metric == "auc":
            result = float(test_metrics.get("auc", np.nan))
        elif metric == "boyce+auc":
            result = float(
                test_metrics.get("cbi", 0.0) + test_metrics.get("auc", 0.0)
            )
        elif metric == "aicc":
            result = float(test_metrics.get("aicc", np.nan))
        elif metric == "omission":
            result = float(test_metrics.get("omission", np.nan))
        else:
            result = float(test_metrics.get("auc", np.nan))
        log.info("Trial %d completed", trial.number)
        return result
    finally:
        root_logger.removeHandler(file_handler)
        file_handler.close()


# -----------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Optuna search for multitemporal models")
    ap.add_argument("--config", required=True, help="Path to configuration YAML")
    args = ap.parse_args()
    cfg = load_config(args.config)

    fr_poly = _france_outline_from_mask(cfg["boundary_mask"])
    test_year = cfg.get("multitemporal", {}).get("test_year", 2021)

    base_root = Path(cfg.get("multitemporal", {}).get("output_dir", "expy"))

    for season in cfg["multitemporal"]["seasons"]:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_cfg = cfg.setdefault("experiment", {})
        exp_cfg["dir_prefix"] = exp_cfg.get("dir_prefix", "multi-regu")
        exp_cfg["name"] = f"{season}-{ts}"
        exp_cfg["base_dir"] = str(base_root)

        run_dir = _experiment_dir(cfg)
        cfg["multitemporal"]["output_dir"] = str(run_dir)
        ensure_dir(run_dir)
        vis_dir = ensure_dir(run_dir / "visualizations")
        exp_cfg["vis_dir"] = str(vis_dir)
        with open(Path(run_dir) / "config.yaml", "w") as fh:
            yaml.safe_dump(cfg, fh)

        years = cfg["multitemporal"]["years"]
        shift = cfg["multitemporal"]["shift_km"] * 1000
        mapping = _build_matrix(years, test_year, shift)
        cfg_base = copy.deepcopy(cfg)
        cfg_base.setdefault("multitemporal", {})["include_hunting"] = True
        base_stack = _prepare_stack(cfg_base, season, mapping)
        pres_raw = {y: _load_presence(cfg, season, y) for y in years if y != test_year}
        pres_test = _load_presence(cfg, season, test_year)
        env_cache: dict[int, np.ndarray] = {}
        for y in years:
            if y == test_year:
                continue
            ds, _ = _load_stack(cfg_base, season, y)
            vars_year = list(ds.data_vars)
            env_cache[y] = extract_from_stack(ds, pres_raw[y], vars_year)
            ds.close()
        bias_train = _merge_bias_maps(cfg, season, mapping)
        bias_test = _merge_bias_maps(cfg, season, [YearOffset(test_year, 0.0, 0.0)])
        if bias_train is not None:
            _visualise_bias(bias_train, mapping, fr_poly.iloc[0], season, test_year, vis_dir)
        data = _precompute(
            cfg,
            season,
            mapping,
            base_stack,
            pres_raw,
            env_cache,
            pres_test,
            test_year,
            fr_poly,
            bias_train,
            bias_test,
        )
        out_dir = ensure_dir(Path(cfg["multitemporal"]["output_dir"]))

        opt_cfg = cfg.get("optuna", {})
        storage = opt_cfg.get("storage")
        study_name = opt_cfg.get("study_name")
        load_if_exists = opt_cfg.get("load_if_exists")

        if storage is None and study_name is None:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            db_path = out_dir / f"optuna_{ts}.db"
            storage = f"sqlite:///{db_path}"
            study_name = f"{season}_optuna_{ts}"
            load_if_exists = False
            log.info("Starting new Optuna study. DB file: %s", db_path)
        else:
            if storage is None:
                db_path = out_dir / "optuna.db"
                storage = f"sqlite:///{db_path}"
            else:
                db_path = Path(storage.replace("sqlite:///", ""))
            if load_if_exists is None:
                load_if_exists = True
            log.info("Using Optuna storage at %s", storage.replace("sqlite:///", ""))
            if study_name is None:
                study_name = f"{season}_optuna"

        metric = opt_cfg.get("objective_metric", "boyce").lower()
        direction = "minimize" if metric in {"aicc", "omission"} else "maximize"
        storage_url = storage
        storage = RDBStorage(
            storage_url,
            engine_kwargs={"connect_args": {"timeout": 60}},
        )
        study = optuna.create_study(
            direction=direction,
            sampler=TPESampler(),
            pruner=MedianPruner(),
            study_name=study_name,
            storage=storage,
            load_if_exists=load_if_exists,
        )

        csv_path = Path(db_path).with_suffix(".csv")

        def _log_trial_csv(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
            row: dict[str, object] = {"number": trial.number}
            row.update(trial.params)
            row.update(trial.user_attrs)
            status = (
                "success"
                if trial.state == optuna.trial.TrialState.COMPLETE
                else "failed"
            )
            row["status"] = status
            df_row = pd.DataFrame([row])
            df_row.to_csv(
                csv_path, mode="a", header=not csv_path.exists(), index=False
            )

        study.optimize(
            lambda t: _objective(t, cfg, season, data),
            n_trials=cfg.get("optuna", {}).get("n_trials", 10),
            n_jobs=1,
            catch=(Exception,),
            gc_after_trial=True,
            callbacks=[_log_trial_csv],
        )
        df = study.trials_dataframe(
            attrs=("number", "value", "state", "params", "user_attrs")
        )
        df.to_csv(out_dir / "optuna_study.csv", index=False)


if __name__ == "__main__":
    main()
