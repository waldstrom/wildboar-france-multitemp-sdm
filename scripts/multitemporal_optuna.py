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
import yaml

import geopandas as gpd
import numpy as np
import optuna
import pandas as pd
import xarray as xr
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
from optuna.storages import RDBStorage

from scripts.utils import load_config, ensure_dir, extract_from_stack, restrict_test_years
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
    """
    base_dir = ensure_dir(Path(cfg["multitemporal"]["output_dir"]) / season / "optuna_base")

    # Ensure master stack exists before parallel loading to avoid repeated
    # rebuilds. Loading one year sequentially triggers creation if necessary.
    ds_tmp, _ = _load_stack(cfg, season, mapping[0].year)
    ds_tmp.close()

    stack_path, _ = _merge_stacks(cfg, season, mapping, base_dir, apply_vif=False)
    return stack_path.with_suffix(".zarr")


# -----------------------------------------------------------------------------

def _objective(
    trial: optuna.Trial,
    cfg: dict,
    season: str,
    mapping: list[YearOffset],
    base_stack: Path,
    pres_raw: dict[int, gpd.GeoDataFrame],
    env_cache: dict[int, np.ndarray],
    pres_test: gpd.GeoDataFrame,
    test_year: int,
    fr_poly: gpd.GeoSeries,
) -> float:
    cfg_trial = copy.deepcopy(cfg)
    ss = cfg_trial.get("optuna", {}).get("search_space", {})

    thin_pct = 1.0
    if "thinning_percentage" in ss:
        thin_pct = trial.suggest_float(
            "thin_pct",
            ss["thinning_percentage"]["low"],
            ss["thinning_percentage"]["high"],
        )
    train_factor = 1.0
    if "training_sample_factor" in ss:
        train_factor = trial.suggest_categorical(
            "train_factor", ss["training_sample_factor"]["choices"]
        )
    pct = thin_pct * train_factor

    # Hyperparameters toggled via Optuna
    thin_mode = "percentages"
    if "thinning_mode" in ss:
        thin_mode = trial.suggest_categorical(
            "thinning_mode", ss["thinning_mode"]["choices"]
        )

    if "include_hunting" in ss:
        cfg_trial["multitemporal"]["include_hunting"] = trial.suggest_categorical(
            "include_hunting", ss["include_hunting"]["choices"]
        )

    if "product_features" in ss:
        use_prod = trial.suggest_categorical(
            "product_features", ss["product_features"]["choices"]
        )
        ft = list(cfg_trial["maxent"].get("feature_types", []))
        if use_prod and "product" not in ft:
            ft.append("product")
        elif not use_prod:
            ft = [f for f in ft if f != "product"]
        cfg_trial["maxent"]["feature_types"] = ft

    if "max_iterations" in ss:
        cfg_trial["maxent"]["max_iterations"] = trial.suggest_int(
            "max_iterations",
            ss["max_iterations"]["low"],
            ss["max_iterations"]["high"],
            step=ss["max_iterations"].get("step", 1),
        )

    n_sat = trial.suggest_int(
        "n_sat", ss["n_satellite"]["low"], ss["n_satellite"]["high"]
    )
    n_bio = trial.suggest_int(
        "n_bio", ss["n_bioclim"]["low"], ss["n_bioclim"]["high"]
    )

    # ``n_era5`` is optional as some experiments may omit ERA5 predictors.
    # When absent from the search space, fall back to the value provided in the
    # base configuration (defaulting to zero).  This avoids a ``KeyError`` and
    # keeps the existing behaviour when ERA5 layers are not part of the model.
    n_era = cfg_trial.get("multitemporal", {}).get("vif", {}).get("n_era5", 0)
    if "n_era5" in ss:
        n_era = trial.suggest_int(
            "n_era", ss["n_era5"]["low"], ss["n_era5"]["high"]
        )

    if "regularization_multiplier" in ss:
        cfg_trial["maxent"]["regularization_multiplier"] = trial.suggest_float(
            "reg_mu",
            ss["regularization_multiplier"]["low"],
            ss["regularization_multiplier"]["high"],
            step=ss["regularization_multiplier"].get("step"),
        )
    if "n_hinge_features" in ss:
        cfg_trial["maxent"]["n_hinge_features"] = trial.suggest_int(
            "n_hinge",
            ss["n_hinge_features"]["low"],
            ss["n_hinge_features"]["high"],
            step=ss["n_hinge_features"].get("step", 1),
        )
    if "beta_lqp" in ss:
        cfg_trial["maxent"]["beta_lqp"] = trial.suggest_float(
            "beta_lqp",
            ss["beta_lqp"]["low"],
            ss["beta_lqp"]["high"],
            step=ss["beta_lqp"].get("step"),
        )
    if "background_per_year" in ss:
        cfg_trial["multitemporal"]["background_per_year"] = trial.suggest_int(
            "bg_per_year",
            ss["background_per_year"]["low"],
            ss["background_per_year"]["high"],
            step=ss["background_per_year"].get("step", 1),
        )
    bg_per_copy = cfg_trial["multitemporal"].get("background_per_year", 10000)

    # prepare output directories
    cfg_trial.setdefault("experiment", {})
    cfg_trial.setdefault("outputs", {})
    trial_dir = ensure_dir(Path(cfg["multitemporal"]["output_dir"]) / season / f"trial_{trial.number:03d}")
    # Keep the experiment name constant so that predictor stacks built once
    # in ``_prepare_stack`` are reused across trials.  Changing the name would
    # point ``_load_stack`` to a different directory and trigger an expensive
    # rebuild for every Optuna iteration.  Trial-specific outputs are handled
    # via ``trial_dir`` below.
    cfg_trial["experiment"].setdefault("name", cfg.get("experiment", {}).get("name", ""))
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

    # Persist the parameter configuration for this trial early so that it is
    # available even if subsequent steps fail.
    with open(trial_dir / "params.txt", "w") as fh:
        yaml.safe_dump(trial.params, fh)

    # Attach a dedicated log file for this trial so progress can be inspected
    # while ``study.optimize`` runs. Without this handler Optuna's parallel
    # execution hides all log output from worker processes which makes it
    # appear as though the run is stuck.
    log_file = trial_dir / "trial.log"
    root_logger = logging.getLogger()
    file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(name)s > %(message)s")
    )
    root_logger.addHandler(file_handler)
    log.info("Starting trial %d", trial.number)
    try:
        def _config_pct(cfg_loc: dict, season_loc: str, year_loc: int) -> float:
            thin_cfg = cfg_loc.get("presence_data", {}).get("thinning", {})
            season_map = thin_cfg.get("percentages", {}).get(season_loc, {})
            vals: list[float] = []
            for src_map in season_map.values():
                if isinstance(src_map, dict):
                    val = src_map.get(year_loc)
                    if val is None:
                        val = src_map.get(str(year_loc))
                    if val is not None:
                        vals.append(float(val))
                else:
                    vals.append(float(src_map))
            return min(vals) if vals else thin_cfg.get("percentage", 1.0)

        min_cnt = None
        if thin_mode == "equalize_counts":
            min_cnt = min(len(pres_raw[yo.year]) for yo in mapping)

        # Thin and shift presences
        pres_list = []
        for yo in mapping:
            gdf = pres_raw[yo.year]
            env = env_cache[yo.year]
            if thin_mode == "equalize_counts":
                tgt = min(len(gdf), max(1, int(min_cnt * train_factor)))
            else:
                pct_loc = 1.0 if thin_mode == "all_ones" else _config_pct(cfg_trial, season, yo.year)
                pct_loc *= train_factor
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

        # Background for training copies
        bg_list = []
        for i, yo in enumerate(mapping):
            shifted = fr_poly.translate(xoff=yo.dx, yoff=yo.dy)
            g = _sample_background_polygon(shifted, bg_per_copy)
            g["bg_id"] = np.arange(len(g)) + i * bg_per_copy
            bg_list.append(g)
        bg_train = gpd.GeoDataFrame(pd.concat(bg_list, ignore_index=True), crs="EPSG:2154")

        # Split validation subset
        val_frac = cfg["multitemporal"].get("validation_fraction", 0.1)
        rng = np.random.default_rng(cfg["experiment"].get("random_seed", 0))
        pres_val = pres_train.sample(frac=val_frac, random_state=rng.integers(1e9))
        bg_val = bg_train.sample(frac=val_frac, random_state=rng.integers(1e9))
        pres_train = pres_train.drop(pres_val.index).reset_index(drop=True)
        bg_train = bg_train.drop(bg_val.index).reset_index(drop=True)

        # VIF pruning according to trial parameters
        cfg_trial["multitemporal"]["vif"]["n_satellite"] = n_sat
        cfg_trial["multitemporal"]["vif"]["n_bioclim"] = n_bio
        cfg_trial["multitemporal"]["vif"]["n_era5"] = n_era
        ds_base = xr.open_zarr(base_stack, consolidated=False)
        ds_pruned, vars_used = _vif_filter(ds_base, cfg_trial)
        ds_base.close()

        if not cfg_trial["multitemporal"].get("include_hunting", True):
            drop = [v for v in ds_pruned.data_vars if "hunting" in v.lower()]
            if drop:
                ds_pruned = ds_pruned.drop_vars(drop)
                vars_used = [v for v in vars_used if v not in drop]

        # Ensure all selected predictors are available in the test stack. Optuna
        # trials may select variables that are missing for the test year which would
        # otherwise trigger a ``KeyError`` during extraction.  Load the test-year
        # stack and fuse yearly prefixes so names match those of the merged
        # training stack.
        ds_test, test_stack = _load_stack(cfg_trial, season, test_year)
        ds_test = fuse_yearly_layers(ds_test)
        missing = [v for v in vars_used if v not in ds_test.data_vars]
        if missing:
            available = sorted(ds_test.data_vars)
            for var in missing:
                log.warning(
                    "Predictor '%s' missing from test stack %s", var, test_stack
                )
            log.warning(
                "Dropping %d predictor(s) absent from test stack: %s",
                len(missing),
                ", ".join(missing),
            )
            log.info("Available predictors in test stack: %s", ", ".join(available))
            vars_used = [v for v in vars_used if v in ds_test.data_vars]
            select_vars = vars_used
            if "boundary_mask" in ds_pruned.data_vars:
                select_vars = vars_used + ["boundary_mask"]
            ds_pruned = ds_pruned[select_vars]
        else:
            log.debug("All selected predictors present in test stack")
        ds_test.close()

        stack_path = trial_dir / "stack.zarr"
        ds_pruned.to_zarr(stack_path, mode="w")

        model, tif_out, vars_used = train_model(
            cfg_trial, stack_path, vars_used, pres_train, bg_train
        )

        X_pres_train = extract_from_stack(stack_path, pres_train, vars_used)
        X_bg_train = extract_from_stack(stack_path, bg_train, vars_used)
        X_pres_val = extract_from_stack(stack_path, pres_val, vars_used)
        X_bg_val = extract_from_stack(stack_path, bg_val, vars_used)
        y_true_train = np.concatenate(
            [np.ones(len(X_pres_train)), np.zeros(len(X_bg_train))]
        )
        y_score_train = np.concatenate(
            [model.predict(X_pres_train), model.predict(X_bg_train)]
        )
        y_true_val = np.concatenate(
            [np.ones(len(X_pres_val)), np.zeros(len(X_bg_val))]
        )
        y_score_val = np.concatenate(
            [model.predict(X_pres_val), model.predict(X_bg_val)]
        )
        train_metrics = _compute_metrics(
            y_true_train,
            y_score_train,
            threshold=cfg_trial.get("evaluation", {}).get("classification_threshold", 0.5),
        )
        val_metrics = _compute_metrics(
            y_true_val,
            y_score_val,
            threshold=cfg_trial.get("evaluation", {}).get("classification_threshold", 0.5),
        )

        bg_test = _sample_background_polygon(fr_poly, bg_per_copy)
        X_pres_test = extract_from_stack(ds_test, pres_test, vars_used)
        X_bg_test = extract_from_stack(ds_test, bg_test, vars_used)
        y_true_test = np.concatenate(
            [np.ones(len(X_pres_test)), np.zeros(len(X_bg_test))]
        )
        y_score_test = np.concatenate(
            [model.predict(X_pres_test), model.predict(X_bg_test)]
        )
        test_metrics = _compute_metrics(
            y_true_test,
            y_score_test,
            threshold=cfg_trial.get("evaluation", {}).get("classification_threshold", 0.5),
        )

        for k, v in train_metrics.items():
            if not isinstance(v, np.ndarray):
                trial.set_user_attr(f"train_{k}", float(v))
        for k, v in val_metrics.items():
            if not isinstance(v, np.ndarray):
                trial.set_user_attr(f"val_{k}", float(v))
        for k, v in test_metrics.items():
            if not isinstance(v, np.ndarray):
                trial.set_user_attr(f"test_{k}", float(v))
        ds_test.close()

        # Persist metrics and predictor list for this trial
        stats_dir = Path(cfg_trial["experiment"]["stats_dir"])
        def _is_scalar(d: dict, key: str) -> bool:
            v = d.get(key)
            return not isinstance(v, np.ndarray) or np.asarray(v).size == 1

        metrics = sorted(
            m
            for m in set(train_metrics) | set(val_metrics) | set(test_metrics)
            if all(_is_scalar(d, m) for d in (train_metrics, val_metrics, test_metrics))
        )

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

        result = float(test_metrics.get("auc", 0.0) + test_metrics.get("cbi", 0.0))
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

    for season in cfg["multitemporal"]["seasons"]:
        years = cfg["multitemporal"]["years"]
        shift = cfg["multitemporal"]["shift_km"] * 1000
        mapping = _build_matrix(years, test_year, shift)
        cfg_base = copy.deepcopy(cfg)
        cfg_base.setdefault("multitemporal", {})["include_hunting"] = True
        base_stack = _prepare_stack(cfg_base, season, mapping)
        pres_raw = {y: _load_presence(cfg, season, y) for y in years if y != test_year}
        pres_test = _load_presence(cfg, season, test_year)
        ds_test, _ = _load_stack(cfg_base, season, test_year)
        ds_test.close()
        env_cache: dict[int, np.ndarray] = {}
        for y in years:
            if y == test_year:
                continue
            ds, _ = _load_stack(cfg_base, season, y)
            vars_year = list(ds.data_vars)
            env_cache[y] = extract_from_stack(ds, pres_raw[y], vars_year)
            ds.close()
        out_dir = ensure_dir(Path(cfg["multitemporal"]["output_dir"]) / season)

        opt_cfg = cfg.get("optuna", {})
        storage = opt_cfg.get("storage")
        study_name = opt_cfg.get("study_name")
        load_if_exists = opt_cfg.get("load_if_exists")

        if storage is None and study_name is None:
            from datetime import datetime

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

        storage_url = storage
        storage = RDBStorage(
            storage_url,
            engine_kwargs={"connect_args": {"timeout": 60}},
        )
        study = optuna.create_study(
            direction="maximize",
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
            lambda t: _objective(
                t,
                cfg,
                season,
                mapping,
                base_stack,
                pres_raw,
                env_cache,
                pres_test,
                test_year,
                fr_poly,
            ),
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
