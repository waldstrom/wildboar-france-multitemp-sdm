# =============================================================================
# Project Title: Wild Boar Distribution Model for France
# Path: scripts/maha-thin-multi.py
# Purpose: Systematically evaluate Mahalanobis thinning percentage variants with precomputed subsamples.
# Created by Adrian Meyer, Institute Geomatics, FHNW (adrian.meyer@fhnw.ch)
# Created: July 2025
# Version: v.0.1.0
# =============================================================================
"""Systematic study for multitemporal Mahalanobis thinning.

This script evaluates a multitemporal MaxEnt workflow while varying the
Mahalanobis thinning percentage. Predictor stacks and environmental distances
are prepared once and reused across all trials to minimise runtime.
"""
from __future__ import annotations

import argparse
import copy
import logging
from pathlib import Path
import select
import sys

import geopandas as gpd
import numpy as np
import optuna
import pandas as pd
import xarray as xr
from optuna.storages import RDBStorage

from scripts.utils import load_config, ensure_dir, extract_from_stack, count_model_params
from scripts.multitemporal import (
    YearOffset,
    _build_matrix,
    _merge_stacks,
    _load_presence,
    _load_stack,
    _france_outline_from_mask,
    _sample_background_polygon,
    _merge_bias_maps,
    _sample_background_bias,
    _vif_filter,
    fuse_yearly_layers,
    _visualise_matrix,
    _visualise_bias,
)
from scripts.thinning import mahalanobis_thin
from scripts.maha_thin_utils import _build_trial_params
from scripts.maxent_training import train_model
from scripts.evaluation import _compute_metrics

log = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
def _prepare_pruned_stack(
    cfg: dict,
    season: str,
    mapping: list[YearOffset],
    exp_dir: Path,
    cache_dir: Path,
    reuse: bool = False,
) -> tuple[Path, list[str]]:
    """Build merged predictor stack and apply VIF filtering once.

    When ``reuse`` is ``True`` and a cached stack exists in ``cache_dir``,
    it is loaded directly to avoid rebuilding.
    """
    pruned_path = cache_dir / "stack_pruned.zarr"
    vars_file = cache_dir / "vars_used.txt"
    if reuse and pruned_path.exists() and vars_file.exists():
        vars_used = [v for v in vars_file.read_text().strip().splitlines() if v not in {"x", "y"}]
        return pruned_path, vars_used

    ds_tmp, _ = _load_stack(cfg, season, mapping[0].year)
    ds_tmp.close()
    stack_path, _ = _merge_stacks(cfg, season, mapping, cache_dir, apply_vif=False)
    zarr_path = stack_path.with_suffix(".zarr")
    ds_base = xr.open_zarr(zarr_path, consolidated=False)
    ds_pruned, vars_used = _vif_filter(ds_base, cfg)
    vars_used = [v for v in vars_used if v not in {"x", "y"}]
    ds_base.close()
    ds_pruned.to_zarr(pruned_path, mode="w")
    with open(vars_file, "w") as fh:
        fh.write("\n".join(vars_used))
    return pruned_path, vars_used


# -----------------------------------------------------------------------------

def _precompute_thinning(
    cfg: dict,
    season: str,
    years: list[int],
    test_year: int,
    env_cache: dict[int, dict[str, np.ndarray]],
    pres_raw: dict[int, dict[str, gpd.GeoDataFrame]],
    levels: list[float],
    cache_dir: Path,
    reuse: bool = False,
) -> dict[float, dict[int, dict[str, gpd.GeoDataFrame]]]:
    """Return thinned presence samples for each percentage level, year and source.

    Thinned samples are cached to ``cache_dir`` and reused when available.
    """
    out: dict[float, dict[int, dict[str, gpd.GeoDataFrame]]] = {}
    rng_seed = cfg.get("experiment", {}).get("random_seed", 0)
    cache_dir = ensure_dir(cache_dir / "thinning")
    for lvl in levels:
        out[lvl] = {}
        for y in years:
            if y == test_year:
                continue
            out[lvl][y] = {}
            for src, gdf in pres_raw[y].items():
                cache_file = cache_dir / f"thin_{lvl}_{src}_{y}.gpkg"
                if reuse and cache_file.exists():
                    thin = gpd.read_file(cache_file)
                else:
                    if len(gdf) == 0:
                        thin = gdf.copy()
                    else:
                        env = env_cache[y][src]
                        pct = lvl
                        tgt = min(len(gdf), max(1, int(len(gdf) * pct)))
                        if tgt >= len(gdf):
                            thin = gdf.copy()
                        else:
                            np.random.seed(rng_seed)
                            thin = gpd.GeoDataFrame(
                                mahalanobis_thin(gdf, tgt, env), crs="EPSG:2154"
                            )
                    thin["x"] = thin.geometry.x
                    thin["y"] = thin.geometry.y
                    thin.to_file(cache_file, driver="GPKG")
                out[lvl][y][src] = thin
    return out


# -----------------------------------------------------------------------------

def _prompt_reuse(timeout: int = 10) -> bool:
    """Ask the user whether to reuse cached data with a timeout."""
    print("Reuse cached data? [y/N] ", end="", flush=True)
    rlist, _, _ = select.select([sys.stdin], [], [], timeout)
    if rlist:
        ans = sys.stdin.readline().strip().lower()
        return ans.startswith("y")
    print()
    return False


# -----------------------------------------------------------------------------

def _print_trial_percentages(
    cfg: dict,
    season: str,
    sources: list[str],
    thin_pct: float,
    variant: str,
) -> None:
    """Print the effective thinning percentages for this trial."""
    years = cfg.get("multitemporal", {}).get("years", [])
    print("percentages:")
    print(f"  {season}:")
    for src in sources:
        print(f"    {src}:")
        for y in years:
            val = thin_pct if (variant == "global" or variant == f"{src}_{y}") else 1.0
            print(f"      {y}: {val:.3f}")

def _objective(
    trial: optuna.Trial,
    cfg: dict,
    season: str,
    mapping: list[YearOffset],
    stack_path: Path,
    vars_used: list[str],
    pres_pre: dict[float, dict[int, dict[str, gpd.GeoDataFrame]]],
    bg_full: gpd.GeoDataFrame,
    pres_test: gpd.GeoDataFrame,
    ds_test: xr.Dataset,
    fr_poly: gpd.GeoSeries,
    levels: list[float],
    variants: list[str],
    sources: list[str],
    exp_dir: Path,
    bias_test: xr.DataArray | None,
) -> float:
    cfg_trial = copy.deepcopy(cfg)
    bounds = (min(levels), max(levels))
    thin_pct = float(trial.suggest_float("thin_pct", bounds[0], bounds[1]))
    idx = trial.suggest_int("thin_variant_idx", 0, len(variants) - 1)
    variant = variants[idx]

    _print_trial_percentages(cfg, season, sources, thin_pct, variant)

    trial_dir = ensure_dir(exp_dir / f"trial_{trial.number:03d}")
    cfg_trial.setdefault("outputs", {})
    cfg_trial["outputs"]["root"] = str(trial_dir)
    cfg_trial["outputs"]["logs_dir"] = str(trial_dir / "logs")
    cfg_trial["outputs"]["model_dir"] = str(trial_dir / "models")
    cfg_trial["outputs"]["maps_dir"] = str(trial_dir / "maps")
    cfg_trial["outputs"]["figures_dir"] = str(trial_dir / "figures")
    cfg_trial.setdefault("experiment", {})
    cfg_trial["experiment"]["vis_dir"] = str(trial_dir / "visualisations")
    cfg_trial["experiment"]["stats_dir"] = str(trial_dir / "stats")
    for p in [
        cfg_trial["outputs"]["logs_dir"],
        cfg_trial["outputs"]["model_dir"],
        cfg_trial["outputs"]["maps_dir"],
        cfg_trial["outputs"]["figures_dir"],
        cfg_trial["experiment"]["vis_dir"],
        cfg_trial["experiment"]["stats_dir"],
    ]:
        ensure_dir(p)

    pres_list = []
    for yo in mapping:
        year = yo.year
        gdfs = []
        for src in sources:
            lvl = thin_pct if (variant == "global" or variant == f"{src}_{year}") else 1.0
            gdfs.append(pres_pre[lvl][year][src])
        gdf = gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True), crs="EPSG:2154")
        gdf.geometry = gdf.geometry.translate(xoff=yo.dx, yoff=yo.dy)
        gdf["x"] = gdf.geometry.x
        gdf["y"] = gdf.geometry.y
        pres_list.append(gdf)
    pres_train_full = gpd.GeoDataFrame(pd.concat(pres_list, ignore_index=True), crs="EPSG:2154")
    bg_train_full = bg_full.copy()

    val_frac = cfg["multitemporal"].get("validation_fraction", 0.1)
    rng = np.random.default_rng(cfg["experiment"].get("random_seed", 0))
    pres_val = pres_train_full.sample(frac=val_frac, random_state=rng.integers(1e9))
    bg_val = bg_train_full.sample(frac=val_frac, random_state=rng.integers(1e9))
    pres_train = pres_train_full.drop(pres_val.index).reset_index(drop=True)
    bg_train = bg_train_full.drop(bg_val.index).reset_index(drop=True)

    bounds = fr_poly.iloc[0]
    _visualise_matrix(pres_list, mapping, bounds, season, cfg["multitemporal"]["test_year"], Path(cfg_trial["experiment"]["vis_dir"]))

    model, tif_out, vars_used = train_model(cfg_trial, stack_path, vars_used, pres_train, bg_train)

    X_pres_train = extract_from_stack(stack_path, pres_train, vars_used)
    X_bg_train = extract_from_stack(stack_path, bg_train, vars_used)
    X_pres_val = extract_from_stack(stack_path, pres_val, vars_used)
    X_bg_val = extract_from_stack(stack_path, bg_val, vars_used)
    y_true_train = np.concatenate([np.ones(len(X_pres_train)), np.zeros(len(X_bg_train))])
    y_score_train = np.concatenate([model.predict(X_pres_train), model.predict(X_bg_train)])
    y_true_val = np.concatenate([np.ones(len(X_pres_val)), np.zeros(len(X_bg_val))])
    y_score_val = np.concatenate([model.predict(X_pres_val), model.predict(X_bg_val)])
    n_params = count_model_params(model)
    train_metrics = _compute_metrics(y_true_train, y_score_train, n_params=n_params)
    val_metrics = _compute_metrics(y_true_val, y_score_val, n_params=n_params)

    bg_per_year = cfg["multitemporal"].get("background_per_year", 10000)
    bias_strength = cfg.get("background", {}).get("bias_strength", 0.4)
    use_bias = cfg.get("background", {}).get("use_bias_grid", False) and bias_strength > 0
    if use_bias and bias_test is not None:
        bg_test = _sample_background_bias(bias_test, bg_per_year, bias_strength)
    else:
        bg_test = _sample_background_polygon(fr_poly, bg_per_year)
    X_pres_test = extract_from_stack(ds_test, pres_test, vars_used)
    X_bg_test = extract_from_stack(ds_test, bg_test, vars_used)
    y_true_test = np.concatenate([np.ones(len(X_pres_test)), np.zeros(len(X_bg_test))])
    y_score_test = np.concatenate([model.predict(X_pres_test), model.predict(X_bg_test)])
    test_metrics = _compute_metrics(y_true_test, y_score_test, n_params=n_params)

    for k, v in test_metrics.items():
        if not isinstance(v, np.ndarray):
            trial.set_user_attr(f"test_{k}", float(v))
            print(f"Trial {trial.number} {k}: {v}")

    stats_dir = Path(cfg_trial["experiment"]["stats_dir"])
    df = pd.DataFrame(
        {
            "metric": sorted(set(train_metrics) | set(val_metrics) | set(test_metrics)),
            "train": [train_metrics.get(m) for m in sorted(set(train_metrics) | set(val_metrics) | set(test_metrics))],
            "validation": [val_metrics.get(m) for m in sorted(set(train_metrics) | set(val_metrics) | set(test_metrics))],
            "test": [test_metrics.get(m) for m in sorted(set(train_metrics) | set(val_metrics) | set(test_metrics))],
        }
    )
    df.to_csv(stats_dir / "metrics.csv", index=False)
    with open(stats_dir / "predictors.txt", "w") as fh:
        fh.write("\n".join(vars_used))

    result = float(test_metrics.get("auc", 0.0) + test_metrics.get("cbi", 0.0))
    return result


# -----------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Optuna search over Mahalanobis thinning levels")
    ap.add_argument("--config", required=True, help="Path to configuration YAML")
    args = ap.parse_args()
    cfg = load_config(args.config)

    reuse = _prompt_reuse()
    cache_root = ensure_dir(Path(cfg.get("cache_dir", "cache")))

    fr_poly = _france_outline_from_mask(cfg["boundary_mask"])
    test_year = cfg.get("multitemporal", {}).get("test_year", 2021)
    thin_levels = sorted(cfg.get("optuna", {}).get("thinning_levels", [0.4, 0.8, 1.0]))
    from datetime import datetime

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = cfg.get("experiment", {}).get("name", "experiment")
    exp_name = f"{base_name}_{ts}"
    cfg.setdefault("experiment", {})
    cfg["experiment"]["name"] = exp_name

    for season in cfg["multitemporal"]["seasons"]:
        years = cfg["multitemporal"]["years"]
        shift = cfg["multitemporal"]["shift_km"] * 1000
        mapping = _build_matrix(years, test_year, shift)
        exp_dir = ensure_dir(Path(cfg["multitemporal"]["output_dir"]) / season / exp_name)
        season_cache = ensure_dir(cache_root / season)
        stack_path, vars_used = _prepare_pruned_stack(
            cfg, season, mapping, exp_dir, season_cache, reuse
        )
        sources = list(
            cfg.get("presence_data", {})
            .get("thinning", {})
            .get("percentages", {})
            .get(season, {})
            .keys()
        )
        pres_raw = {}
        for y in years:
            if y == test_year:
                continue
            gdf = _load_presence(cfg, season, y)
            pres_raw[y] = {
                src: gdf[gdf["source"] == src].reset_index(drop=True) for src in sources
            }
        pres_test = _load_presence(cfg, season, test_year)
        ds_test, _ = _load_stack(cfg, season, test_year)
        ds_test = fuse_yearly_layers(ds_test)
        env_cache: dict[int, dict[str, np.ndarray]] = {}
        for y in years:
            if y == test_year:
                continue
            env_cache[y] = {}
            for src in sources:
                env_file = season_cache / f"env_{src}_{y}.npy"
                if reuse and env_file.exists():
                    env_cache[y][src] = np.load(env_file)
                else:
                    ds, _ = _load_stack(cfg, season, y)
                    vars_year = list(ds.data_vars)
                    env_cache[y][src] = extract_from_stack(ds, pres_raw[y][src], vars_year)
                    np.save(env_file, env_cache[y][src])
                    ds.close()
        pre_levels = sorted(set([1.0] + thin_levels))
        pres_pre = _precompute_thinning(
            cfg,
            season,
            years,
            test_year,
            env_cache,
            pres_raw,
            pre_levels,
            season_cache,
            reuse,
        )

        bias_strength = cfg.get("background", {}).get("bias_strength", 0.4)
        use_bias = cfg.get("background", {}).get("use_bias_grid", False) and bias_strength > 0
        bias_train = bias_test = None
        if use_bias:
            bias_train = _merge_bias_maps(cfg, season, mapping)
            bias_test = _merge_bias_maps(cfg, season, [YearOffset(test_year, 0.0, 0.0)])
            if bias_train is not None:
                _visualise_bias(bias_train, mapping, fr_poly.iloc[0], season, test_year, exp_dir)

        bg_per_copy = cfg["multitemporal"].get("background_per_year", 10000)
        if use_bias and bias_train is not None:
            bg_full = _sample_background_bias(
                bias_train, bg_per_copy * len(mapping), bias_strength
            )
        else:
            bg_list = []
            for i, yo in enumerate(mapping):
                shifted = fr_poly.translate(xoff=yo.dx, yoff=yo.dy)
                g = _sample_background_polygon(shifted, bg_per_copy)
                g["bg_id"] = np.arange(len(g)) + i * bg_per_copy
                bg_list.append(g)
            bg_full = gpd.GeoDataFrame(pd.concat(bg_list, ignore_index=True), crs="EPSG:2154")
        opt_cfg = cfg.get("optuna", {})
        storage = opt_cfg.get("storage")
        study_name = opt_cfg.get("study_name")
        if storage is None:
            db_path = exp_dir / f"optuna_{ts}.db"
            storage = f"sqlite:///{db_path}"[0:]
            study_name = f"{season}_maha_optuna_{ts}"
            log.info("Starting new Optuna study. DB file: %s", db_path)
        storage_obj = RDBStorage(storage)

        variants = ["global"] + [f"{src}_{y}" for y in years if y != test_year for src in sources]
        trial_params = _build_trial_params(thin_levels, variants)
        study = optuna.create_study(direction="maximize", study_name=study_name, storage=storage_obj)
        for p in trial_params:
            study.enqueue_trial(p)
        study.optimize(
            lambda t: _objective(
                t,
                cfg,
                season,
                mapping,
                stack_path,
                vars_used,
                pres_pre,
                bg_full,
                pres_test,
                ds_test,
                fr_poly,
                thin_levels,
                variants,
                sources,
                exp_dir,
                bias_test,
            ),
            n_trials=len(trial_params),
            n_jobs=1,
            catch=(Exception,),
        )
        df = study.trials_dataframe(attrs=("number", "value", "state", "params", "user_attrs"))
        df.to_csv(exp_dir / "maha_trials.csv", index=False)
        ds_test.close()


if __name__ == "__main__":
    main()
