# Path: scripts/bias-strength-multi-optuna.py
# Purpose: Systematically evaluate bias correction strength variants for multitemporal models.
# Created by Adrian Meyer, Institute Geomatics, FHNW (adrian.meyer@fhnw.ch)
# Created: July 2025
# Version: v.0.1.0
# =============================================================================
"""Optuna study over bias correction strength for multitemporal models."""
from __future__ import annotations

import argparse
import copy
import logging
from pathlib import Path
from typing import Sequence
import select
import sys
from datetime import datetime

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
from scripts.maxent_training import train_model
from scripts.evaluation import _compute_metrics

log = logging.getLogger(__name__)

DEFAULT_CONFIG_PATHS = [
    Path("configmultitemporal_winter.yaml"),
    Path("configmultitemporal_summer.yaml"),
]
DEFAULT_OUTPUT_ROOT = Path("exps") / "bias-test"


def _resolve_config_paths(paths: Sequence[str] | None) -> list[Path]:
    """Return configuration files to process."""

    if paths:
        return [Path(p) for p in paths]
    return list(DEFAULT_CONFIG_PATHS)


def _experiment_dir(output_root: Path, season: str, exp_name: str) -> Path:
    """Construct the experiment directory for ``season`` and ``exp_name``."""

    safe_season = season.strip().lower().replace(" ", "-")
    return ensure_dir(output_root / safe_season / exp_name)


def _summaries_from_trials(
    trials: Sequence[optuna.trial.FrozenTrial],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return detailed and aggregated bias strength summaries."""

    rows: list[dict[str, float]] = []
    for tr in trials:
        if tr.state != optuna.trial.TrialState.COMPLETE:
            continue
        bias = float(tr.params.get("bias_strength", float("nan")))
        auc = float(tr.user_attrs.get("test_auc", float("nan")))
        cbi_pearson = float(
            tr.user_attrs.get(
                "test_cbi_pearson", tr.user_attrs.get("test_cbi", float("nan"))
            )
        )
        cbi_spearman = float(tr.user_attrs.get("test_cbi_spearman", float("nan")))
        rows.append(
            {
                "bias_strength": bias,
                "auc": auc,
                "cbi_pearson": cbi_pearson,
                "cbi_spearman": cbi_spearman,
            }
        )

    metrics_df = pd.DataFrame(rows)
    if metrics_df.empty:
        return metrics_df, pd.DataFrame()

    metrics_df = metrics_df.sort_values("bias_strength").reset_index(drop=True)
    summary_df = (
        metrics_df.groupby("bias_strength")
        .agg(
            auc_mean=("auc", "mean"),
            auc_std=("auc", "std"),
            cbi_pearson_mean=("cbi_pearson", "mean"),
            cbi_pearson_std=("cbi_pearson", "std"),
            cbi_spearman_mean=("cbi_spearman", "mean"),
            cbi_spearman_std=("cbi_spearman", "std"),
            n_trials=("auc", "count"),
        )
        .reset_index()
        .sort_values("bias_strength")
        .reset_index(drop=True)
    )
    return metrics_df, summary_df


def _save_summary_tables(
    trials: Sequence[optuna.trial.FrozenTrial], exp_dir: Path
) -> None:
    """Persist bias strength metric tables for ``trials`` into ``exp_dir``."""

    metrics_df, summary_df = _summaries_from_trials(trials)
    if metrics_df.empty:
        log.warning("No successful trials to summarise for bias strength study")
        return

    metrics_df.to_csv(exp_dir / "bias_strength_metrics.csv", index=False)
    summary_df.to_csv(exp_dir / "bias_strength_summary.csv", index=False)


def _prepare_pruned_stack(
    cfg: dict,
    season: str,
    mapping: list[YearOffset],
    exp_dir: Path,
    cache_dir: Path,
    reuse: bool = False,
) -> tuple[Path, list[str]]:
    """Build merged predictor stack and apply VIF filtering once."""
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


def _prompt_reuse(timeout: int = 10) -> bool:
    """Ask the user whether to reuse cached data with a timeout."""
    print("Reuse cached data? [y/N] ", end="", flush=True)
    rlist, _, _ = select.select([sys.stdin], [], [], timeout)
    if rlist:
        ans = sys.stdin.readline().strip().lower()
        return ans.startswith("y")
    print()
    return False


def _objective(
    trial: optuna.Trial,
    cfg: dict,
    season: str,
    test_year: int,
    mapping: list[YearOffset],
    stack_path: Path,
    vars_used: list[str],
    pres_raw: dict[int, dict[str, gpd.GeoDataFrame]],
    bias_train: xr.DataArray | None,
    pres_test: gpd.GeoDataFrame,
    ds_test: xr.Dataset,
    fr_poly: gpd.GeoSeries,
    exp_dir: Path,
    bias_test: xr.DataArray | None,
) -> float:
    cfg_trial = copy.deepcopy(cfg)
    cfg_trial.setdefault("multitemporal", {})
    cfg_trial["multitemporal"]["test_year"] = test_year
    strength = float(trial.suggest_float("bias_strength", 0.0, 1.0))
    cfg_trial.setdefault("background", {})
    cfg_trial["background"]["bias_strength"] = strength
    cfg_trial["background"]["use_bias_grid"] = strength > 0

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

    # Assemble training presences
    pres_list = []
    sources = list(next(iter(pres_raw.values())).keys()) if pres_raw else []
    for yo in mapping:
        gdfs = [pres_raw[yo.year][src] for src in sources]
        gdf = gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True), crs="EPSG:2154")
        gdf.geometry = gdf.geometry.translate(xoff=yo.dx, yoff=yo.dy)
        gdf["x"] = gdf.geometry.x
        gdf["y"] = gdf.geometry.y
        pres_list.append(gdf)
    pres_train_full = gpd.GeoDataFrame(pd.concat(pres_list, ignore_index=True), crs="EPSG:2154")

    # Sample background according to strength
    bg_per_copy = cfg["multitemporal"].get("background_per_year", 10000)
    if strength > 0 and bias_train is not None:
        bg_full = _sample_background_bias(bias_train, bg_per_copy * len(mapping), strength)
    else:
        bg_list = []
        for i, yo in enumerate(mapping):
            shifted = fr_poly.translate(xoff=yo.dx, yoff=yo.dy)
            g = _sample_background_polygon(shifted, bg_per_copy)
            g["bg_id"] = np.arange(len(g)) + i * bg_per_copy
            bg_list.append(g)
        bg_full = gpd.GeoDataFrame(pd.concat(bg_list, ignore_index=True), crs="EPSG:2154")

    val_frac = cfg["multitemporal"].get("validation_fraction", 0.1)
    rng = np.random.default_rng(cfg["experiment"].get("random_seed", 0))
    pres_val = pres_train_full.sample(frac=val_frac, random_state=rng.integers(1e9))
    bg_val = bg_full.sample(frac=val_frac, random_state=rng.integers(1e9))
    pres_train = pres_train_full.drop(pres_val.index).reset_index(drop=True)
    bg_train = bg_full.drop(bg_val.index).reset_index(drop=True)

    bounds = fr_poly.iloc[0]
    _visualise_matrix(
        pres_list,
        mapping,
        bounds,
        season,
        test_year,
        Path(cfg_trial["experiment"]["vis_dir"]),
    )

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
    if strength > 0 and bias_test is not None:
        bg_test = _sample_background_bias(bias_test, bg_per_year, strength)
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


def _run_for_config(cfg_path: Path, reuse: bool, output_root: Path) -> None:
    """Execute the bias strength study for a single configuration."""

    cfg = load_config(cfg_path)
    cache_root = ensure_dir(Path(cfg.get("cache_dir", "cache")))
    fr_poly = _france_outline_from_mask(cfg["boundary_mask"])
    test_year = cfg.get("multitemporal", {}).get("test_year", 2021)
    cfg.setdefault("multitemporal", {})
    cfg["multitemporal"]["test_year"] = test_year

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = cfg.get("experiment", {}).get("name") or cfg_path.stem
    exp_name = f"{base_name}_{ts}"
    cfg.setdefault("experiment", {})
    cfg["experiment"]["name"] = exp_name

    for season in cfg["multitemporal"]["seasons"]:
        log.info("Running bias strength study for %s using %s", season, cfg_path)
        years = cfg["multitemporal"]["years"]
        shift = cfg["multitemporal"]["shift_km"] * 1000
        mapping = _build_matrix(years, test_year, shift)
        exp_dir = _experiment_dir(output_root, season, exp_name)
        safe_season = season.strip().lower().replace(" ", "-")
        cfg["multitemporal"]["output_dir"] = str(exp_dir.parent)
        season_cache = ensure_dir(cache_root / safe_season)
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

        bias_train = _merge_bias_maps(cfg, season, mapping)
        bias_test = _merge_bias_maps(cfg, season, [YearOffset(test_year, 0.0, 0.0)])
        if bias_train is not None:
            _visualise_bias(bias_train, mapping, fr_poly.iloc[0], season, test_year, exp_dir)

        bias_levels = [round(x * 0.2, 1) for x in range(6)]
        storage = cfg.get("optuna", {}).get("storage")
        study_name = cfg.get("optuna", {}).get("study_name")
        if storage is None:
            db_path = exp_dir / f"optuna_bias_{ts}.db"
            storage = f"sqlite:///{db_path}"[0:]
            study_name = f"{season}_bias_optuna_{ts}"
            log.info("Starting new Optuna study. DB file: %s", db_path)
        storage_obj = RDBStorage(storage)
        study = optuna.create_study(
            direction="maximize", study_name=study_name, storage=storage_obj
        )
        for lvl in bias_levels:
            study.enqueue_trial({"bias_strength": lvl})
        study.optimize(
            lambda t: _objective(
                t,
                cfg,
                season,
                test_year,
                mapping,
                stack_path,
                vars_used,
                pres_raw,
                bias_train,
                pres_test,
                ds_test,
                fr_poly,
                exp_dir,
                bias_test,
            ),
            n_trials=len(bias_levels),
            n_jobs=1,
            catch=(Exception,),
        )
        df = study.trials_dataframe(attrs=("number", "value", "state", "params", "user_attrs"))
        df.to_csv(exp_dir / "bias_strength_trials.csv", index=False)
        _save_summary_tables(study.trials, exp_dir)
        ds_test.close()


def main() -> None:
    ap = argparse.ArgumentParser(description="Optuna search over bias strengths")
    ap.add_argument(
        "--config",
        action="append",
        help=(
            "Path to configuration YAML. Provide multiple times to override the "
            "default summer and winter setups."
        ),
    )
    ap.add_argument(
        "--output-root",
        default=str(DEFAULT_OUTPUT_ROOT),
        help="Directory where bias test experiments are stored.",
    )
    args = ap.parse_args()

    reuse = _prompt_reuse()
    config_paths = _resolve_config_paths(args.config)
    output_root = ensure_dir(Path(args.output_root))

    for cfg_path in config_paths:
        _run_for_config(cfg_path, reuse, output_root)


if __name__ == "__main__":
    main()
