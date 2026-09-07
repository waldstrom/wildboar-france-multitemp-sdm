"""Multitemporal positive control experiment runner.

This script orchestrates a multitemporal MaxEnt experiment for a fixed test
year (2021) that augments the predictor stack with an explicit
``positive_control`` layer marking all cells that contain at least one filtered
training presence record.

The control layer acts as a sanity check.  A well configured classifier should
attribute an overwhelming importance to this perfectly predictive feature and,
as a consequence, yield near optimal evaluation metrics.  The implementation
closely follows :mod:`scripts.multitemporal` to ensure that the resulting
outputs remain comparable to existing experiments.
"""

from __future__ import annotations

import argparse
import copy
import logging
import os
import pickle
import shutil
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import xarray as xr
from tqdm.auto import tqdm

from scripts.utils import (
    deduplicate_cells,
    drop_background_overlaps,
    ensure_dir,
    load_config,
    load_mask,
    sanitize_name,
)
from scripts.positive_control_utils import (
    positive_control_da,
    write_positive_control_test_layer,
)
from scripts import multitemporal as mt
from scripts.cross_validation import run_cv
from scripts.evaluation import _compute_metrics, evaluate
from scripts.maxent_training import train_model
from scripts.multitemp_preprocessing import export_geotiffs
from scripts.thinning import grid_thin, mahalanobis_thin
from scripts.variable_importance import permutation_importance
from scripts.visualization import (
    compile_response_curves_pdf,
    export_response_curves,
    plot_cloglog_points,
)


log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
def _process_season(
    cfg: dict,
    season: str,
    rebuild: bool,
    *,
    test_year: int,
    include_positive_control: bool = False,
) -> None:
    """Run the multitemporal workflow for a single season.

    The implementation mirrors :func:`scripts.multitemporal.process_season`
    with the optional ability to inject a positive control predictor.
    """

    years = cfg["multitemporal"]["years"]
    shift = cfg["multitemporal"]["shift_km"] * 1000
    exp_dir = ensure_dir(mt._experiment_dir(cfg))
    out_root = ensure_dir(exp_dir / "output" / season)
    stack_dir = ensure_dir(exp_dir / "data-stack")
    mt.log_heading(f"Processing season {season}")

    if test_year not in years:
        raise ValueError(f"Test year {test_year} not available in configuration")

    master_path = mt._master_stack_path(cfg, season)
    if rebuild or not master_path.exists():
        mt._build_master_stack(cfg, season)

    fr_poly = mt._france_outline_from_mask(cfg["boundary_mask"])
    bounds = fr_poly.iloc[0]
    mask = load_mask(cfg["boundary_mask"])

    pres_raw: dict[int, gpd.GeoDataFrame] = {
        y: mt._load_presence(cfg, season, y) for y in years
    }
    thin_cfg = cfg.get("presence_data", {}).get("thinning", {})
    vars_fixed: list[str] | None = None

    metrics_records = []

    mapping = mt._build_matrix(years, test_year, shift)
    mt.log_heading(f"{season} - test year {test_year}")
    log.info("Processing test year %d", test_year)
    run_dir = ensure_dir(out_root / f"test_{test_year}")
    input_dir = ensure_dir(exp_dir / f"input-data-test-{test_year}")

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
        logging.Formatter("%(asctime)s [%(levelname)s] %(name)s > %(message)s")
    )
    logging.getLogger().addHandler(fh)

    bias_da = mt._merge_bias_maps(cfg_run, season, mapping)
    if bias_da is not None:
        export_geotiffs(
            xr.Dataset({"bias": bias_da}),
            input_dir,
            cfg_run["predictors"].get("nodata_value", np.nan),
            n_time_steps=1,
            compress=cfg_run["maxent"].get("geotiff_compression", "LZW"),
        )
        mt._visualise_bias(bias_da, mapping, bounds, season, test_year, vis_dir)

    stack_name = f"{season}_stack"
    stack_path = input_dir / stack_name
    if rebuild or not stack_path.exists():
        mt.log_heading("Building combined predictor stack")
        stack_path, vars_used = mt._merge_stacks(
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
        vars_used = [
            f.stem for f in stack_path.glob("*.tif") if f.stem != "boundary_mask"
        ]
        if vars_fixed is None:
            vars_fixed = vars_used
    log.info("Using predictors: %s", ", ".join(vars_used))

    mt.log_heading("Preparing training data")
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
        log.info("Grid thinning %.0f km retained %d records", grid_km, len(pres_train))
    log.info("Finished shifting presence points (%d records)", len(pres_train))

    if thin_cfg.get("enable", False):
        env = mt.extract_from_stack(stack_path, pres_train, vars_used)
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
                        log.warning("No presence records for %s; skipping thinning", src)
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

    if include_positive_control:
        mt.log_heading("Generating positive control layer")
        positive_control_da(stack_path, pres_train)
        if "positive_control" not in vars_used:
            vars_used = [*vars_used, "positive_control"]
        if vars_fixed is not None and "positive_control" not in vars_fixed:
            vars_fixed = [*vars_fixed, "positive_control"]

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
        bg_train = mt._sample_background_bias(
            bias_da, bg_per_copy * len(mapping), bias_strength, mask
        )
        bg_train = deduplicate_cells(bg_train)
    else:
        log.info("Sampling background points for %d matrix copies", len(mapping))
        bg_list = []
        for i, yo in enumerate(tqdm(mapping, desc="Sample background", leave=False)):
            shifted = fr_poly.translate(xoff=yo.dx, yoff=yo.dy)
            g = mt._sample_background_polygon(shifted, bg_per_copy)
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

    val_frac = cfg["multitemporal"].get("validation_fraction", 0.1)
    rng = np.random.default_rng(cfg["experiment"].get("random_seed", 0))
    pres_val = pres_train.sample(frac=val_frac, random_state=rng.integers(1e9))
    bg_val = bg_train.sample(frac=val_frac, random_state=rng.integers(1e9))
    pres_train = pres_train.drop(pres_val.index).reset_index(drop=True)
    bg_train = bg_train.drop(bg_val.index).reset_index(drop=True)
    log.debug("Validation presences: %d", len(pres_val))
    log.debug("Validation background points: %d", len(bg_val))

    pres_test = mt._load_presence(cfg, season, test_year)
    bias_test = mt._merge_bias_maps(cfg_run, season, [mt.YearOffset(test_year, 0.0, 0.0)])
    if (
        cfg_run.get("background", {}).get("use_bias_grid", False)
        and bias_test is not None
        and bias_strength > 0
    ):
        bg_test = mt._sample_background_bias(bias_test, bg_per_copy, bias_strength, mask)
    else:
        bg_test = mt._sample_background(cfg, bg_per_copy)
    bg_test = deduplicate_cells(bg_test)

    test_stack, _ = mt._load_stack(cfg_run, season, test_year, rebuild, validate=rebuild)
    test_stack = mt.fuse_yearly_layers(test_stack)
    missing = [v for v in vars_used if v not in test_stack.data_vars]
    if include_positive_control and "positive_control" in missing:
        missing.remove("positive_control")
    if missing:
        log.warning(
            "Missing predictors for test year %d: %s",
            test_year,
            ", ".join(missing),
        )
        vars_used = [v for v in vars_used if v not in missing]
        if vars_fixed is not None:
            vars_fixed = [v for v in vars_fixed if v in vars_used]
    export_vars = [v for v in vars_used if v in test_stack.data_vars]
    if len(export_vars) < len(vars_used):
        missing_exports = sorted(set(vars_used) - set(export_vars))
        if missing_exports:
            log.debug(
                "Skipping export of predictors absent from test stack: %s",
                ", ".join(missing_exports),
            )
    if "boundary_mask" in test_stack.data_vars and "boundary_mask" not in export_vars:
        export_vars.append("boundary_mask")
    test_stack = test_stack.sortby(["y", "x"])[export_vars]
    test_stack_path = run_dir / "test_stack"
    export_geotiffs(
        test_stack,
        test_stack_path,
        cfg_run["predictors"].get("nodata_value", np.nan),
        compress=cfg_run["maxent"].get("geotiff_compression", "LZW"),
        n_workers=cfg_run["multitemporal"].get("n_workers", os.cpu_count() or 1),
    )
    if include_positive_control:
        write_positive_control_test_layer(test_stack_path)
    log.info("Test stack written to %s", test_stack_path)

    env_test = mt.extract_from_stack(test_stack_path, pres_test, vars_used)
    if thin_cfg.get("enable", False):
        if "percentages" in thin_cfg:
            season_map = thin_cfg.get("percentages", {}).get(season, {})
            frames = []
            for src, year_map in season_map.items():
                g_src = pres_test[pres_test["source"] == src]
                if len(g_src) == 0:
                    continue
                pct = year_map.get(test_year, 1.0) if isinstance(year_map, dict) else year_map
                if pct < 1.0 and len(g_src) > 3:
                    tgt = max(1, int(len(g_src) * pct))
                    env_src = env_test[pres_test["source"] == src]
                    frames.append(mahalanobis_thin(g_src, tgt, env_src))
                else:
                    frames.append(g_src)
            remaining = pres_test.loc[~pres_test["source"].isin(season_map.keys())]
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

    pts_dir = ensure_dir(run_dir / "points")
    pres_train.to_file(pts_dir / "train.gpkg", driver="GPKG")
    pres_val.to_file(pts_dir / "val.gpkg", driver="GPKG")
    pres_test.to_file(pts_dir / "test.gpkg", driver="GPKG")
    env_test = mt.extract_from_stack(test_stack_path, pres_test, vars_used)

    X_pres = env_test
    X_bg = mt.extract_from_stack(test_stack_path, bg_test, vars_used)

    pres_file = input_dir / f"{season}_train_points.gpkg"
    bg_file = input_dir / f"{season}_background_points.gpkg"
    pres_train.to_file(pres_file, driver="GPKG")
    bg_train.to_file(bg_file, driver="GPKG")
    env_pres = mt.extract_from_stack(stack_path, pres_train, vars_used)
    df_pres = pres_train.drop(columns="geometry").copy()
    pres_env_df = pd.DataFrame(env_pres, columns=vars_used)
    df_pres = pd.concat([df_pres.reset_index(drop=True), pres_env_df], axis=1)
    df_pres.to_csv(input_dir / f"{season}_train_features.csv", index=False)

    env_bg = mt.extract_from_stack(stack_path, bg_train, vars_used)
    df_bg = bg_train.drop(columns="geometry").copy()
    bg_env_df = pd.DataFrame(env_bg, columns=vars_used)
    df_bg = pd.concat([df_bg.reset_index(drop=True), bg_env_df], axis=1)
    df_bg.to_csv(input_dir / f"{season}_background_features.csv", index=False)
    log.debug("Saved training data to %s", input_dir)

    env_pres_val = mt.extract_from_stack(stack_path, pres_val, vars_used)
    env_bg_val = mt.extract_from_stack(stack_path, bg_val, vars_used)

    mt._visualise_points(
        pres_train, mapping, bounds, season, test_year, vis_dir, prefix="train"
    )
    mt._visualise_points(
        bg_train, mapping, bounds, season, test_year, vis_dir, prefix="background"
    )
    mt._visualise_features(stack_path, vars_used, season, test_year, vis_dir)

    if cfg_run.get("export_java_compatible_input_data", True):
        mt.export_java_compatible_bundle(
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

    mt._visualise_matrix(pres_list, mapping, bounds, season, test_year, vis_dir)

    model, _, vars_used = train_model(
        cfg_run, stack_path, vars_used, pres_train, bg_train
    )

    response_dir = vis_dir / "response_curves"
    export_response_curves(
        model,
        pd.concat([pres_env_df, bg_env_df], ignore_index=True),
        vars_used,
        response_dir,
    )
    compile_response_curves_pdf(response_dir, response_dir / "response_curves.pdf")

    if cfg_run.get("evaluation", {}).get("cross_validation", {}).get("enable", False):
        run_cv(
            cfg_run,
            pres_gdf=pres_train,
            bg_gdf=bg_train,
            stack_path=stack_path,
            vars_used=vars_used,
        )

    mt._visualise_test(pres_test, bounds, season, test_year, vis_dir)
    train_tif = mt._predict_raster(model, stack_path, vars_used, cfg_run, suffix="train")
    test_tif = mt._predict_raster(model, test_stack_path, vars_used, cfg_run, suffix="test")
    plot_cloglog_points(
        train_tif,
        pres_train,
        bounds,
        mapping,
        vis_dir / f"train_cloglog_train_points_{test_year}.png",
        f"{season} train points – test {test_year}",
        edgecolor="white",
    )
    plot_cloglog_points(
        train_tif,
        pres_val,
        bounds,
        mapping,
        vis_dir / f"train_cloglog_val_points_{test_year}.png",
        f"{season} val points – test {test_year}",
        edgecolor="green",
    )
    plot_cloglog_points(
        test_tif,
        pres_test,
        bounds,
        [mt.YearOffset(test_year, 0, 0)],
        vis_dir / f"test_cloglog_test_points_{test_year}.png",
        f"{season} test points – test {test_year}",
        edgecolor="blue",
        label_years=False,
    )

    y_true_train = np.concatenate([np.ones(len(env_pres)), np.zeros(len(env_bg))])
    y_score_train = np.concatenate([model.predict(env_pres), model.predict(env_bg)])
    y_true_val = np.concatenate([np.ones(len(env_pres_val)), np.zeros(len(env_bg_val))])
    y_score_val = np.concatenate(
        [model.predict(env_pres_val), model.predict(env_bg_val)]
    )
    y_true_test = np.concatenate([np.ones(len(X_pres)), np.zeros(len(X_bg))])
    y_score_test = np.concatenate([model.predict(X_pres), model.predict(X_bg)])
    train_gdf = gpd.GeoDataFrame(
        pd.concat([pres_train, bg_train], ignore_index=True), crs=pres_train.crs
    )
    val_gdf = gpd.GeoDataFrame(
        pd.concat([pres_val, bg_val], ignore_index=True), crs=pres_val.crs
    )
    test_gdf = gpd.GeoDataFrame(
        pd.concat([pres_test, bg_test], ignore_index=True), crs=pres_test.crs
    )
    n_params = mt.count_model_params(model)
    evaluate(
        f"{season}_test_{test_year}",
        y_true_train,
        y_score_train,
        cfg_run,
        y_true_test,
        y_score_test,
        y_true_val,
        y_score_val,
        n_params=n_params,
        gdf_train=train_gdf,
        gdf_test=test_gdf,
        gdf_val=val_gdf,
    )
    fig_dir = Path(cfg_run["outputs"]["figures_dir"])
    safe_name = sanitize_name(f"{season}_test_{test_year}")
    for name in [f"{safe_name}_threshold_auc.png", f"{safe_name}_threshold_cbi.png"]:
        src = fig_dir / name
        if src.exists():
            shutil.copy(src, vis_dir / name)
    thr = cfg_run["evaluation"].get("classification_threshold", 0.5)
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
            **train_metrics,
        }
    )
    metrics_records.append(
        {
            "season": season,
            "test_year": test_year,
            "split": "validation",
            **val_metrics,
        }
    )
    metrics_records.append(
        {
            "season": season,
            "test_year": test_year,
            "split": "test",
            **test_metrics,
        }
    )

    model_dir = Path(cfg_run["outputs"]["model_dir"])
    model_file = model_dir / f"{sanitize_name(cfg_run['experiment']['name'])}_model.pkl"
    with open(model_file, "wb") as f:
        pickle.dump(model, f)
    log.info("Model weights saved to %s", model_file)

    if (
        cfg_run.get("variable_importance", {})
        .get("permutation", {})
        .get("enable", False)
    ):
        permutation_importance(
            cfg_run,
            model,
            stack_path,
            vars_used,
            pres_train,
            bg_train,
            run_dir / "importance",
        )

    logging.getLogger().removeHandler(fh)
    fh.close()

    metrics_df = pd.DataFrame(metrics_records)
    metrics_path = out_root / "temporal_cv_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    log.info("Temporal CV metrics written to %s", metrics_path)
    summary_records: list[dict] = []
    for split, grp in metrics_df.groupby("split"):
        summary = {"split": split}
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
            summary[f"{metric}_ci95"] = mt._confidence_interval(vals)
        summary_records.append(summary)
    summary_path = out_root / "temporal_cv_summary.csv"
    pd.DataFrame(summary_records).to_csv(summary_path, index=False)
    log.info("Temporal CV summary written to %s", summary_path)


# ---------------------------------------------------------------------------
def _run_workflow(
    cfg: dict,
    *,
    suffix: str,
    include_positive_control: bool,
    rebuild: bool,
    test_year: int,
) -> None:
    """Execute the workflow for all configured seasons."""

    cfg_copy = copy.deepcopy(cfg)
    cfg_copy.setdefault("experiment", {})
    base_name = cfg_copy["experiment"].get("name", "multitemporal")
    cfg_copy["experiment"]["name"] = f"{base_name}_{suffix}"
    cfg_copy["experiment"].setdefault("append_timestamp", False)
    cfg_copy["run_test_year"] = test_year

    for season in tqdm(cfg_copy["multitemporal"]["seasons"], desc="Seasons", unit="season"):
        log.info(
            "Running season %s (%s)",
            season,
            "positive control" if include_positive_control else "baseline",
        )
        _process_season(
            cfg_copy,
            season,
            rebuild,
            test_year=test_year,
            include_positive_control=include_positive_control,
        )


# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run multitemporal positive control experiments for 2021"
    )
    parser.add_argument("--config", required=True, help="Path to configuration YAML")
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level (DEBUG, INFO, WARNING, ERROR)",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Force rebuilding of merged predictor stacks",
    )
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))
    cfg = load_config(args.config)

    test_year = 2021
    _run_workflow(
        cfg,
        suffix="positive_control",
        include_positive_control=True,
        rebuild=args.rebuild,
        test_year=test_year,
    )


if __name__ == "__main__":
    main()

