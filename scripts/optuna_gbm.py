"""Optuna study to tune LightGBM hyperparameters for the GBM engine."""
from __future__ import annotations

import argparse
import copy
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import pandas as pd
import yaml
from elapid import GeographicKFold
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

from scripts.bias_grid import create_bias_grid
from scripts.background_sampling import sample_background
from scripts.evaluation import _compute_metrics
from scripts.gbm_training import _prepare_training_matrix, fit_lightgbm
from scripts.preprocessing import build_stack
from scripts.thinning import run_thinning
from scripts.utils import (
    ensure_dir,
    extract_from_stack,
    get_processed_dir,
    load_config,
)
from scripts.variable_selection import run_selection

log = logging.getLogger(__name__)


def _suggest_param(trial: optuna.Trial, name: str, spec: dict[str, Any]) -> Any:
    if "choices" in spec:
        return trial.suggest_categorical(name, spec["choices"])
    low, high = spec.get("low"), spec.get("high")
    step = spec.get("step")
    log_scale = spec.get("log", False)
    if step is not None and isinstance(step, int):
        return trial.suggest_int(name, int(low), int(high), step=step)
    if spec.get("dtype") == "int":
        return trial.suggest_int(name, int(low), int(high), step=step or 1)
    if isinstance(low, int) and isinstance(high, int) and not log_scale and step is None:
        return trial.suggest_int(name, int(low), int(high))
    return trial.suggest_float(name, float(low), float(high), step=step, log=log_scale)


def objective(
    trial: optuna.Trial,
    cfg_base: dict,
    static_parts: dict[str, Any],
    exp_dir: Path,
) -> float:
    cfg = copy.deepcopy(cfg_base)
    trial_dir = exp_dir / f"trial_{trial.number:03d}"
    cfg.setdefault("experiment", {})
    cfg["experiment"]["name"] = f"{cfg['experiment'].get('name', 'gbm_optuna')}_t{trial.number:03d}"
    cfg["experiment"]["vis_dir"] = str(trial_dir / "visualizations")
    cfg["experiment"]["fi_dir"] = str(trial_dir / "feature_importance")
    cfg["experiment"]["stats_dir"] = str(trial_dir / "stats")
    cfg.setdefault("outputs", {})
    cfg["outputs"]["logs_dir"] = str(trial_dir / "logs")
    for key in ["model_dir", "maps_dir", "figures_dir", "root"]:
        cfg["outputs"][key] = str(trial_dir / "outputs" / key)
    for path in [
        cfg["experiment"]["vis_dir"],
        cfg["experiment"]["fi_dir"],
        cfg["experiment"]["stats_dir"],
        cfg["outputs"]["logs_dir"],
        cfg["outputs"]["model_dir"],
        cfg["outputs"]["maps_dir"],
        cfg["outputs"]["figures_dir"],
        cfg["outputs"]["root"],
    ]:
        ensure_dir(path)

    search = cfg.get("optuna", {}).get("search_space", {})
    gbm_cfg = cfg.setdefault("gbm", {})
    for param_name, spec in search.items():
        gbm_cfg[param_name] = _suggest_param(trial, param_name, spec)

    gkf = GeographicKFold(n_splits=cfg.get("optuna", {}).get("n_splits", 5))
    stack_path: Path = static_parts["stack_path"]
    base_vars: list[str] = static_parts["vars_used"]
    pres_gdf = static_parts["pres_gdf"]
    bg_gdf = static_parts["bg_gdf"]

    aucs: list[float] = []
    pr_aucs: list[float] = []
    cbis: list[float] = []
    omissions: list[float] = []

    for fold, (train_idx, test_idx) in enumerate(gkf.split(pres_gdf), start=1):
        pres_train = pres_gdf.iloc[train_idx]
        pres_test = pres_gdf.iloc[test_idx]
        cfg_fold = copy.deepcopy(cfg)
        cfg_fold["experiment"]["name"] = f"{cfg['experiment']['name']}_f{fold}"
        fold_dir = trial_dir / f"fold_{fold:02d}"
        cfg_fold["experiment"]["vis_dir"] = str(fold_dir / "visualizations")
        ensure_dir(cfg_fold["experiment"]["vis_dir"])

        try:
            X_train, y_train, vars_clean, weights, _ = _prepare_training_matrix(
                cfg_fold,
                stack_path,
                base_vars,
                pres_train,
                bg_gdf,
            )
        except Exception as exc:
            log.warning("Fold %d failed during matrix preparation: %s", fold, exc)
            continue

        model, _ = fit_lightgbm(cfg_fold, X_train, y_train, vars_clean, weights)

        pres_vals = extract_from_stack(stack_path, pres_test, vars_clean)
        bg_vals = extract_from_stack(stack_path, bg_gdf, vars_clean)
        mask_pres = ~np.any(np.isnan(pres_vals), axis=1)
        mask_bg = ~np.any(np.isnan(bg_vals), axis=1)
        if not np.any(mask_pres):
            log.warning("Fold %d lacks valid presence samples after extraction", fold)
            continue
        if not np.any(mask_bg):
            log.warning("Fold %d lacks valid background samples after extraction", fold)
            continue
        pres_vals = pres_vals[mask_pres]
        bg_vals = bg_vals[mask_bg]

        y_true = np.concatenate([
            np.ones(pres_vals.shape[0], dtype=int),
            np.zeros(bg_vals.shape[0], dtype=int),
        ])
        y_score = np.concatenate([
            model.predict(pres_vals),
            model.predict(bg_vals),
        ])
        mask_valid = ~np.isnan(y_score)
        if not np.all(mask_valid):
            dropped = int(np.sum(~mask_valid))
            log.warning("Fold %d produced %d NaN predictions", fold, dropped)
            y_true = y_true[mask_valid]
            y_score = y_score[mask_valid]
        if len(np.unique(y_true)) < 2:
            log.warning("Fold %d lacks both classes after filtering", fold)
            continue

        metrics = _compute_metrics(y_true, y_score, threshold="auto")
        aucs.append(metrics.get("auc", float("nan")))
        pr_aucs.append(metrics.get("pr_auc", float("nan")))
        cbis.append(metrics.get("cbi", float("nan")))
        omissions.append(metrics.get("omission", float("nan")))

    if not aucs:
        return float("nan")

    auc_val = float(np.nanmean(aucs))
    pr_val = float(np.nanmean(pr_aucs))
    cbi_val = float(np.nanmean(cbis))
    omission_val = float(np.nanmean(omissions))

    metric = cfg.get("optuna", {}).get("objective_metric", "auc").lower()
    if metric == "boyce":
        value = cbi_val
    elif metric == "boyce+auc":
        value = (cbi_val + auc_val) / 2.0
    elif metric == "omission":
        value = omission_val
    else:
        value = auc_val

    stats_file = Path(cfg["experiment"]["stats_dir"]) / "metrics.csv"
    ensure_dir(stats_file.parent)
    header = not stats_file.exists()
    with stats_file.open("a", newline="") as fh:
        if header:
            fh.write("trial,auc,pr_auc,cbi,omission\n")
        fh.write(
            f"{trial.number},{auc_val:.6f},{pr_val:.6f},{cbi_val:.6f},{omission_val:.6f}\n"
        )

    Path(trial_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    log.info(
        "Trial %d finished with value %.4f (AUC %.3f, PR-AUC %.3f, CBI %.3f)",
        trial.number,
        value,
        auc_val,
        pr_val,
        cbi_val,
    )
    return value


def run_study(
    cfg_input: Path | dict = Path("config.yaml"),
    suffix: str | None = None,
    prebuilt_stack: Path | None = None,
) -> tuple[Path, dict[str, Any]]:
    if isinstance(cfg_input, (str, Path)):
        cfg = load_config(cfg_input)
        cfg_name = Path(cfg_input).stem
    else:
        cfg = copy.deepcopy(cfg_input)
        cfg_name = "gbm_study"

    opt_cfg = cfg.setdefault("optuna", {})
    opt_cfg.setdefault("objective_metric", "auc")
    opt_cfg.setdefault("n_trials", 50)
    opt_cfg.setdefault("direction", "maximize")
    opt_cfg.setdefault("study_name", cfg_name)

    if suffix is None:
        suffix = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = f"{opt_cfg['study_name']}_{suffix}"
    exp_dir = Path("exps") / opt_cfg["study_name"] / exp_name
    ensure_dir(exp_dir)

    cfg.setdefault("experiment", {})["name"] = exp_name
    cfg["experiment"]["vis_dir"] = str(exp_dir / "visualizations")
    cfg["experiment"]["fi_dir"] = str(exp_dir / "feature_importance")
    cfg["experiment"]["stats_dir"] = str(exp_dir / "stats")
    cfg.setdefault("outputs", {})
    cfg["outputs"]["logs_dir"] = str(exp_dir / "logs")
    cfg["outputs"]["model_dir"] = str(exp_dir / "models")
    cfg["outputs"]["maps_dir"] = str(exp_dir / "maps")
    cfg["outputs"]["figures_dir"] = str(exp_dir / "figures")
    cfg["outputs"]["root"] = str(exp_dir / "outputs")
    for key in [
        cfg["experiment"]["vis_dir"],
        cfg["experiment"]["fi_dir"],
        cfg["experiment"]["stats_dir"],
        cfg["outputs"]["logs_dir"],
        cfg["outputs"]["model_dir"],
        cfg["outputs"]["maps_dir"],
        cfg["outputs"]["figures_dir"],
        cfg["outputs"]["root"],
    ]:
        ensure_dir(key)

    cfg["optuna"]["storage"] = f"sqlite:///{exp_dir}/study.db"
    Path(exp_dir / "study_config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))

    sampler = TPESampler(
        multivariate=True,
        group=True,
        seed=cfg["experiment"].get("random_seed", 0),
    )
    pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=0)
    study = optuna.create_study(
        study_name=opt_cfg["study_name"],
        storage=cfg["optuna"]["storage"],
        direction=opt_cfg["direction"],
        sampler=sampler,
        pruner=pruner,
        load_if_exists=True,
    )

    stack_path = prebuilt_stack if prebuilt_stack is not None else build_stack(cfg)
    vars_used = run_selection(cfg, stack_path)

    need_env = any(
        m.get("enable", True) and m.get("name") == "mahalanobis"
        for m in cfg["presence_data"].get("thinning", {}).get("methods", [])
    )
    env_df = None
    if need_env:
        raw_df = pd.read_csv(cfg["presence_data"]["file"], encoding="utf-8-sig")
        raw_df = raw_df.rename(
            columns={
                "dd long": "x",
                "dd lat": "y",
                "year-int": "year",
                "month-int": "month_int",
                "source": "dataset",
            }
        )
        raw_df["x"] = pd.to_numeric(raw_df["x"], errors="coerce")
        raw_df["y"] = pd.to_numeric(raw_df["y"], errors="coerce")
        raw_df = raw_df.dropna(subset=["x", "y"]).reset_index(drop=True)
        from scripts.utils import load_mask, filter_points_by_mask

        mask = load_mask(cfg["boundary_mask"])
        raw_df = filter_points_by_mask(raw_df, mask)
        import geopandas as gpd

        gdf_temp = gpd.GeoDataFrame(
            raw_df,
            geometry=gpd.points_from_xy(raw_df["x"], raw_df["y"]),
            crs="EPSG:2154",
        )
        env_vals = extract_from_stack(stack_path, gdf_temp, vars_used)
        env_df = pd.DataFrame(env_vals, columns=vars_used)

    pres_gdf = run_thinning(cfg, env_df=env_df)
    if cfg.get("bias_grid", {}).get("enable", False):
        bias_tif = create_bias_grid(pres_gdf, cfg)
    else:
        bias_tif = Path()
    bg_gdf = sample_background(cfg, bias_tif if bias_tif.exists() else None)

    static_parts = {
        "stack_path": stack_path,
        "vars_used": vars_used,
        "pres_gdf": pres_gdf,
        "bg_gdf": bg_gdf,
    }

    def _cb(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        print(f"Trial {trial.number}: value={trial.value:.4f} params={trial.params}")

    study.optimize(
        lambda t: objective(t, cfg, static_parts, exp_dir),
        n_trials=cfg["optuna"]["n_trials"],
        callbacks=[_cb],
    )

    best_params = {"value": study.best_trial.value, **study.best_trial.params}
    Path(exp_dir / "best_params.yaml").write_text(yaml.safe_dump(best_params, sort_keys=False))

    try:
        import optuna.visualization.matplotlib as oviz
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig_dir = exp_dir / "figures"
        ensure_dir(fig_dir)
        fig1 = oviz.plot_optimization_history(study)
        fig1.figure.savefig(fig_dir / "optimization_history.png", dpi=150)
        plt.close(fig1.figure)
        fig2 = oviz.plot_param_importances(study)
        fig2.figure.savefig(fig_dir / "param_importance.png", dpi=150)
        plt.close(fig2.figure)
    except Exception as exc:  # pragma: no cover - optional visualisations
        log.warning("Failed to create Optuna visualisations: %s", exc)

    print("Best trial", study.best_trial.number, study.best_trial.value, study.best_trial.params)
    return exp_dir, study.best_trial.params


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    parser = argparse.ArgumentParser(description="Tune LightGBM hyperparameters with Optuna")
    parser.add_argument("--config", default="configtuning.yaml", help="YAML tuning configuration")
    args = parser.parse_args()

    cfg_full = load_config(args.config)
    tune_cfg = cfg_full.get("tuning", {})
    if tune_cfg.get("search_space"):
        cfg_full.setdefault("optuna", {}).setdefault("search_space", {}).update(
            tune_cfg["search_space"]
        )

    years = tune_cfg.get(
        "years",
        [cfg_full["predictors"]["variable_selection"].get("reference_year", 2022)],
    )
    seasons_opt = tune_cfg.get("seasons")
    dataset_opts = tune_cfg.get(
        "datasets",
        [
            {
                "sources": cfg_full["presence_data"].get("source_filter"),
                "bias_map": cfg_full.get("bias_grid", {}).get("file"),
            }
        ],
    )

    prebuilt = None
    stack_years = sorted(set(years))
    stack_cfg = copy.deepcopy(cfg_full)
    stack_cfg.setdefault("experiment", {})["name"] = "prebuilt"
    stack_cfg["outputs"]["root"] = str(get_processed_dir())
    for layer in ["landcover_fraction", "veg_index", "hunting_bag"]:
        if layer in stack_cfg["predictors"]["layers"]:
            stack_cfg["predictors"]["layers"][layer]["years"] = stack_years
    prebuilt = build_stack(stack_cfg)

    summary_rows = []

    for ds in dataset_opts:
        cfg_ds = copy.deepcopy(cfg_full)
        cfg_ds["presence_data"]["source_filter"] = ds.get("sources")
        bias_map = ds.get("bias_map")
        if bias_map:
            cfg_ds.setdefault("bias_grid", {})["strategy"] = "predefined"
            if "{season}" in bias_map:
                cfg_ds["bias_grid"]["pattern"] = bias_map
            else:
                cfg_ds["bias_grid"]["file"] = bias_map

        for year in years:
            cfg_year = copy.deepcopy(cfg_ds)
            cfg_year["predictors"]["variable_selection"]["reference_year"] = year
            for layer in ["landcover_fraction", "veg_index", "hunting_bag"]:
                if layer in cfg_year["predictors"]["layers"]:
                    cfg_year["predictors"]["layers"][layer]["years"] = [year]

            season_sets = seasons_opt or [[], [f"{year}-Summer"], [f"{year}-Winter"]]
            for seasons in season_sets:
                cfg_year["seasons"]["active"] = seasons
                exp_dir, params = run_study(cfg_year, suffix=f"{year}_{len(seasons)}", prebuilt_stack=prebuilt)
                summary_rows.append(
                    {
                        "year": year,
                        "seasons": ",".join(seasons) if seasons else "all",
                        "best_value": params.get("value"),
                        **{f"param_{k}": v for k, v in params.items() if k != "value"},
                        "experiment_dir": str(exp_dir),
                    }
                )

    if summary_rows:
        df = pd.DataFrame(summary_rows)
        out_file = Path("exps") / "gbm_optuna_summary.csv"
        ensure_dir(out_file.parent)
        df.to_csv(out_file, index=False)
if __name__ == "__main__":
    main()
