# =============================================================================
# Project Title: Wild Boar Distribution Model for France
# Path: scripts/optuna_tuning.py
# Purpose: Wrap Optuna for hyper-parameter optimisation of the MaxEnt model.
# Process Step: Performs search over settings and stores best parameter sets.
# Created by Adrian Meyer, Institute Geomatics, FHNW (adrian.meyer@fhnw.ch)
# Created: July 2025
# Version: v.0.1.0
# =============================================================================
"""
Optuna study wrapper for hyper-parameter optimisation.
"""
from __future__ import annotations
import os

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

import logging
import optuna
from pathlib import Path
from scripts.maxent_training import train_model
from scripts.variable_selection import run_selection
from scripts.utils import (
    load_config,
    extract_from_stack,
    ensure_dir,
    get_processed_dir,
    count_model_params,
)
from scripts.evaluation import _compute_metrics
from scripts.preprocessing import build_stack
import numpy as np
import argparse
import yaml
import copy
from elapid import GeographicKFold
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
import pandas as pd

from datetime import datetime


log = logging.getLogger(__name__)

# -----------------------------------------------------------------------------  
def objective(trial: optuna.Trial, cfg: dict, static_parts: dict, exp_dir: Path):
    """Single Optuna objective evaluating one MaxEnt configuration."""
    cfg = copy.deepcopy(cfg)  # keep base config intact

    search = cfg.get("optuna", {}).get("search_space", {})

    trial_name = f"{cfg['experiment']['name']}_t{trial.number:03d}"
    trial_dir = exp_dir / f"trial_{trial.number:03d}"
    cfg["experiment"]["name"] = trial_name
    cfg["outputs"]["root"] = str(trial_dir)
    cfg["outputs"]["logs_dir"] = str(trial_dir / "logs")
    cfg["outputs"]["model_dir"] = str(trial_dir / "models")
    cfg["outputs"]["maps_dir"] = str(trial_dir / "maps")
    cfg["outputs"]["figures_dir"] = str(trial_dir / "figures")
    cfg["experiment"]["vis_dir"] = str(trial_dir / "visualizations")
    cfg["experiment"]["fi_dir"] = str(trial_dir / "feature_importance")
    cfg["experiment"]["stats_dir"] = str(trial_dir / "stats")

    for p in [
        cfg["outputs"]["logs_dir"],
        cfg["outputs"]["model_dir"],
        cfg["outputs"]["maps_dir"],
        cfg["outputs"]["figures_dir"],
        cfg["experiment"]["vis_dir"],
        cfg["experiment"]["fi_dir"],
        cfg["experiment"]["stats_dir"],
    ]:

        ensure_dir(p)

    if "regularization_multiplier" in search:
        ss = search["regularization_multiplier"]
        cfg["maxent"]["regularization_multiplier"] = trial.suggest_float(
            "reg_mu", ss["low"], ss["high"], step=ss.get("step"), log=ss.get("log", False)
        )

    if "feature_types" in search:
        cfg["maxent"]["feature_types"] = trial.suggest_categorical(
            "features", search["feature_types"]["choices"]
        )

    if "num_background" in search:
        ss = search["num_background"]
        cfg["background"]["n_points"] = trial.suggest_int(
            "n_bg", ss["low"], ss["high"], step=ss.get("step", 1)
        )

    if "n_hinge_features" in search:
        ss = search["n_hinge_features"]
        cfg["maxent"]["n_hinge_features"] = trial.suggest_int(
            "n_hinge", ss["low"], ss["high"], step=ss.get("step", 1)
        )

    if "beta_lqp" in search:
        ss = search["beta_lqp"]
        cfg["maxent"]["beta_lqp"] = trial.suggest_float(
            "beta_lqp", ss["low"], ss["high"], step=ss.get("step")
        )

    if "max_iterations" in search:
        ss = search["max_iterations"]
        cfg["maxent"]["max_iterations"] = trial.suggest_int(
            "max_iter", ss["low"], ss["high"], step=ss.get("step", 1)
        )

    if "convergence_tolerance" in search:
        ss = search["convergence_tolerance"]
        low, high = float(ss["low"]), float(ss["high"])
        if high < low:
            log.warning(
                "Swapping convergence_tolerance bounds low=%s high=%s", low, high
            )
            low, high = high, low
        cfg["maxent"]["convergence_tolerance"] = trial.suggest_float(
            "conv_tol", low, high, log=ss.get("log", False)
        )

    print(f"Starting trial {trial.number} with parameters:")
    print(yaml.safe_dump({
        "regularization_multiplier": cfg["maxent"].get("regularization_multiplier"),
        "feature_types": cfg["maxent"].get("feature_types"),
        "num_background": cfg["background"].get("n_points"),
        "n_hinge_features": cfg["maxent"].get("n_hinge_features"),
        "beta_lqp": cfg["maxent"].get("beta_lqp"),
        "max_iterations": cfg["maxent"].get("max_iterations"),
        "convergence_tolerance": cfg["maxent"].get("convergence_tolerance"),
    }, sort_keys=False))

    gkf = GeographicKFold(n_splits=5)
    aucs, pr_aucs, cbis, aiccs, omissions = [], [], [], [], []
    base_vars = static_parts["vars_used"]
    for fold, (train_idx, test_idx) in enumerate(gkf.split(static_parts["pres_gdf"]), start=1):
        pres_train = static_parts["pres_gdf"].iloc[train_idx]
        pres_test = static_parts["pres_gdf"].iloc[test_idx]
        fold_cfg = copy.deepcopy(cfg)
        # Give each fold a unique experiment name so that artifacts like
        # rasters and figures include the fold index in their filenames
        # and do not overwrite one another in the trial's output
        # directories.
        fold_cfg["experiment"]["name"] = f"{cfg['experiment']['name']}_f{fold}"
        model, _, used = train_model(
            fold_cfg,
            static_parts["stack_path"],
            base_vars,
            pres_train,
            static_parts["bg_gdf"],
        )
        y_pres = model.predict(
            extract_from_stack(static_parts["stack_path"], pres_test, used)
        )
        y_bg = model.predict(
            extract_from_stack(static_parts["stack_path"], static_parts["bg_gdf"], used)
        )
        y_true = np.concatenate([np.ones_like(y_pres), np.zeros_like(y_bg)])
        y_score = np.concatenate([y_pres, y_bg])
        mask = ~np.isnan(y_score)
        if not np.all(mask):
            dropped = int(np.sum(~mask))
            log.warning(
                "Trial %d fold %d contains %d NaN predictions - excluding from metrics",
                trial.number,
                fold,
                dropped,
            )
        y_true = y_true[mask]
        y_score = y_score[mask]
        if len(np.unique(y_true)) < 2:
            log.warning(
                "Trial %d fold %d lacks positive or negative samples after filtering",
                trial.number,
                fold,
            )
            continue
        n_params = count_model_params(model)
        thr = cfg.get("evaluation", {}).get("classification_threshold", 0.5)
        metrics = _compute_metrics(y_true, y_score, threshold=thr, n_params=n_params)
        aucs.append(metrics["auc"])
        pr_aucs.append(metrics["pr_auc"])
        cbis.append(metrics["cbi"])
        aiccs.append(metrics["aicc"])
        omissions.append(metrics["omission"])

    if not aucs:
        return float("nan")

    auc_val = float(np.mean(aucs)) if aucs else float("nan")
    pr_val = float(np.mean(pr_aucs)) if pr_aucs else float("nan")
    cbi_val = float(np.mean(cbis)) if cbis else float("nan")
    aicc_val = float(np.mean(aiccs)) if aiccs else float("nan")
    omission_val = float(np.mean(omissions)) if omissions else float("nan")

    obj_metric = cfg.get("optuna", {}).get("objective_metric", "boyce").lower()
    if obj_metric == "boyce":
        value = cbi_val
    elif obj_metric == "auc":
        value = auc_val
    elif obj_metric == "boyce+auc":
        value = (cbi_val + auc_val) / 2.0
    elif obj_metric == "aicc":
        value = aicc_val
    elif obj_metric == "omission":
        value = omission_val
    else:
        value = auc_val

    print(
        f"Trial {trial.number} metrics: AUC {auc_val:.3f} PR-AUC {pr_val:.3f} CBI {cbi_val:.3f} AICc {aicc_val:.3f} Omission {omission_val:.3f}"
    )


    # Save metrics for later reference
    stats_file = Path(cfg["experiment"]["stats_dir"]) / "metrics.csv"
    header_needed = not stats_file.exists()
    with stats_file.open("a", newline="") as fh:
        if header_needed:
            fh.write("trial,auc,pr_auc,cbi,aicc,omission\n")
        fh.write(
            f"{trial.number},{auc_val:.6f},{pr_val:.6f},{cbi_val:.6f},{aicc_val:.6f},{omission_val:.6f}\n"
        )

    # Save lambda weights if available
    lambdas = getattr(model, "lambdas_", getattr(model, "lambdas", None))
    if lambdas is not None:
        np.savetxt(trial_dir / "lambda_weights.csv", lambdas, delimiter=",")

    # Persist trial configuration
    Path(trial_dir / "config.yaml").write_text(yaml.safe_dump(cfg))

    log.info("Trial %d finished with %.4f", trial.number, value)
    return value

# -----------------------------------------------------------------------------  

def run_study(
    cfg_file: Path | dict = Path("configs/legacy/config.yaml"),
    suffix: str | None = None,
    prebuilt_stack: Path | None = None,
):

    if isinstance(cfg_file, (str, Path)):
        cfg = load_config(cfg_file)
        cfg_name = Path(cfg_file).stem
    else:
        cfg = cfg_file
        cfg_name = "study"

    opt_cfg = cfg.setdefault("optuna", {})
    metric = opt_cfg.setdefault("objective_metric", "boyce").lower()

    study_name = opt_cfg.get("study_name", cfg_name)
    opt_cfg.setdefault("study_name", study_name)
    opt_cfg.setdefault("storage", None)
    opt_cfg.setdefault("n_trials", 50)
    if metric in {"aicc", "omission"}:
        opt_cfg["direction"] = "minimize"
    else:
        opt_cfg["direction"] = "maximize"
    if not suffix:
        suffix = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = f"{study_name}_{suffix}"
    exp_dir = Path("exps") / study_name / exp_name
    ensure_dir(exp_dir)
    cfg.setdefault("experiment", {})["name"] = exp_name

    # Set up additional experiment directories
    vis_dir = exp_dir / "visualizations"
    fi_dir = exp_dir / "feature_importance"
    stats_dir = exp_dir / "stats"
    cfg["experiment"]["vis_dir"] = str(vis_dir)
    cfg["experiment"]["fi_dir"] = str(fi_dir)
    cfg["experiment"]["stats_dir"] = str(stats_dir)
    for p in [vis_dir, fi_dir, stats_dir]:
        ensure_dir(p)

    # place the Optuna DB inside the experiment directory
    cfg["optuna"]["storage"] = f"sqlite:///{exp_dir}/study.db"

    def _update_paths(d: dict) -> None:
        for k, v in d.items():
            if isinstance(v, dict):
                _update_paths(v)
            elif isinstance(v, str):
                if v.startswith("outputs/"):
                    d[k] = str(exp_dir / v)

    _update_paths(cfg)
    cfg.setdefault("maxent", {})
    threads = cfg["maxent"].get("threads", _DEFAULT_THREADS)
    if threads == -1:
        threads = os.cpu_count() or 1
    cfg["maxent"]["threads"] = threads
    log.info("Using %d CPU cores for training", threads)
    for ev in [
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "BLIS_NUM_THREADS",
    ]:
        os.environ[ev] = str(threads)
    os.environ["LOGS_DIR"] = str(Path(cfg["outputs"]["logs_dir"]))
    # ensure all output folders exist before creating the Optuna DB
    for p in [
        cfg["outputs"]["logs_dir"],
        cfg["outputs"]["model_dir"],
        cfg["outputs"]["maps_dir"],
        cfg["outputs"]["figures_dir"],
        cfg["experiment"]["vis_dir"],
        cfg["experiment"]["fi_dir"],
        cfg["experiment"]["stats_dir"],
    ]:
        ensure_dir(p)

    # Save the full configuration used for this study
    Path(exp_dir / "study_config.yaml").write_text(yaml.safe_dump(cfg))

    print("Running study with configuration:\n" + yaml.safe_dump(cfg))

    sampler = TPESampler(
        multivariate=True, group=True, seed=cfg["experiment"].get("random_seed", 0)
    )
    pruner = MedianPruner(n_startup_trials=10, n_warmup_steps=0)
    study = optuna.create_study(
        study_name=cfg["optuna"]["study_name"],
        storage=cfg["optuna"]["storage"],
        direction=cfg["optuna"]["direction"],
        load_if_exists=True,
        sampler=sampler,
        pruner=pruner,
    )
    from scripts.thinning import run_thinning
    from scripts.bias_grid import create_bias_grid
    from scripts.background_sampling import sample_background

    stack_path = prebuilt_stack if prebuilt_stack is not None else build_stack(cfg)
    vars_used = run_selection(cfg, stack_path)

    need_env = any(
        m.get("enable", True) and m.get("name") == "mahalanobis"
        for m in cfg["presence_data"]["thinning"].get("methods", [])
    )
    env_df = None
    if need_env:
        import pandas as pd
        import geopandas as gpd

        df = pd.read_csv(cfg["presence_data"]["file"], encoding="utf-8-sig")
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
        df = df.dropna(subset=["x", "y"]).reset_index(drop=True)
        from scripts.utils import load_mask, filter_points_by_mask

        mask = load_mask(cfg["boundary_mask"])
        df = filter_points_by_mask(df, mask)
        gdf_temp = gpd.GeoDataFrame(
            df, geometry=gpd.points_from_xy(df["x"], df["y"]), crs="EPSG:2154"
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
        print(f"Trial {trial.number} -> {trial.value:.4f} using {trial.params}")

    study.optimize(
        lambda t: objective(t, cfg, static_parts, exp_dir),
        n_trials=cfg["optuna"]["n_trials"],
        callbacks=[_cb],
    )

    best_params = {"value": study.best_trial.value, **study.best_trial.params}
    Path(exp_dir / "best_params.yaml").write_text(yaml.safe_dump(best_params))

    # Generate Optuna visualizations in the study directory
    try:
        import optuna.visualization.matplotlib as oviz
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig_dir = Path(exp_dir) / "figures"
        ensure_dir(fig_dir)
        fig1 = oviz.plot_optimization_history(study)
        fig1.figure.savefig(fig_dir / "optimization_history.png", dpi=150)
        plt.close(fig1.figure)
        fig2 = oviz.plot_param_importances(study)
        fig2.figure.savefig(fig_dir / "param_importance.png", dpi=150)
        plt.close(fig2.figure)
    except Exception as exc:  # pragma: no cover - optional
        log.warning("Failed to create Optuna plots: %s", exc)

    print("Best trial:", study.best_trial.number, study.best_trial.value, study.best_trial.params)
    return exp_dir, study.best_trial.params


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--config",
        default="configs/tuning/configtuning.yaml",
        help="Combined tuning configuration YAML",
    )
    args = ap.parse_args()

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
    corr_vals = tune_cfg.get(
        "correlation_cutoffs",
        [cfg_full["predictors"]["variable_selection"].get("correlation_cutoff", 0.7)],
    )
    bias_opts = tune_cfg.get(
        "bias_strategies",
        [cfg_full.get("bias_grid", {}).get("strategy", "predefined")],
    )
    thin_opts = tune_cfg.get("thinning_options")
    dataset_opts = tune_cfg.get(
        "datasets",
        [
            {
                "sources": cfg_full["presence_data"].get("source_filter"),
                "bias_map": cfg_full.get("bias_grid", {}).get("file"),
            }
        ],
    )
    bg_bias_opts = tune_cfg.get(
        "use_bias_grid_options",
        [cfg_full.get("background", {}).get("use_bias_grid", True)],
    )
    vif_vals = tune_cfg.get(
        "vif_thresholds",
        [cfg_full["predictors"]["variable_selection"].get("vif_threshold", 8)],
    )
    mah_cfg = tune_cfg.get("mahalanobis_thresholds")

    summary = []

    # ------------------------------------------------------------------
    # Build a combined predictor stack covering all requested years once
    # and reuse it for every Optuna study run to avoid redundant I/O.
    # ------------------------------------------------------------------
    stack_years = sorted(set(years))
    stack_cfg = copy.deepcopy(cfg_full)
    stack_cfg.setdefault("experiment", {})["name"] = "prebuilt"
    stack_cfg["outputs"]["root"] = str(get_processed_dir())
    for layer in ["landcover_fraction", "veg_index", "hunting_bag"]:
        if layer in stack_cfg["predictors"]["layers"]:
            stack_cfg["predictors"]["layers"][layer]["years"] = stack_years
    prebuilt_stack_path = build_stack(stack_cfg)

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

        sources = ds.get("sources") or []
        if isinstance(mah_cfg, dict):
            from itertools import product

            keys = sources or list(mah_cfg.keys())
            val_lists = [mah_cfg.get(k, [0.5]) for k in keys]
            mah_vals = [dict(zip(keys, combo)) for combo in product(*val_lists)]
        else:
            mah_vals = mah_cfg
            if not mah_vals:
                perc = 0.5
                for m in cfg_full["presence_data"]["thinning"]["methods"]:
                    if m["name"] == "mahalanobis":
                        perc = m["params"].get("percentage", perc)
                mah_vals = [perc]

        for year in years:
            cfg_year = copy.deepcopy(cfg_ds)
            cfg_year["predictors"]["variable_selection"]["reference_year"] = year
            for layer in ["landcover_fraction", "veg_index", "hunting_bag"]:
                if layer in cfg_year["predictors"]["layers"]:
                    cfg_year["predictors"]["layers"][layer]["years"] = [year]

            if seasons_opt:
                seasons_sets = [
                    [s.format(year=year) for s in season_list]
                    for season_list in seasons_opt
                ]
            else:
                seasons_sets = [[], [f"{year}-Summer"], [f"{year}-Winter"]]

            for seasons in seasons_sets:
                cfg_year["seasons"]["active"] = seasons
                for corr in corr_vals:
                    cfg_corr = copy.deepcopy(cfg_year)
                    cfg_corr["predictors"]["variable_selection"]["correlation_cutoff"] = corr

                    for bias in bias_opts:
                        cfg_bias = copy.deepcopy(cfg_corr)
                        cfg_bias.setdefault("bias_grid", {})
                        if bias == "predefined_all":
                            cfg_bias["bias_grid"]["strategy"] = "predefined"
                            cfg_bias["bias_grid"]["file"] = "data/bias-maps/Combined_All_Semesters.asc"
                        elif bias == "predefined_season":
                            cfg_bias["bias_grid"]["strategy"] = "predefined"
                            cfg_bias["bias_grid"]["pattern"] = "data/bias-maps/Combined_{season}.asc"
                        elif bias == "occurrence_density":
                            cfg_bias["bias_grid"]["strategy"] = "occurrence_density"
                            cfg_bias["bias_grid"]["kernel_bandwidth_km"] = tune_cfg.get("kde_bandwidth_km", 5)
                        for use_bg in bg_bias_opts:
                            cfg_bg = copy.deepcopy(cfg_bias)
                            cfg_bg.setdefault("background", {})["use_bias_grid"] = use_bg
                            thinning_sets = thin_opts if thin_opts else [cfg_bg["presence_data"]["thinning"]["methods"]]
                            for thinning in thinning_sets:
                                cfg_thin = copy.deepcopy(cfg_bg)
                                cfg_thin["presence_data"]["thinning"]["methods"] = thinning
                                for vif in vif_vals:
                                    cfg_vif = copy.deepcopy(cfg_thin)
                                    cfg_vif["predictors"]["variable_selection"]["vif_threshold"] = vif
                                    for mah in mah_vals:
                                        cfg_final = copy.deepcopy(cfg_vif)
                                        for m in cfg_final["presence_data"]["thinning"]["methods"]:
                                            if m["name"] == "mahalanobis":
                                                m["params"]["percentage"] = mah
                                        cfg_final["optuna"]["n_trials"] = tune_cfg.get(
                                            "n_trials",
                                            cfg_final["optuna"].get("n_trials", 50),
                                        )

                                        season_tag = "all" if not seasons else "-".join(seasons)
                                        thin_tag = len(thinning) if isinstance(thinning, list) else thinning
                                        src_tag = (
                                            "all"
                                            if not ds.get("sources")
                                            else "_".join(ds["sources"])
                                        )
                                        if isinstance(mah, dict):
                                            mah_tag = "_".join(
                                                f"{k}{int(v*100)}" for k, v in mah.items()
                                            )
                                        else:
                                            mah_tag = int(mah * 100)
                                        suffix = (
                                            f"src{src_tag}_y{year}_{season_tag}_corr{corr}_"
                                            f"{bias}_bg{int(use_bg)}_thin{thin_tag}_vif{vif}_mah{mah_tag}"
                                        )
                                        exp_dir, best = run_study(
                                            cfg_final,
                                            suffix=suffix,
                                            prebuilt_stack=prebuilt_stack_path,
                                        )
                                        mah_record = (
                                            mah_tag if isinstance(mah, dict) else mah
                                        )
                                        summary.append(
                                            {
                                                "sources": src_tag,
                                                "year": year,
                                                "season": season_tag,
                                                "corr": corr,
                                                "bias": bias,
                                                "use_bias_bg": use_bg,
                                                "thinning": thin_tag,
                                                "vif": vif,
                                                "mahalanobis": mah_record,
                                                **best,
                                            }
                                        )
    if summary:
        study_name = cfg_full.get("optuna", {}).get("study_name", "study")
        summary_dir = Path("exps") / study_name
        ensure_dir(summary_dir)
        pd.DataFrame(summary).to_csv(summary_dir / "best_params_summary.csv", index=False)


if __name__ == "__main__":
    main()
