# =============================================================================
# Project Title: Wild Boar Distribution Model for France
# Path: scripts/run_pipeline.py
# Purpose: Unified command-line front-end for running the modelling pipeline.
# Process Step: Orchestrates each stage using settings from the YAML config.
# Created by Adrian Meyer, Institute Geomatics, FHNW (adrian.meyer@fhnw.ch)
# Created: July 2025
# Version: v.0.1.0
# =============================================================================
"""
Unified command-line front-end:  python scripts/run_pipeline.py --config config.yaml
"""
from __future__ import annotations
import argparse
import json
import logging
from pathlib import Path
import sys
import os
import shutil
from datetime import datetime
from copy import deepcopy
from sklearn.model_selection import train_test_split
import yaml
import pandas as pd
import geopandas as gpd

# Allow running this script directly by adding the repository root to the
# Python module search path. This keeps absolute imports functional when the
# script is executed as ``python scripts/run_pipeline.py``.
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))



log = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
def _load_selected_vars(cfg: dict, stack_path: Path) -> list[str]:
    """Return list of variables when selection step is skipped.

    If the expected ``selected_predictors`` file is missing, fall back to all
    variables present in the predictor stack (excluding ``boundary_mask``) and
    write them to that file so downstream steps receive a sensible default.
    """

    sel_file = Path(cfg["predictors"]["variable_selection"]["save_selected_list"])
    if sel_file.exists():
        return sel_file.read_text().splitlines()

    import xarray as xr

    ds = xr.open_zarr(stack_path, consolidated=False)
    vars_used = [v for v in ds.data_vars if v != "boundary_mask"]
    sel_file.parent.mkdir(parents=True, exist_ok=True)
    sel_file.write_text("\n".join(vars_used))
    log.warning(
        "Selected predictor list %s missing – using all %d variables from stack",
        sel_file,
        len(vars_used),
    )
    return vars_used


def _load_thinned_presences(cfg: dict):
    """Return presence GeoDataFrame when thinning step is skipped.

    If the saved thinned file is absent or stale, the raw presence CSV is loaded,
    filtered, deduplicated and written to the expected GeoJSON file.
    """

    from scripts.thinning import thinning_signature
    from scripts.utils import (
        ensure_gdf_crs,
        load_mask,
        filter_points_by_mask,
        deduplicate_cells,
        ensure_dir,
    )
    import geopandas as gpd
    import pandas as pd

    out_file = Path(cfg["presence_data"]["thinning"]["save_thinned_points"])
    out_meta = Path(f"{out_file}.meta.json")
    signature = thinning_signature(cfg)

    def _read_signature(meta_path: Path) -> str | None:
        try:
            data = json.loads(meta_path.read_text())
        except FileNotFoundError:
            return None
        except json.JSONDecodeError:
            log.warning("Failed to parse thinning metadata from %s", meta_path)
            return None
        return data.get("signature")

    existing_sig = _read_signature(out_meta)
    if out_file.exists() and existing_sig == signature:
        gdf = gpd.read_file(out_file)
        gdf = ensure_gdf_crs(gdf, context="thinned file")
        mask = load_mask(cfg["boundary_mask"])
        df_temp = filter_points_by_mask(gdf.drop(columns="geometry"), mask)
        gdf = gpd.GeoDataFrame(
            df_temp,
            geometry=gpd.points_from_xy(df_temp["x"], df_temp["y"]),
            crs="EPSG:2154",
        )
        return deduplicate_cells(gdf)

    if out_file.exists():
        log.warning(
            "Discarding thinned presence file %s due to signature mismatch (expected %s, found %s)",
            out_file,
            signature,
            existing_sig,
        )
        out_file.unlink(missing_ok=True)
        out_meta.unlink(missing_ok=True)

    log.warning(
        "Thinned presence file %s not found – using raw presence data",
        out_file,
    )
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
    mask = load_mask(cfg["boundary_mask"])
    df = filter_points_by_mask(df, mask)
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df["x"], df["y"]), crs="EPSG:2154")
    gdf = deduplicate_cells(gdf)
    ensure_dir(out_file.parent)
    gdf.to_file(out_file, driver="GeoJSON")
    out_meta.write_text(json.dumps({"signature": signature}))
    return gdf
# -----------------------------------------------------------------------------  
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml", help="Path to YAML configuration")
    ap.add_argument(
        "--debug", action="store_true", help="Enable debug mode (limit to 2021 test year)"
    )
    args = ap.parse_args()

    raw_cfg = yaml.safe_load(Path(args.config).read_text())
    exp_section = raw_cfg.setdefault("experiment", {})
    debug_mode = bool(args.debug or exp_section.get("debug_mode"))
    exp_section["debug_mode"] = debug_mode
    if debug_mode:
        os.environ["RUN_PIPELINE_DEBUG"] = "1"
    else:
        os.environ.pop("RUN_PIPELINE_DEBUG", None)
    exp_name = exp_section.get("name")
    batch_name = exp_section.get("batch_name") or os.environ.get("RUN_PIPELINE_BATCH")
    if not batch_name:
        batch_name = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    exp_section["batch_name"] = batch_name
    leaf_name = exp_name if exp_name else "main"
    leaf_path = Path(leaf_name)
    combined_name = "/".join([batch_name, *leaf_path.parts])

    batch_dir = Path("exps") / batch_name
    exp_dir = batch_dir / leaf_path
    shared_dir = exp_dir / "shared"
    shared_outputs = shared_dir / "outputs"
    shared_vis_dir = shared_dir / "visualizations"
    shared_fi_dir = shared_dir / "feature_importance"
    shared_stats_dir = shared_dir / "stats"

    os.environ["RUN_PIPELINE_BATCH"] = batch_name

    maxent_threads = raw_cfg.get("maxent", {}).get("threads", -1)
    if maxent_threads == -1:
        maxent_threads = os.cpu_count() or 1
    for ev in [
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "BLIS_NUM_THREADS",
    ]:
        os.environ[ev] = str(maxent_threads)
    log.info("Using %d CPU cores for training", maxent_threads)

    shared_outputs.mkdir(parents=True, exist_ok=True)
    shared_vis_dir.mkdir(parents=True, exist_ok=True)
    shared_fi_dir.mkdir(parents=True, exist_ok=True)
    shared_stats_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = shared_outputs / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    os.environ["LOGS_DIR"] = str(logs_dir)

    config_archive = exp_dir / "config_used.yaml"
    config_archive.parent.mkdir(parents=True, exist_ok=True)
    try:
        shutil.copy2(args.config, config_archive)
    except Exception as exc:
        log.warning("Failed to archive configuration: %s", exc)

    from scripts.utils import (
        load_config,
        set_global_seed,
        extract_from_stack,
        ensure_gdf_crs,
        sanitize_name,
        count_model_params,
    )
    from scripts.visualization import (
        plot_points,
        plot_variables,
        plot_raster,
        plot_variable_histograms,
        export_response_curves,
    )
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from tqdm import tqdm
    from scripts.preprocessing import build_stack
    import pandas as pd
    from scripts.variable_selection import run_selection
    from scripts.thinning import run_thinning
    from scripts.bias_grid import create_bias_grid
    from scripts.background_sampling import sample_background
    from scripts.maxent_training import train_model as train_maxent_model
    from scripts.evaluation import evaluate
    from scripts.cross_validation import run_cv
    import geopandas as gpd
    import numpy as np

    cfg = load_config(args.config)
    cfg.setdefault("maxent", {})["threads"] = maxent_threads
    cfg.setdefault("experiment", {})
    cfg["experiment"]["name"] = combined_name
    cfg["experiment"]["batch_name"] = batch_name
    cfg["experiment"]["batch_dir"] = str(batch_dir)
    cfg["experiment"]["debug_mode"] = debug_mode
    cfg["experiment"]["vis_dir"] = str(shared_vis_dir)
    cfg["experiment"]["fi_dir"] = str(shared_fi_dir)
    cfg["experiment"]["stats_dir"] = str(shared_stats_dir)

    def _update_paths(d: dict) -> None:
        for k, v in d.items():
            if isinstance(v, dict):
                _update_paths(v)
            elif isinstance(v, str):
                if v.startswith("outputs/"):
                    rel = Path(v).relative_to("outputs")
                    d[k] = str(shared_outputs / rel)
                elif v.startswith("sqlite:///outputs/"):
                    rel = Path(v[len("sqlite:///"):]).relative_to("outputs")
                    d[k] = "sqlite:///" + str(shared_outputs / rel)

    _update_paths(cfg)

    def _rebase_outputs(d: dict, old_root: Path, new_root: Path) -> None:
        for key, value in d.items():
            if isinstance(value, dict):
                _rebase_outputs(value, old_root, new_root)
            elif isinstance(value, str):
                if value.startswith(str(old_root)):
                    d[key] = str(new_root / Path(value).relative_to(old_root))
                elif value.startswith("sqlite:///") and value[len("sqlite:///") :].startswith(
                    str(old_root)
                ):
                    rel = Path(value[len("sqlite:///") :]).relative_to(old_root)
                    d[key] = "sqlite:///" + str(new_root / rel)

    cfg.setdefault("model", {}).setdefault("engines", ["maxent", "gbm"])

    # Determine target year from selected seasons and update predictor settings
    seasons = cfg.get("seasons", {}).get("active", [])
    if seasons:
        years = [int(s.split("-")[0]) for s in seasons]
        last_year = max(years)
        last_entry = [s for s in seasons if s.startswith(str(last_year))][-1]
        last_season = last_entry.split("-")[1].lower()
        cfg["predictors"]["variable_selection"]["reference_year"] = last_year
        for group, spec in cfg.get("predictors", {}).get("layers", {}).items():
            if isinstance(spec, dict):
                if "years" in spec:
                    spec["years"] = [last_year]
                if "seasons" in spec:
                    spec["seasons"] = [last_season]

    set_global_seed(cfg["experiment"]["random_seed"])

    print("Starting pipeline")
    log.info("Starting pipeline")

    # 1. Raster stack
    stack_path = None
    if cfg["steps"].get("preprocessing", True):
        print("[1/8] Preprocessing")
        log.info("Step 1 - preprocessing")
        stack_path = build_stack(cfg)
        import pandas as pd
        import geopandas as gpd
        df_all = pd.read_csv(cfg["presence_data"]["file"], encoding="utf-8-sig")
        df_all = df_all.rename(columns={"dd long": "x", "dd lat": "y", "year-int": "year", "month-int": "month_int", "source": "dataset"})
        df_all["x"] = pd.to_numeric(df_all["x"], errors="coerce")
        df_all["y"] = pd.to_numeric(df_all["y"], errors="coerce")
        if cfg.get("seasons", {}).get("active"):
            df_all = df_all[df_all["semester"].isin(cfg["seasons"]["active"])]
        else:
            df_all = df_all[df_all["year"].between(*cfg["presence_data"]["year_filter"])]
        if cfg["presence_data"].get("source_filter"):
            df_all = df_all[df_all["dataset"].isin(cfg["presence_data"]["source_filter"])]
        df_all = df_all.dropna(subset=["x", "y"]).reset_index(drop=True)
        from scripts.utils import load_mask, filter_points_by_mask
        mask = load_mask(cfg["boundary_mask"])
        df_all = filter_points_by_mask(df_all, mask)
        gdf_all = gpd.GeoDataFrame(df_all, geometry=gpd.points_from_xy(df_all["x"], df_all["y"]), crs="EPSG:2154")
        plot_points(gdf_all, Path(cfg["experiment"]["vis_dir"]) / "step1_all_points.png", "All records")
    else:
        print("[1/8] Preprocessing skipped")
        log.info("Preprocessing skipped")
        stack_path = Path(cfg["outputs"]["root"]) / "stack" / f"{cfg['experiment']['name']}_predictor_stack.zarr"

    # 2. Variable selection
    vars_used = []
    if cfg["steps"].get("variable_selection", True):
        print("[2/8] Variable selection")
        log.info("Step 2 - variable selection")
        vars_used = run_selection(cfg, stack_path)
        log.debug("Variables selected: %s", ", ".join(vars_used))
        plot_variables(stack_path, vars_used, Path(cfg["experiment"]["vis_dir"]) / "step2_selected_variables.png")
    else:
        print("[2/8] Variable selection skipped")
        log.info("Variable selection skipped")
        vars_used = _load_selected_vars(cfg, stack_path)

    # 3. Presence thinning
    pres_gdf = None
    if cfg["steps"].get("thinning", True):
        print("[3/8] Thinning")
        log.info("Step 3 - thinning")

        # Mahalanobis thinning requires environmental values for each
        # presence record. Only compute them if any enabled method uses it.
        need_env = any(
            m.get("enable", True) and m.get("name") == "mahalanobis"
            for m in cfg["presence_data"]["thinning"].get("methods", [])
        )
        env_df = None
        if need_env:
            import pandas as pd
            import geopandas as gpd

            df = pd.read_csv(
                cfg["presence_data"]["file"], encoding="utf-8-sig"
            )
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
            # Remove records with missing coordinates to avoid NaNs during
            # thinning and environmental value extraction
            df = df.dropna(subset=["x", "y"]).reset_index(drop=True)
            from scripts.utils import load_mask, filter_points_by_mask
            mask = load_mask(cfg["boundary_mask"])
            df = filter_points_by_mask(df, mask)
            gdf_temp = gpd.GeoDataFrame(
                df, geometry=gpd.points_from_xy(df["x"], df["y"]), crs="EPSG:2154"
            )
            env_vals = extract_from_stack(stack_path, gdf_temp, vars_used)
            env_df = pd.DataFrame(env_vals, columns=vars_used)

        from scripts.utils import deduplicate_cells
        pres_gdf = run_thinning(cfg, env_df=env_df)
        pres_gdf = deduplicate_cells(pres_gdf)
        plot_points(pres_gdf, Path(cfg["experiment"]["vis_dir"]) / "step3_thinned_points.png", "Thinned presences")
    else:
        print("[3/8] Thinning skipped")
        log.info("Thinning skipped")
        pres_gdf = _load_thinned_presences(cfg)

    # 4. Bias grid
    bias_tif = None
    if cfg["steps"].get("bias_grid", False):
        print("[4/8] Bias grid")
        log.info("Step 4 - bias grid")
        bias_tif = create_bias_grid(pres_gdf, cfg)
        if bias_tif and bias_tif.exists():
            plot_raster(bias_tif, Path(cfg["experiment"]["vis_dir"]) / "step4_bias_grid.png", "Bias grid")
    else:
        print("[4/8] Bias grid skipped")
        log.info("Bias grid skipped")
        bias_tif = Path(cfg["bias_grid"]["output_file"])

    # 5. Background sampling
    bg_gdf = None
    if cfg["steps"].get("background", True):
        print("[5/8] Background sampling")
        log.info("Step 5 - background sampling")
        from scripts.utils import deduplicate_cells
        bg_gdf = sample_background(cfg, bias_tif if bias_tif.exists() else None)
        bg_gdf = deduplicate_cells(bg_gdf)
        plot_points(
            bg_gdf,
            Path(cfg["experiment"]["vis_dir"]) / "step5_background_points.png",
            "Background samples",
            markersize=0.5,
        )
    else:
        print("[5/8] Background sampling skipped")
        log.info("Background sampling skipped")
        bg_gdf = gpd.read_file(cfg["background"]["output_file"])
        bg_gdf = ensure_gdf_crs(bg_gdf, context="background file")
        from scripts.utils import load_mask, filter_points_by_mask, deduplicate_cells
        mask = load_mask(cfg["boundary_mask"])
        df_bg = filter_points_by_mask(bg_gdf.drop(columns="geometry"), mask)
        bg_gdf = gpd.GeoDataFrame(df_bg, geometry=gpd.points_from_xy(df_bg["x"], df_bg["y"]), crs="EPSG:2154")
        bg_gdf = deduplicate_cells(bg_gdf)
        plot_points(
            bg_gdf,
            Path(cfg["experiment"]["vis_dir"]) / "step5_background_points.png",
            "Background samples",
            markersize=0.5,
        )

    # Optional train/test split when not using cross-validation
    test_prop = cfg.get("train_test_split", {}).get("proportion", 0)
    do_split = test_prop > 0 and not cfg["evaluation"].get("cross_validation", {}).get("enable", False)
    if do_split:
        stratify_col = cfg.get("train_test_split", {}).get("stratify_by")
        stratify_vals = None
        if stratify_col and stratify_col in pres_gdf.columns:
            counts = pres_gdf[stratify_col].value_counts(dropna=False)
            if counts.min() < 2:
                log.warning(
                    "Cannot stratify train/test split by '%s' because the smallest class has only %d sample(s)",
                    stratify_col,
                    counts.min(),
                )
            else:
                stratify_vals = pres_gdf[stratify_col]
        pres_train, pres_test = train_test_split(
            pres_gdf,
            test_size=test_prop,
            random_state=cfg["experiment"]["random_seed"],
            stratify=stratify_vals,
        )
        bg_train, bg_test = train_test_split(
            bg_gdf,
            test_size=test_prop,
            random_state=cfg["experiment"]["random_seed"],
        )
    else:
        pres_train, pres_test = pres_gdf, pres_gdf.iloc[0:0].copy()
        bg_train, bg_test = bg_gdf, bg_gdf.iloc[0:0].copy()

    engines = [str(e).lower() for e in cfg.get("model", {}).get("engines", ["maxent", "gbm"])]
    engine_results: list[tuple[str, Path | None]] = []

    for engine in engines:
        log.info("Starting engine %s", engine)
        engine_dir = exp_dir / "engines" / engine
        engine_outputs = engine_dir / "outputs"
        engine_vis = engine_dir / "visualizations"
        engine_fi = engine_dir / "feature_importance"
        engine_stats = engine_dir / "stats"
        for path in [engine_outputs, engine_vis, engine_fi, engine_stats]:
            path.mkdir(parents=True, exist_ok=True)
        engine_logs = engine_outputs / "logs"
        engine_logs.mkdir(parents=True, exist_ok=True)
        os.environ["LOGS_DIR"] = str(engine_logs)

        cfg_engine = deepcopy(cfg)
        cfg_engine["experiment"]["name"] = f"{combined_name}_{engine}"
        cfg_engine["experiment"]["vis_dir"] = str(engine_vis)
        cfg_engine["experiment"]["fi_dir"] = str(engine_fi)
        cfg_engine["experiment"]["stats_dir"] = str(engine_stats)
        _rebase_outputs(cfg_engine, shared_outputs, engine_outputs)
        cfg_engine.setdefault("outputs", {})["root"] = str(engine_outputs)

        vars_engine = list(vars_used)
        model = None
        map_tif: Path | None = None

        step_prefix = f"[{engine.upper()}]"
        if cfg_engine["steps"].get("training", True):
            print(f"[6/8] Training {engine}")
            log.info("Step 6 - training (%s)", engine)
            if engine == "maxent":
                model, map_tif, vars_engine = train_maxent_model(
                    cfg_engine, stack_path, vars_engine, pres_train, bg_train
                )
            elif engine == "gbm":
                from scripts.gbm_training import train_model as train_gbm_model

                model, map_tif, vars_engine = train_gbm_model(
                    cfg_engine, stack_path, vars_engine, pres_train, bg_train
                )
            else:
                raise ValueError(f"Unknown modelling engine '{engine}'")

            if map_tif and map_tif.exists():
                plot_raster(
                    map_tif,
                    Path(cfg_engine["experiment"]["vis_dir"]) / f"step6_{engine}_map.png",
                    "Predicted suitability",
                )
            safe_name = sanitize_name(cfg_engine["experiment"]["name"])
            auc_src = Path(cfg_engine["outputs"]["figures_dir"]) / f"{safe_name}_auc.png"
            if auc_src.exists():
                shutil.copy(
                    auc_src,
                    Path(cfg_engine["experiment"]["vis_dir"]) / f"step6_{engine}_auc_curve.png",
                )

            try:
                pres_vals = extract_from_stack(stack_path, pres_train, vars_engine)
                bg_vals = extract_from_stack(stack_path, bg_train, vars_engine)
                df_pres = pd.DataFrame(pres_vals, columns=vars_engine)
                df_bg = pd.DataFrame(bg_vals, columns=vars_engine)
                plot_variable_histograms(
                    df_pres,
                    df_bg,
                    vars_engine,
                    Path(cfg_engine["experiment"]["vis_dir"]) / f"step6_{engine}_variable_hist.png",
                )
                stats_dir = Path(cfg_engine["experiment"]["stats_dir"])
                stats_dir.mkdir(parents=True, exist_ok=True)
                summary = [
                    {
                        "variable": var,
                        "mean_presence": float(df_pres[var].mean()),
                        "mean_background": float(df_bg[var].mean()),
                        "std_presence": float(df_pres[var].std()),
                        "std_background": float(df_bg[var].std()),
                    }
                    for var in vars_engine
                ]
                pd.DataFrame(summary).to_csv(
                    stats_dir / f"{engine}_variable_summary.csv", index=False
                )
                export_response_curves(
                    model,
                    pd.concat([df_pres, df_bg], ignore_index=True),
                    vars_engine,
                    Path(cfg_engine["experiment"]["vis_dir"]) / f"response_curves_{engine}",
                )
            except Exception as exc:
                log.warning("Failed to generate additional stats for %s: %s", engine, exc)
        else:
            print(f"[6/8] Training {engine} skipped")
            log.info("Training skipped for %s", engine)
            safe_name = sanitize_name(cfg_engine["experiment"]["name"])
            map_paths = [
                Path(cfg_engine["outputs"]["maps_dir"]) / f"{safe_name}_cloglog.tif",
                Path(cfg_engine["outputs"]["maps_dir"]) / f"{safe_name}_gbm_probability.tif",
            ]
            map_tif = next((p for p in map_paths if p.exists()), None)
            if map_tif and map_tif.exists():
                plot_raster(
                    map_tif,
                    Path(cfg_engine["experiment"]["vis_dir"]) / f"step6_{engine}_map.png",
                    "Predicted suitability",
                )
            auc_src = Path(cfg_engine["outputs"]["figures_dir"]) / (
                f"{sanitize_name(cfg_engine['experiment']['name'])}_auc.png"
            )
            if auc_src.exists():
                shutil.copy(
                    auc_src,
                    Path(cfg_engine["experiment"]["vis_dir"]) / f"step6_{engine}_auc_curve.png",
                )

        if cfg_engine["steps"].get("evaluation", True) and model is not None:
            print(f"[7/8] Evaluation {engine}")
            log.info("Step 7 - evaluation (%s)", engine)
            if cfg_engine["evaluation"].get("cross_validation", {}).get("enable", False):
                run_cv(
                    cfg_engine,
                    stack_path=stack_path,
                    vars_used=vars_engine,
                    pres_gdf=pres_gdf,
                    bg_gdf=bg_gdf,
                )
                fig_dir = Path(cfg_engine["outputs"]["figures_dir"])
                safe_name = sanitize_name(cfg_engine["experiment"]["name"])
                for src_name in [
                    "calibration_plot.png",
                    f"{safe_name}_pr.png",
                    f"{safe_name}_threshold_auc.png",
                    f"{safe_name}_threshold_cbi.png",
                ]:
                    src = fig_dir / src_name
                    if src.exists():
                        shutil.copy(
                            src,
                            Path(cfg_engine["experiment"]["vis_dir"]) / f"step7_{engine}_{src.name}",
                        )
            else:
                env_pres = extract_from_stack(stack_path, pres_train, vars_engine)
                env_bg = extract_from_stack(stack_path, bg_train, vars_engine)
                mask_pres = ~np.isnan(env_pres).any(axis=1)
                mask_bg = ~np.isnan(env_bg).any(axis=1)
                pres_train_eval = pres_train.loc[mask_pres].reset_index(drop=True)
                bg_train_eval = bg_train.loc[mask_bg].reset_index(drop=True)
                env_pres = env_pres[mask_pres]
                env_bg = env_bg[mask_bg]
                pres_scores_train = model.predict(env_pres)
                bg_scores_train = model.predict(env_bg)
                y_true_train = np.concatenate(
                    [np.ones_like(pres_scores_train), np.zeros_like(bg_scores_train)]
                )
                y_score_train = np.concatenate([pres_scores_train, bg_scores_train])

                train_gdf = gpd.GeoDataFrame(
                    pd.concat([pres_train_eval, bg_train_eval], ignore_index=True),
                    crs=pres_train_eval.crs,
                )

                env_pres_test = extract_from_stack(stack_path, pres_test, vars_engine)
                env_bg_test = extract_from_stack(stack_path, bg_test, vars_engine)
                mask_pres_test = ~np.isnan(env_pres_test).any(axis=1)
                mask_bg_test = ~np.isnan(env_bg_test).any(axis=1)
                pres_test_eval = pres_test.loc[mask_pres_test].reset_index(drop=True)
                bg_test_eval = bg_test.loc[mask_bg_test].reset_index(drop=True)
                env_pres_test = env_pres_test[mask_pres_test]
                env_bg_test = env_bg_test[mask_bg_test]
                pres_scores_test = (
                    model.predict(env_pres_test) if len(pres_test_eval) > 0 else np.array([])
                )
                bg_scores_test = (
                    model.predict(env_bg_test) if len(bg_test_eval) > 0 else np.array([])
                )

                y_true_test = (
                    np.concatenate(
                        [np.ones_like(pres_scores_test), np.zeros_like(bg_scores_test)]
                    )
                    if len(pres_scores_test) + len(bg_scores_test) > 0
                    else None
                )
                y_score_test = (
                    np.concatenate([pres_scores_test, bg_scores_test])
                    if len(pres_scores_test) + len(bg_scores_test) > 0
                    else None
                )
                test_gdf = (
                    gpd.GeoDataFrame(
                        pd.concat([pres_test_eval, bg_test_eval], ignore_index=True),
                        crs=pres_test_eval.crs,
                    )
                    if len(pres_scores_test) + len(bg_scores_test) > 0
                    else None
                )
                n_params = count_model_params(model)

                evaluate(
                    cfg_engine["experiment"]["name"],
                    y_true_train,
                    y_score_train,
                    cfg_engine,
                    y_true_test,
                    y_score_test,
                    n_params=n_params,
                    gdf_train=train_gdf,
                    gdf_test=test_gdf,
                )
                fig_dir = Path(cfg_engine["outputs"]["figures_dir"])
                safe_name = sanitize_name(cfg_engine["experiment"]["name"])
                for src_name, dst_name in [
                    ("calibration_plot.png", f"step7_{engine}_calibration_plot.png"),
                    (f"{safe_name}_pr.png", f"step7_{engine}_pr_curve.png"),
                    (f"{safe_name}_threshold_auc.png", f"step7_{engine}_threshold_auc.png"),
                    (f"{safe_name}_threshold_cbi.png", f"step7_{engine}_threshold_cbi.png"),
                ]:
                    src = fig_dir / src_name
                    if src.exists():
                        shutil.copy(src, Path(cfg_engine["experiment"]["vis_dir"]) / dst_name)
        else:
            print(f"[7/8] Evaluation {engine} skipped")
            log.info("Evaluation skipped for %s", engine)

        if cfg_engine["steps"].get("variable_importance", False) and model is not None:
            print(f"[8/8] Variable importance {engine}")
            log.info("Step 8 - variable importance (%s)", engine)
            from scripts.variable_importance import run_importance

            run_importance(
                cfg_engine,
                stack_path=stack_path,
                vars_used=vars_engine,
                model=model,
                pres_gdf=pres_train,
                bg_gdf=bg_train,
                out_dir=Path(cfg_engine["experiment"]["fi_dir"]),
            )
        else:
            print(f"[8/8] Variable importance {engine} skipped")
            log.info("Variable importance skipped for %s", engine)

        engine_results.append((engine, map_tif))

    for engine_name, map_path in engine_results:
        print(f"Pipeline completed for {engine_name} - output map: {map_path}")
        log.info("Pipeline completed for %s - output map: %s", engine_name, map_path)

if __name__ == "__main__":
    main()
