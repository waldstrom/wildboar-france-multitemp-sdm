# =============================================================================
# Project Title: Wild Boar Distribution Model for France
# Path: scripts/cross_validation.py
# Purpose: Provide spatial K-fold and other cross-validation utilities.
# Process Step: Splits training data into folds and evaluates model performance.
# Created by Adrian Meyer, Institute Geomatics, FHNW (adrian.meyer@fhnw.ch)
# Created: July 2025
# Version: v.0.1.0
# =============================================================================
"""
Spatial K-fold or other CV methods - callable via separate entry-point.
"""
from __future__ import annotations
import logging
import numpy as np
from sklearn.model_selection import KFold
from sklearn.cluster import KMeans
from elapid import GeographicKFold
from scripts.maxent_training import train_model
from scripts.utils import extract_from_stack, ensure_dir
from scripts.evaluation import _compute_metrics
import xarray as xr
import copy
from tqdm import tqdm
from pathlib import Path
import pandas as pd

log = logging.getLogger(__name__)

# -----------------------------------------------------------------------------  
def spatial_block_split(gdf, k: int, block_km: int, random_state: int | None = None):
    # Simple lat/lon block splitter in EPSG:2154 metres
    gdf = gdf.copy()
    gdf["bx"] = (gdf.geometry.x // (block_km * 1000)).astype(int)
    gdf["by"] = (gdf.geometry.y // (block_km * 1000)).astype(int)
    blocks = gdf.groupby(["bx", "by"]).ngroup().to_numpy()
    unique_blocks = np.unique(blocks)
    if unique_blocks.size < k:
        raise ValueError(
            f"Requested {k} spatial folds but only {unique_blocks.size} spatial blocks were found."
        )

    rng = np.random.default_rng(random_state)
    rng.shuffle(unique_blocks)
    for fold_blocks in np.array_split(unique_blocks, k):
        test_mask = np.isin(blocks, fold_blocks)
        test_idx = gdf.index[test_mask]
        train_idx = gdf.index[~test_mask]
        yield train_idx, test_idx

# -----------------------------------------------------------------------------

def environmental_block_split(gdf, stack_path: Path, vars_used: list[str], k: int, random_state: int = 0):
    """Cluster records by environmental similarity and yield fold indices."""
    X = extract_from_stack(stack_path, gdf, vars_used)
    mask = ~np.any(np.isnan(X), axis=1)
    if not np.any(mask):
        raise ValueError("No valid records for environmental blocking")
    kmeans = KMeans(n_clusters=k, random_state=random_state)
    clusters = kmeans.fit_predict(X[mask])
    gdf = gdf.copy()
    gdf.loc[mask, "cluster"] = clusters
    gdf.loc[~mask, "cluster"] = -1
    for fold in range(k):
        test_idx = gdf.index[gdf.cluster == fold]
        train_idx = gdf.index[gdf.cluster != fold]
        yield train_idx, test_idx

# -----------------------------------------------------------------------------  
def run_cv(cfg: dict, **kwargs):
    print("    -> Starting cross-validation")
    method_cfg = cfg["evaluation"]["cross_validation"]["method"]
    if method_cfg == "compare":
        methods = ["geographic_kfold", "spatial_block", "env_block"]
    elif isinstance(method_cfg, (list, tuple)):
        methods = list(method_cfg)
    else:
        methods = [method_cfg]
    k = cfg["evaluation"]["cross_validation"]["params"]["k_folds"]

    pres_gdf = kwargs["pres_gdf"].reset_index(drop=True)
    bg_gdf = kwargs["bg_gdf"]
    stack_path = kwargs["stack_path"]
    vars_used = kwargs["vars_used"]
    if not vars_used:
        with xr.open_zarr(stack_path) as ds:
            vars_used = list(ds.data_vars)
        log.info("No predictors supplied; using all %d variables from stack", len(vars_used))

    stats_dir = Path(cfg["experiment"].get("stats_dir", "stats"))
    ensure_dir(stats_dir)

    summary = []
    for method in methods:
        if method == "random_kfold":
            kf = KFold(k, shuffle=True, random_state=cfg["experiment"]["random_seed"])
            splits = kf.split(pres_gdf)
        elif method == "spatial_block":
            splits = spatial_block_split(
                pres_gdf,
                k,
                cfg["evaluation"]["cross_validation"]["params"]["block_size_km"],
                random_state=cfg["experiment"].get("random_seed"),
            )
        elif method == "geographic_kfold":
            gkf = GeographicKFold(n_splits=k)
            splits = gkf.split(pres_gdf)
        elif method == "env_block":
            splits = environmental_block_split(
                pres_gdf,
                stack_path,
                vars_used,
                k,
                cfg["experiment"]["random_seed"],
            )
        else:
            raise NotImplementedError(method)

        aucs = []
        cbis = []
        omissions = []
        for i, (train_idx, test_idx) in enumerate(
            tqdm(list(splits), total=k, desc=f"{method} folds"), start=1
        ):
            log.info("CV %s fold %d/%d", method, i, k)
            print(f"    -> {method} fold {i}/{k}")
            pres_train = pres_gdf.iloc[train_idx]
            pres_test = pres_gdf.iloc[test_idx]

            fold_dir = Path(cfg["experiment"].get("stats_dir", "stats")).parent / "cv" / method / f"fold_{i}"
            ensure_dir(fold_dir)
            fold_cfg = copy.deepcopy(cfg)
            fold_cfg["outputs"]["root"] = str(fold_dir / "outputs")
            fold_cfg["outputs"]["logs_dir"] = str(Path(fold_cfg["outputs"]["root"]) / "logs")
            fold_cfg["outputs"]["model_dir"] = str(Path(fold_cfg["outputs"]["root"]) / "models")
            fold_cfg["outputs"]["maps_dir"] = str(Path(fold_cfg["outputs"]["root"]) / "maps")
            fold_cfg["outputs"]["figures_dir"] = str(Path(fold_cfg["outputs"]["root"]) / "figures")
            for p in [
                fold_cfg["outputs"]["logs_dir"],
                fold_cfg["outputs"]["model_dir"],
                fold_cfg["outputs"]["maps_dir"],
                fold_cfg["outputs"]["figures_dir"],
            ]:
                ensure_dir(p)

            model, _, used = train_model(fold_cfg, stack_path, vars_used, pres_train, bg_gdf)
            vars_used = used

            y_pres = model.predict(extract_from_stack(stack_path, pres_test, vars_used))
            y_bg = model.predict(extract_from_stack(stack_path, bg_gdf, vars_used))
            y_true = np.concatenate([np.ones_like(y_pres), np.zeros_like(y_bg)])
            y_score = np.concatenate([y_pres, y_bg])
            mask = ~np.isnan(y_score)
            if not np.all(mask):
                dropped = int(np.sum(~mask))
                log.warning(
                    "Fold %d contains %d NaN predictions - excluding from metrics", i, dropped
                )
            y_true = y_true[mask]
            y_score = y_score[mask]
            if len(np.unique(y_true)) < 2:
                log.warning("Fold %d lacks positive or negative samples after filtering", i)
                metrics = {"auc": float("nan"), "cbi": float("nan"), "omission": float("nan")}
            else:
                thr = cfg.get("evaluation", {}).get("classification_threshold", 0.5)
                metrics = _compute_metrics(y_true, y_score, threshold=thr)
            aucs.append(metrics["auc"])
            cbis.append(metrics["cbi"])
            omissions.append(metrics.get("omission", 1 - metrics.get("recall", 0)))
            log.info(
                "Fold %d AUC %.3f CBI %.3f Omission %.3f",
                i,
                metrics["auc"],
                metrics["cbi"],
                metrics.get("omission", 1 - metrics.get("recall", 0)),
            )
            print(
                f"       AUC {metrics['auc']:.3f} CBI {metrics['cbi']:.3f} Omission {metrics.get('omission', 1 - metrics.get('recall', 0)):.3f}"
            )

        mean_auc = float(np.nanmean(aucs)) if aucs else float("nan")
        mean_cbi = float(np.nanmean(cbis)) if cbis else float("nan")
        mean_omission = float(np.nanmean(omissions)) if omissions else float("nan")
        print(
            f"    -> {method} mean CBI {mean_cbi:.3f} mean AUC {mean_auc:.3f} mean Omission {mean_omission:.3f}"
        )
        log.info(
            "Cross-validation %s mean CBI %.3f mean AUC %.3f mean Omission %.3f",
            method,
            mean_cbi,
            mean_auc,
            mean_omission,
        )
        df = pd.DataFrame(
            {
                "fold": list(range(1, len(aucs) + 1)),
                "auc": aucs,
                "cbi": cbis,
                "omission": omissions,
            }
        )
        df.loc[len(df)] = ["mean", mean_auc, mean_cbi, mean_omission]
        df.to_csv(stats_dir / f"cross_validation_{method}.csv", index=False)
        summary.append(
            {
                "method": method,
                "mean_cbi": mean_cbi,
                "mean_auc": mean_auc,
                "mean_omission": mean_omission,
            }
        )

    pd.DataFrame(summary).to_csv(stats_dir / "cross_validation_summary.csv", index=False)
