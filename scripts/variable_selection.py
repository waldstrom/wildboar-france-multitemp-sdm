# =============================================================================
# Project Title: Wild Boar Distribution Model for France
# Path: scripts/variable_selection.py
# Purpose: Select environmental variables using correlation clustering and VIF.
# Process Step: Implements pruning strategy to remove redundant predictors.
# Created by Adrian Meyer, Institute Geomatics, FHNW (adrian.meyer@fhnw.ch)
# Created: July 2025
# Version: v.0.1.0
# =============================================================================
"""
Correlation-tree pruning and VIF < 5 following the Bosch et al. procedure.
The correlation cut height defaults to 0.7, following the recommendations of
Dormann et al. (2013), to avoid overly aggressive variable removal.
"""
from __future__ import annotations
import logging
from pathlib import Path
import re
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from statsmodels.stats.outliers_influence import variance_inflation_factor
import xarray as xr
from tqdm import tqdm
from scripts.utils import ensure_dir, get_processed_dir
import yaml
import shutil

log = logging.getLogger(__name__)

# -----------------------------------------------------------------------------  
def pick_one_from_cluster(cluster_vars: list[str], env_df: pd.DataFrame) -> str:
    # Pick variable with lowest mean correlation to others in cluster
    sub = env_df[cluster_vars]
    corr = sub.corr().abs()
    score = corr.mean()
    return score.idxmin()

# -----------------------------------------------------------------------------  
def vif_filter(df: pd.DataFrame, threshold: float = 5.0) -> list[str]:
    variables = list(df.columns)
    if len(variables) <= 1:
        # Nothing to filter when one or zero variables remain
        return variables

    while True:
        # Statsmodels' VIF computation fails when only a single predictor is
        # provided. Guard against this by exiting the loop if that happens
        if len(variables) <= 1:
            break
        vif = [variance_inflation_factor(df[variables].values, i) for i in range(len(variables))]
        max_vif = max(vif)
        if max_vif > threshold:
            drop_var = variables[vif.index(max_vif)]
            variables.remove(drop_var)
            log.debug("VIF %.2f - dropping %s", max_vif, drop_var)
        else:
            break
    return variables

# -----------------------------------------------------------------------------

def _select_group(df: pd.DataFrame, cfg: dict, group: str | int | None = None) -> list[str]:
    """Run correlation clustering and VIF pruning on a subset of variables.

    ``group`` allows overriding the VIF threshold using
    ``cfg['vif_by_group']`` if provided.
    """
    if df.shape[1] <= 1:
        return list(df.columns)

    # Remove constant predictors which would yield NaN correlations and
    # subsequently break the linkage computation. Any columns with a single
    # unique value are ignored with a warning so that at least two variables
    # remain for clustering.
    nunique = df.nunique(dropna=False)
    constant_cols = nunique.index[nunique <= 1].tolist()
    if constant_cols:
        log.warning("Ignoring %d constant variables", len(constant_cols))
        df = df.drop(columns=constant_cols)
    if df.shape[1] <= 1:
        return list(df.columns)

    # Fill potential NaN values created during correlation calculation in case
    # very similar predictors cause undefined coefficients.
    corr = df.corr().abs().fillna(0)
    Z = linkage(corr, method=cfg["linkage"])
    clusters = fcluster(Z, t=cfg["correlation_cutoff"], criterion="distance")
    selected = []
    for cl in tqdm(np.unique(clusters), desc="clusters", leave=False):
        members = df.columns[clusters == cl].tolist()
        if len(members) == 1:
            selected.append(members[0])
        else:
            selected.append(pick_one_from_cluster(members, df))
    threshold = cfg.get("vif_threshold", 5.0)
    if group is not None:
        group_map = cfg.get("vif_by_group", {})
        threshold = group_map.get(str(group), threshold)
    return vif_filter(df[selected], threshold)

# -----------------------------------------------------------------------------  
def run_selection(cfg: dict, stack_path: Path) -> list[str]:
    print("    -> Running variable selection")
    proc_dir = ensure_dir(get_processed_dir())
    proc_file = proc_dir / "selected_predictors.txt"

    sel_file = Path(cfg["predictors"]["variable_selection"]["save_selected_list"])
    ensure_dir(sel_file.parent)

    if proc_file.exists():
        log.info("Using cached selected predictors from %s", proc_file)
        if not sel_file.exists():
            shutil.copy2(proc_file, sel_file)
        return proc_file.read_text().splitlines()
    # Avoid fallback to non-consolidated metadata when opening the stack
    try:
        ds = xr.open_zarr(stack_path, consolidated=False)
    except Exception as exc:
        log.warning("Failed reading stack at %s: %s - rebuilding", stack_path, exc)
        proc_path = get_processed_dir() / "stack" / "predictor_stack.zarr"
        shutil.rmtree(stack_path, ignore_errors=True)
        shutil.rmtree(proc_path, ignore_errors=True)
        from scripts.preprocessing import build_stack
        stack_path = build_stack(cfg)
        ds = xr.open_zarr(stack_path, consolidated=False)
    sample_frac = cfg["predictors"]["variable_selection"]["sample_fraction"]
    # Draw a pseudorandom subset of grid cells; no occurrence points are needed
    df = (
        ds.to_dataframe()
        .drop(columns=[c for c in ds.data_vars if c == "boundary_mask"])
        .sample(
            frac=sample_frac,
            random_state=cfg["experiment"]["random_seed"],
        )
        .dropna()
    )

    ref_year = str(cfg["predictors"]["variable_selection"].get("reference_year", ""))
    only_ref = cfg["predictors"]["variable_selection"].get("lc_reference_year_only", True)
    if ref_year and only_ref:
        year_pat = re.compile(r"(19|20)\d{2}")
        keep_cols: list[str] = []
        for col in df.columns:
            m = year_pat.search(col)
            if m:
                if m.group() == ref_year or "_class" in col:
                    keep_cols.append(col)
            else:
                keep_cols.append(col)
        df = df[keep_cols]
    elif not only_ref:
        pat = re.compile(r"(\d{4})_class(\d+_[A-Z]+)")
        class_groups: dict[str, list[str]] = {}
        for col in list(df.columns):
            m = pat.match(col)
            if m:
                class_groups.setdefault(m.group(2), []).append(col)
        for cl, cols_cl in class_groups.items():
            df[f"LC_mean_{cl}"] = df[cols_cl].mean(axis=1)
        df = df.drop(columns=[c for cols_cl in class_groups.values() for c in cols_cl])

    # Apply drop rules from feature configuration if available
    fc_path = cfg["predictors"]["variable_selection"].get("feature_config", "featureconfig.yaml")
    if Path(fc_path).exists():
        try:
            feature_cfg = yaml.safe_load(Path(fc_path).read_text())
            drop_vars = [k for k, v in feature_cfg.get("variables", {}).items() if str(v).lower() == "drop"]
            drop_in_df = [v for v in drop_vars if v in df.columns]
            if drop_in_df:
                log.info("Dropping %d variables defined as 'drop' in %s", len(drop_in_df), fc_path)
                df = df.drop(columns=drop_in_df)
        except Exception as e:
            log.warning("Failed loading feature config %s: %s", fc_path, e)

    cols = list(df.columns)
    lc_vars = [c for c in cols if re.search(r"\d{4}_class\d+_", c)]
    drop_lc_urban = [v for v in lc_vars if "_class11_URBAN" in v]
    if drop_lc_urban:
        log.info("Excluding %d LC Urban variables", len(drop_lc_urban))
        df = df.drop(columns=drop_lc_urban)
        lc_vars = [v for v in lc_vars if v not in drop_lc_urban]
        cols = [c for c in cols if c not in drop_lc_urban]
    bioclim_vars = [c for c in cols if c.lower().startswith("bio")]
    sat_vars = {
        "04": [c for c in cols if re.search(fr"{ref_year}_04_", c)],
        "07": [c for c in cols if re.search(fr"{ref_year}_07_", c)],
        "10": [c for c in cols if re.search(fr"{ref_year}_10_", c)],
    }
    aux_vars = [c for c in cols if c.startswith("DEM_") or re.match(r"LC\d+_", c) or any(k in c.lower() for k in ["small_woody", "zones_humides", "humanfootprint"])]
    no_filter = [c for c in cols if "fragment" in c.lower() or "hunting" in c.lower()]
    used = set(lc_vars + bioclim_vars + aux_vars + no_filter + sat_vars["04"] + sat_vars["07"] + sat_vars["10"])
    other_vars = [c for c in cols if c not in used]

    # Drop variables highly correlated with any land cover variable
    lc_cutoff = cfg["predictors"]["variable_selection"].get("lc_overlap_cutoff", 0.9)
    if lc_vars:
        corr = df[lc_vars + other_vars].corr().abs()
        drop_others = []
        for var in other_vars:
            if corr.loc[var, lc_vars].max() >= lc_cutoff:
                drop_others.append(var)
        other_vars = [v for v in other_vars if v not in drop_others]

    selected = []
    selected.extend(lc_vars)
    selected.extend(no_filter)
    selected.extend(
        _select_group(
            df[bioclim_vars], cfg["predictors"]["variable_selection"], group=1
        )
    )
    for mth in ["04", "07", "10"]:
        selected.extend(
            _select_group(
                df[sat_vars[mth]],
                cfg["predictors"]["variable_selection"],
                group=2,
            )
        )
    selected.extend(
        _select_group(
            df[aux_vars + other_vars],
            cfg["predictors"]["variable_selection"],
            group=3,
        )
    )

    max_vars = cfg["predictors"]["variable_selection"].get("max_variables")
    if max_vars and len(selected) > max_vars:
        log.info("Limiting predictor set to %d variables", max_vars)
        selected = selected[: max_vars]

    sel_file.write_text("\n".join(selected))
    if not proc_file.exists():
        shutil.copy2(sel_file, proc_file)
    # Re-compute correlation matrix and VIFs for the final set to document
    # collinearity of predictors after filtering.
    final_df = df[selected].dropna()
    corr = final_df.corr().abs()
    max_corr = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool)).max().max()
    vif_vals = [variance_inflation_factor(final_df.values, i) for i in range(final_df.shape[1])]
    for name, vif in zip(final_df.columns, vif_vals):
        log.info("Final VIF %s %.2f", name, vif)
    log.info("Max pairwise correlation after selection %.2f", max_corr)
    log.info("Variable selection complete - %d variables retained", len(selected))
    print(f"    -> Selected {len(selected)} variables")
    return selected
