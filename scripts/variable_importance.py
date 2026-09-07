# =============================================================================
# Project Title: Wild Boar Distribution Model for France
# Path: scripts/variable_importance.py
# Purpose: Compute variable importance metrics for trained models.
# Process Step: Permutation, gain and jackknife importance utilities.
# Created by Adrian Meyer, Institute Geomatics, FHNW (adrian.meyer@fhnw.ch)
# Created: July 2025
# Version: v.0.1.0
# =============================================================================
"""
Utilities to quantify variable importance.

Three methods are provided:

``permutation`` – shuffle each predictor and measure the drop in AUC.
``gain`` – extract model reported feature gains, if available.
``jackknife`` – retrain with each variable individually and while leaving it
out to approximate the traditional MaxEnt jackknife test.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Sequence, Any
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import yaml
from joblib import Parallel, delayed

from scripts.utils import extract_from_stack, ensure_dir
from scripts.maxent_training import train_model
from elapid import MaxentModel, distance_weights

log = logging.getLogger(__name__)


def _auc_score(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Minimal AUC implementation to avoid external dependencies."""
    pos = y_score[y_true == 1]
    neg = y_score[y_true == 0]
    n_pos = len(pos)
    n_neg = len(neg)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    # Mann-Whitney U statistic
    U = np.sum(pos[:, None] > neg) + 0.5 * np.sum(pos[:, None] == neg)
    return float(U / (n_pos * n_neg))


def permutation_importance_array(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    *,
    n_repeats: int = 5,
    random_state: int = 0,
    n_jobs: int = 1,
) -> np.ndarray:
    """Return mean and std importance for each column in ``X``."""
    base_auc = _auc_score(y, model.predict(X))

    def _score_col(i: int) -> tuple[float, float]:
        """Permute column ``i`` and measure the AUC drop."""
        rng = np.random.default_rng(random_state + i)
        scores: list[float] = []
        for _ in range(n_repeats):
            X_perm = X.copy()
            rng.shuffle(X_perm[:, i])
            scores.append(base_auc - _auc_score(y, model.predict(X_perm)))
        return float(np.mean(scores)), float(np.std(scores))

    importances = Parallel(n_jobs=n_jobs)(delayed(_score_col)(i) for i in range(X.shape[1]))
    return np.asarray(importances)


def permutation_importance(
    cfg: dict,
    model: Any,
    stack_path: Path,
    vars_used: Sequence[str],
    pres_gdf,
    bg_gdf,
    out_dir: Path,
) -> pd.DataFrame:
    """Compute permutation importance and save results."""

    log.info("Computing permutation importance")
    n_rep = cfg["variable_importance"]["permutation"].get("n_repeats", 5)
    X_pres = extract_from_stack(stack_path, pres_gdf, list(vars_used))
    X_bg = extract_from_stack(stack_path, bg_gdf, list(vars_used))
    X = np.vstack([X_pres, X_bg])
    y = np.hstack([np.ones(len(X_pres)), np.zeros(len(X_bg))])

    mask = ~np.isnan(X).any(axis=1)
    X = X[mask]
    y = y[mask]

    threads = cfg.get("maxent", {}).get("threads", 1)
    if threads == -1:
        threads = os.cpu_count() or 1
    imp = permutation_importance_array(
        model,
        X,
        y,
        n_repeats=n_rep,
        random_state=cfg["experiment"]["random_seed"],
        n_jobs=threads,
    )
    df = pd.DataFrame(
        {
            "variable": list(vars_used),
            "importance_mean": imp[:, 0],
            "importance_std": imp[:, 1],
        }
    ).sort_values("importance_mean", ascending=False)

    ensure_dir(out_dir)
    out = out_dir / "permutation_importance.csv"
    df.to_csv(out, index=False)
    stats_dir = Path(cfg["experiment"].get("stats_dir", "stats"))
    ensure_dir(stats_dir)
    df.to_csv(stats_dir / "permutation_importance.csv", index=False)
    # Plot bar chart
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(df["variable"], df["importance_mean"], yerr=df["importance_std"], color="tab:blue")
    ax.set_ylabel("AUC drop")
    ax.set_title("Permutation importance")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "permutation_importance.png", dpi=150)
    plt.close(fig)
    print(df.to_string(index=False))
    return df


def gain_importance(model: Any, vars_used: Sequence[str]) -> pd.DataFrame:
    """Return gain-based importance reported by the model."""

    if hasattr(model, "feature_gains_"):
        gains = np.asarray(model.feature_gains_)
    elif hasattr(model, "feature_importances_"):
        gains = np.asarray(model.feature_importances_)
    elif hasattr(model, "feature_importance_"):
        gains = np.asarray(model.feature_importance_)
    else:
        raise AttributeError("Model does not expose gain information")

    df = pd.DataFrame({"variable": list(vars_used), "gain": gains})
    df = df.sort_values("gain", ascending=False)
    return df


def _evaluate_model(model: Any, X_pres: np.ndarray, X_bg: np.ndarray) -> float:
    y_true = np.hstack([np.ones(len(X_pres)), np.zeros(len(X_bg))])
    y_score = np.hstack([model.predict(X_pres), model.predict(X_bg)])
    return _auc_score(y_true, y_score)


def jackknife_importance(
    cfg: dict,
    stack_path: Path,
    vars_used: Sequence[str],
    pres_gdf,
    bg_gdf,
    out_dir: Path,
) -> dict:
    """Run isolation and ablation models for each predictor."""

    log.info("Running jackknife importance")
    results = {"baseline": None, "only": {}, "without": {}}

    X_pres_all = extract_from_stack(stack_path, pres_gdf, list(vars_used))
    X_bg_all = extract_from_stack(stack_path, bg_gdf, list(vars_used))
    w_pres = distance_weights(pres_gdf, n_neighbors=-1)
    w_bg = distance_weights(bg_gdf, n_neighbors=-1)
    base_weight = np.concatenate([w_pres, w_bg])

    def _fit(Xp: np.ndarray, Xb: np.ndarray, subset: list[str]) -> Any:
        X = np.vstack([Xp, Xb])
        y = np.hstack([np.ones(len(Xp)), np.zeros(len(Xb))])
        mask = ~np.any(np.isnan(X), axis=1)
        X = X[mask]
        y = y[mask]
        sw = base_weight[mask]
        variances = np.nanvar(X, axis=0)
        keep = variances > 0
        X = X[:, keep]
        sub_vars = [v for v, k in zip(subset, keep) if k]
        n_hinge = cfg["maxent"].get("n_hinge_features")
        if n_hinge == "auto":
            n_hinge = None
        model_kwargs = dict(
            feature_types=cfg["maxent"].get("feature_types"),
            beta_multiplier=cfg["maxent"]["regularization_multiplier"],
            n_cpus=cfg["maxent"]["threads"],
            transform=cfg["maxent"]["output_type"],
            n_lambdas=cfg["maxent"].get("max_iterations", 1000),
            convergence_tolerance=cfg["maxent"].get("convergence_tolerance", 1e-5),
        )
        if n_hinge is not None:
            model_kwargs["n_hinge_features"] = n_hinge
        model = MaxentModel(**model_kwargs)
        model.fit(X, y, sample_weight=sw, labels=sub_vars)
        return model, X, sub_vars

    base_model, _, _ = _fit(X_pres_all, X_bg_all, list(vars_used))
    results["baseline"] = _evaluate_model(base_model, X_pres_all, X_bg_all)

    for i, var in enumerate(vars_used):
        log.info("Jackknife - only %s", var)
        model_only, X_only, _ = _fit(
            X_pres_all[:, [i]], X_bg_all[:, [i]], [var]
        )
        results["only"][var] = _evaluate_model(model_only, X_pres_all[:, [i]], X_bg_all[:, [i]])

        remaining = [v for v in vars_used if v != var]
        log.info("Jackknife - without %s", var)
        if remaining:
            Xp_r = np.delete(X_pres_all, i, axis=1)
            Xb_r = np.delete(X_bg_all, i, axis=1)
            model_drop, _, _ = _fit(Xp_r, Xb_r, remaining)
            results["without"][var] = _evaluate_model(model_drop, Xp_r, Xb_r)
        else:
            results["without"][var] = float("nan")

    ensure_dir(out_dir)
    out = out_dir / "jackknife_importance.yaml"
    Path(out).write_text(yaml.safe_dump(results, sort_keys=False))

    stats_dir = Path(cfg["experiment"].get("stats_dir", "stats"))
    ensure_dir(stats_dir)

    # Bar plots for only/without models
    df_only = pd.Series(results["only"]).sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(6, 4))
    df_only.plot.bar(ax=ax, color="tab:green")
    ax.set_ylabel("AUC")
    ax.set_title("Jackknife - only one variable")
    fig.tight_layout()
    fig.savefig(out_dir / "jackknife_only.png", dpi=150)
    plt.close(fig)
    df_only.to_csv(out_dir / "jackknife_only.csv", header=["AUC"])
    df_only.to_csv(stats_dir / "jackknife_only.csv", header=["AUC"])

    df_without = pd.Series(results["without"]).sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(6, 4))
    df_without.plot.bar(ax=ax, color="tab:red")
    ax.set_ylabel("AUC")
    ax.set_title("Jackknife - without variable")
    fig.tight_layout()
    fig.savefig(out_dir / "jackknife_without.png", dpi=150)
    plt.close(fig)
    df_without.to_csv(out_dir / "jackknife_without.csv", header=["AUC"])
    df_without.to_csv(stats_dir / "jackknife_without.csv", header=["AUC"])

    pd.DataFrame({"baseline_auc": [results["baseline"]]}).to_csv(
        stats_dir / "jackknife_baseline.csv", index=False
    )

    print("Baseline AUC", results["baseline"])
    print("Jackknife only:\n", df_only.to_string())
    print("Jackknife without:\n", df_without.to_string())
    return results


def run_importance(
    cfg: dict,
    stack_path: Path,
    vars_used: Sequence[str],
    model: Any,
    pres_gdf,
    bg_gdf,
    out_dir: Path | None = None,
) -> None:
    """Execute requested importance calculations."""
    if out_dir is None:
        out_dir = Path(cfg["experiment"].get("fi_dir", "feature_importance"))

    if cfg.get("variable_importance", {}).get("permutation", {}).get("enable", False):
        print("    -> Permutation importance")
        permutation_importance(cfg, model, stack_path, vars_used, pres_gdf, bg_gdf, out_dir)

    if cfg.get("variable_importance", {}).get("gain", {}).get("enable", False):
        print("    -> Gain importance")
        try:
            df_gain = gain_importance(model, vars_used)
        except AttributeError as exc:  # pragma: no cover - depends on model
            log.warning("Gain importance unavailable: %s", exc)
        else:
            ensure_dir(out_dir)
            out = out_dir / "gain_importance.csv"
            df_gain.to_csv(out, index=False)
            stats_dir = Path(cfg["experiment"].get("stats_dir", "stats"))
            ensure_dir(stats_dir)
            df_gain.to_csv(stats_dir / "gain_importance.csv", index=False)
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.bar(df_gain["variable"], df_gain["gain"], color="tab:purple")
            ax.set_ylabel("Gain")
            ax.set_title("Gain importance")
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
            fig.tight_layout()
            fig.savefig(out_dir / "gain_importance.png", dpi=150)
            plt.close(fig)
            print(df_gain.to_string(index=False))

    if cfg.get("variable_importance", {}).get("jackknife", {}).get("enable", False):
        print("    -> Jackknife importance")
        jackknife_importance(cfg, stack_path, vars_used, pres_gdf, bg_gdf, out_dir)

