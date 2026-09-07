# =============================================================================
# Project Title: Wild Boar Distribution Model for France
# Path: scripts/evaluation.py
# Purpose: Provide additional evaluation metrics and plotting utilities.
# Process Step: Computes CBI, omission rates, ROC/PR curves and other metrics.
# Created by Adrian Meyer, Institute Geomatics, FHNW (adrian.meyer@fhnw.ch)
# Created: July 2025
# Version: v.0.1.0
# =============================================================================
"""
Wrapper for extra metrics - Boyce, omission rate, PR curve and more.
The chosen metrics follow standard recommendations for presence-only models:
 - AUC: traditional measure but can be optimistic for rare species
 - pROC: partial ROC emphasising omission errors
 - SEDI: robust skill score for rare events
 - PR-AUC: more informative for imbalanced datasets
 - CBI: Continuous Boyce Index, ranking accuracy for presence-only data
 - OR10: omission rate using 10th percentile threshold
 - AUCdiff: difference between train and test AUCs (overfitting)
 - AICc: model complexity penalty
 - TOPSIS: composite score maximising all metrics
"""
from __future__ import annotations
import logging
import numpy as np
import geopandas as gpd
from sklearn.metrics import (
    precision_recall_curve,
    average_precision_score,
    roc_auc_score,
    roc_curve,
    precision_score,
    recall_score,
    f1_score,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd

log = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
def boyce_curve(
    prob_pres: np.ndarray, prob_bg: np.ndarray, bins: int = 25
) -> tuple[np.ndarray, np.ndarray]:
    """Return habitat suitability bin centres and P/E ratios."""

    prob_pres = np.asarray(prob_pres, dtype=float)
    prob_bg = np.asarray(prob_bg, dtype=float)
    if prob_pres.size == 0 or prob_bg.size == 0:
        return np.array([], dtype=float), np.array([], dtype=float)

    bin_edges = np.linspace(0.0, 1.0, bins + 1)
    p_hist, _ = np.histogram(prob_pres, bins=bin_edges, density=True)
    a_hist, _ = np.histogram(prob_bg, bins=bin_edges, density=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(a_hist > 0, p_hist / a_hist, np.nan)
    centres = bin_edges[:-1] + np.diff(bin_edges) / 2
    mask = np.isfinite(ratio)
    return centres[mask], ratio[mask]


def continuous_boyce(
    prob_pres: np.ndarray,
    prob_bg: np.ndarray,
    bins: int = 25,
    method: str = "pearson",
) -> float:
    """Continuous Boyce Index using Pearson or Spearman correlation."""

    centres, ratio = boyce_curve(prob_pres, prob_bg, bins=bins)
    if centres.size < 2:
        return float("nan")

    method = method.lower()
    if method == "pearson":
        return float(np.corrcoef(centres, ratio)[0, 1])
    if method == "spearman":
        ranks_x = pd.Series(centres).rank(method="average").to_numpy()
        ranks_y = pd.Series(ratio).rank(method="average").to_numpy()
        return float(np.corrcoef(ranks_x, ranks_y)[0, 1])
    raise ValueError(f"Unknown Boyce correlation method '{method}'")


def partial_roc(y_true: np.ndarray, y_score: np.ndarray, max_fpr: float = 0.1) -> float:
    """Compute partial AUC up to ``max_fpr`` and normalise by this range."""
    fpr, tpr, _ = roc_curve(y_true, y_score)
    mask = fpr <= max_fpr
    if np.sum(mask) < 2:
        return float("nan")
    fpr = np.concatenate(([0.0], fpr[mask], [max_fpr]))
    tpr = np.concatenate(([0.0], tpr[mask], [tpr[mask][-1]]))
    return float(np.trapezoid(tpr, fpr) / max_fpr)


def _sedi(tp: int, fn: int, fp: int, tn: int) -> float:
    """Symmetric extremal dependence index for binary forecasts."""
    hr = tp / (tp + fn) if tp + fn > 0 else 0.0
    far = fp / (fp + tn) if fp + tn > 0 else 0.0
    if hr in (0.0, 1.0) or far in (0.0, 1.0):
        return float("nan")
    num = np.log(hr * (1 - far)) - np.log((1 - hr) * far)
    den = np.log(hr * (1 - far)) + np.log((1 - hr) * far)
    return float(num / den)

# -----------------------------------------------------------------------------  
def _compute_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float | str = 0.5,
    n_params: int | None = None,
    boyce_bins: int = 25,
) -> dict:
    """Return common classification metrics for a score vector.

    ``threshold`` may either be a numeric value or ``"auto"`` to pick the
    probability threshold that maximises the F1 score. ``n_params`` is the
    number of model parameters used to compute the small sample corrected AIC
    (AICc). If ``None`` the AICc value is returned as ``NaN``.
    """
    mask = ~np.isnan(y_score)
    y_score = y_score[mask]
    y_true = y_true[mask]

    precision_curve, recall_curve, pr_thresholds = precision_recall_curve(
        y_true, y_score
    )
    pr_auc = average_precision_score(y_true, y_score)

    if threshold == "auto":
        # thresholds returned by ``precision_recall_curve`` do not include the
        # endpoints 0 and 1. Compute F1 scores for all thresholds and pick the
        # best value to avoid degenerate precision/recall.
        f1_scores = 2 * precision_curve[1:] * recall_curve[1:] / (
            precision_curve[1:] + recall_curve[1:] + 1e-9
        )
        best_idx = int(np.nanargmax(f1_scores))
        thr = float(pr_thresholds[best_idx])
    else:
        thr = float(threshold)

    y_pred = (y_score >= thr).astype(int)
    precision_raw = precision_score(y_true, y_pred, zero_division=0)
    recall_raw = recall_score(y_true, y_pred, zero_division=0)
    omission = 1 - recall_raw
    f1 = f1_score(y_true, y_pred, zero_division=0)
    auc_score = roc_auc_score(y_true, y_score)
    proc = partial_roc(y_true, y_score)
    prob_pres = y_score[y_true == 1]
    prob_bg = y_score[y_true == 0]
    curve_centres, curve_ratio = boyce_curve(prob_pres, prob_bg, bins=boyce_bins)
    cbi_pearson = continuous_boyce(
        prob_pres, prob_bg, bins=boyce_bins, method="pearson"
    )
    cbi_spearman = continuous_boyce(
        prob_pres, prob_bg, bins=boyce_bins, method="spearman"
    )
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    sedi = _sedi(tp, fn, fp, tn)

    log_lik = float(
        np.sum(
            y_true * np.log(y_score + 1e-9)
            + (1 - y_true) * np.log(1 - y_score + 1e-9)
        )
    )
    if n_params is not None:
        k = int(n_params)
        n = len(y_true)
        aic = 2 * k - 2 * log_lik
        if n - k - 1 > 0:
            aicc = aic + (2 * k * (k + 1)) / (n - k - 1)
        else:
            aicc = float("nan")
    else:
        aicc = float("nan")

    return {
        "precision": precision_raw,
        "recall": recall_raw,
        "f1": f1,
        "auc": auc_score,
        "proc": proc,
        "sedi": sedi,
        "cbi": cbi_pearson,
        "cbi_pearson": cbi_pearson,
        "cbi_spearman": cbi_spearman,
        "omission": omission,
        "pr_auc": pr_auc,
        "aicc": aicc,
        "precision_curve": precision_curve,
        "recall_curve": recall_curve,
        "boyce_curve": (curve_centres, curve_ratio),
    }


def evaluate(
    run_name: str,
    y_true_train: np.ndarray,
    y_score_train: np.ndarray,
    cfg: dict,
    y_true_test: np.ndarray | None = None,
    y_score_test: np.ndarray | None = None,
    y_true_val: np.ndarray | None = None,
    y_score_val: np.ndarray | None = None,
    n_params: int | None = None,
    gdf_train: gpd.GeoDataFrame | None = None,
    gdf_test: gpd.GeoDataFrame | None = None,
    gdf_val: gpd.GeoDataFrame | None = None,
) -> None:
    """Compute and log common evaluation metrics for train/test/validation splits."""
    print("    -> Running evaluation metrics")
    log.info("Starting evaluation for %s", run_name)

    from scripts.utils import sanitize_name
    safe_name = sanitize_name(run_name)
    vis_dir = Path(
        cfg.get("experiment", {}).get("vis_dir", cfg["outputs"]["figures_dir"])
    )
    vis_dir.mkdir(parents=True, exist_ok=True)
    stats_root = cfg.get("experiment", {}).get("stats_dir")
    if stats_root:
        stats_dir = Path(stats_root)
    else:
        stats_dir = Path("stats")
    stats_dir.mkdir(parents=True, exist_ok=True)
    eval_cfg = cfg.get("evaluation", {})
    boyce_bins = max(int(eval_cfg.get("boyce_bins", 25)), 1)

    season_name: str | None = None
    test_year: int | None = None
    if "_test_" in run_name:
        prefix, suffix = run_name.split("_test_", 1)
        season_name = prefix.lower()
        try:
            test_year = int(suffix.split("_")[0])
        except ValueError:
            test_year = None

    def _drop_nan(
        y_true: np.ndarray,
        y_score: np.ndarray,
        gdf: gpd.GeoDataFrame | None,
        label: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        mask = ~np.isnan(y_score)
        n_nan = int(np.sum(~mask))
        if n_nan > 0:
            print(f"    -> {label} dataset contained {n_nan} NaN suitability scores")
            log.warning("%s dataset contained %d NaN suitability scores", label, n_nan)
            if gdf is not None:
                try:
                    out_file = vis_dir / f"{safe_name}_{label.lower()}_nan.gpkg"
                    gdf.loc[~mask].to_file(out_file, driver="GPKG")
                    log.info("NaN %s records exported to %s", label.lower(), out_file)
                except Exception as exc:
                    log.warning("Failed to export NaN %s records: %s", label.lower(), exc)
        return y_true[mask], y_score[mask]

    y_true_train, y_score_train = _drop_nan(y_true_train, y_score_train, gdf_train, "Train")
    if y_true_test is not None and y_score_test is not None:
        y_true_test, y_score_test = _drop_nan(y_true_test, y_score_test, gdf_test, "Test")
    if y_true_val is not None and y_score_val is not None:
        y_true_val, y_score_val = _drop_nan(y_true_val, y_score_val, gdf_val, "Validation")
    thr_cfg = eval_cfg.get("classification_threshold", 0.5)
    train_res = _compute_metrics(
        y_true_train,
        y_score_train,
        threshold=thr_cfg,
        n_params=n_params,
        boyce_bins=boyce_bins,
    )
    test_res = None
    if y_true_test is not None and y_score_test is not None and len(y_true_test) > 0:
        test_res = _compute_metrics(
            y_true_test,
            y_score_test,
            threshold=thr_cfg,
            n_params=n_params,
            boyce_bins=boyce_bins,
        )
    val_res = None
    if y_true_val is not None and y_score_val is not None and len(y_true_val) > 0:
        val_res = _compute_metrics(
            y_true_val,
            y_score_val,
            threshold=thr_cfg,
            n_params=n_params,
            boyce_bins=boyce_bins,
        )

    if np.any(y_true_train == 1):
        or10_thr = float(np.percentile(y_score_train[y_true_train == 1], 10))
    else:
        or10_thr = float("nan")

    def _or10(y_true: np.ndarray, y_score: np.ndarray) -> float:
        if np.isnan(or10_thr) or not np.any(y_true == 1):
            return float("nan")
        return float(np.mean(y_score[y_true == 1] < or10_thr))

    train_res["or10"] = _or10(y_true_train, y_score_train)
    if test_res:
        test_res["or10"] = _or10(y_true_test, y_score_test)
    if val_res:
        val_res["or10"] = _or10(y_true_val, y_score_val)

    train_res["auc_diff"] = 0.0
    if test_res:
        auc_diff = float(train_res["auc"] - test_res["auc"])
        train_res["auc_diff"] = auc_diff
        test_res["auc_diff"] = auc_diff
    if val_res:
        val_res["auc_diff"] = float(train_res["auc"] - val_res["auc"])

    def _topsis(m: dict) -> float:
        beneficial = np.array([m.get("auc"), m.get("proc"), m.get("sedi"), m.get("cbi")])
        cost = np.array([m.get("or10"), abs(m.get("auc_diff", 0.0)), max(m.get("aicc", np.nan), 0.0)])
        beneficial = np.nan_to_num(beneficial, nan=0.0)
        cost = np.nan_to_num(cost, nan=0.0)
        cost_norm = cost / (1 + cost)
        vec = np.concatenate([beneficial, 1 - cost_norm])
        ideal = np.ones_like(vec)
        worst = np.zeros_like(vec)
        d_pos = np.linalg.norm(vec - ideal)
        d_neg = np.linalg.norm(vec - worst)
        return float(d_neg / (d_pos + d_neg))

    train_res["topsis"] = _topsis(train_res)
    if test_res:
        test_res["topsis"] = _topsis(test_res)
    if val_res:
        val_res["topsis"] = _topsis(val_res)

    datasets = [
        ("Train", y_true_train, y_score_train, "red", train_res),
    ]
    if val_res:
        datasets.append(("Validation", y_true_val, y_score_val, "green", val_res))
    if test_res:
        datasets.append(("Test", y_true_test, y_score_test, "blue", test_res))

    boyce_dfs: list[pd.DataFrame] = []
    fig_dir = Path(cfg["outputs"]["figures_dir"])
    fig_dir.mkdir(parents=True, exist_ok=True)
    for label, _y_true, _y_score, color, res in datasets:
        centres, ratios = res.get("boyce_curve", (None, None))
        if centres is None or ratios is None:
            continue
        curve_df = pd.DataFrame(
            {
                "habitat_suitability": np.asarray(centres, dtype=float),
                "pe_ratio": np.asarray(ratios, dtype=float),
            }
        )
        curve_df = curve_df.replace([np.inf, -np.inf], np.nan).dropna()
        if curve_df.empty:
            continue
        curve_df = curve_df.sort_values("habitat_suitability")
        for method in ("pearson", "spearman"):
            cbi_val = res.get(f"cbi_{method}")
            if cbi_val is None or np.isnan(cbi_val):
                continue
            fig_pe, ax_pe = plt.subplots()
            ax_pe.axhline(1.0, color="black", linestyle="--", linewidth=1)
            ax_pe.plot(
                curve_df["habitat_suitability"],
                curve_df["pe_ratio"],
                marker="o",
                color=color,
            )
            ax_pe.set_xlabel("Habitat suitability")
            ax_pe.set_ylabel("P/E ratio")
            ax_pe.set_ylim(bottom=0)
            ax_pe.set_title(
                f"{label} Boyce curve ({method.title()} CBI = {cbi_val:.3f})"
            )
            pe_path = fig_dir / f"{safe_name}_{label.lower()}_boyce_{method}.png"
            fig_pe.savefig(pe_path, dpi=150, bbox_inches="tight")
            plt.close(fig_pe)
            boyce_dfs.append(
                curve_df.assign(
                    run_name=run_name,
                    dataset=label.lower(),
                    method=method,
                    cbi_value=cbi_val,
                    bin_count=boyce_bins,
                    season=season_name,
                    test_year=test_year,
                )
            )

    # Calibration plot
    bins = np.linspace(0, 1, 11)
    fig_cal, ax_cal = plt.subplots()
    for label, y_true, y_score, color, _ in datasets:
        bin_idx = np.digitize(y_score, bins) - 1
        obs_freq = [
            y_true[bin_idx == i].mean() if np.any(bin_idx == i) else 0
            for i in range(len(bins) - 1)
        ]
        ax_cal.plot(
            (bins[:-1] + bins[1:]) / 2,
            obs_freq,
            marker="o",
            color=color,
            label=label,
        )
    ax_cal.set_xlabel("Predicted suitability")
    ax_cal.set_ylabel("Observed frequency")
    ax_cal.set_ylim(0, 1)
    ax_cal.legend()
    fig_cal.savefig(fig_dir / "calibration_plot.png", dpi=150)
    plt.close(fig_cal)

    # PR curve
    fig, ax = plt.subplots()
    for label, _, _, color, res in datasets:
        ax.step(res["recall_curve"], res["precision_curve"], where="post", color=color, label=label)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("PR-curve")
    ax.legend()
    fig.savefig(fig_dir / f"{safe_name}_pr.png", dpi=150)
    plt.close(fig)

    # ROC curve
    fig_roc, ax_roc = plt.subplots()
    ax_roc.plot([0, 1], [0, 1], color="black", linestyle="--")
    for label, y_true, y_score, color, _ in datasets:
        fpr, tpr, _ = roc_curve(y_true, y_score)
        ax_roc.plot(fpr, tpr, color=color, label=label)
    ax_roc.set_xlabel("1 - Specificity")
    ax_roc.set_ylabel("Sensitivity")
    ax_roc.set_title("ROC curve")
    ax_roc.legend()
    fig_roc.savefig(fig_dir / f"{safe_name}_roc.png", dpi=150)
    plt.close(fig_roc)

    # ------------------------------------------------------------------
    # Threshold sweep for AUC and Continuous Boyce Index
    thresholds = np.arange(0.05, 0.95, 0.01)
    curves: dict[str, dict[str, list[float] | str]] = {}
    for label, y_true, y_score, color, _ in datasets:
        auc_curve: list[float] = []
        cbi_curve: list[float] = []
        for thr in thresholds:
            y_pred_bin = (y_score >= thr).astype(int)
            try:
                auc_curve.append(roc_auc_score(y_true, y_pred_bin))
            except ValueError:
                auc_curve.append(np.nan)
            pres_thr = y_score[(y_true == 1) & (y_score >= thr)]
            bg_thr = y_score[(y_true == 0) & (y_score >= thr)]
            if len(pres_thr) >= 2 and len(bg_thr) >= 2:
                cbi_curve.append(
                    continuous_boyce(pres_thr, bg_thr, bins=boyce_bins)
                )
            else:
                cbi_curve.append(np.nan)
        curves[label] = {"auc": auc_curve, "cbi": cbi_curve, "color": color}

    train_cbi = curves.get("Train", {}).get("cbi", [])
    best_idx = int(np.nanargmax(train_cbi)) if np.any(~np.isnan(train_cbi)) else 0
    best_thr = float(thresholds[best_idx])
    log.info("Optimal threshold (CBI): %.3f", best_thr)

    fig_thr_auc, ax_thr_auc = plt.subplots()
    for label, data in curves.items():
        ax_thr_auc.plot(thresholds, data["auc"], color=data["color"], label=label)
    ax_thr_auc.axvline(best_thr, color="red", linestyle="--")
    ax_thr_auc.set_xlabel("Threshold")
    ax_thr_auc.set_ylabel("AUC")
    ax_thr_auc.legend()
    fig_thr_auc.savefig(fig_dir / f"{safe_name}_threshold_auc.png", dpi=150)
    plt.close(fig_thr_auc)

    fig_thr_cbi, ax_thr_cbi = plt.subplots()
    for label, data in curves.items():
        ax_thr_cbi.plot(thresholds, data["cbi"], color=data["color"], label=label)
    ax_thr_cbi.axvline(best_thr, color="red", linestyle="--")
    ax_thr_cbi.set_xlabel("Threshold")
    ax_thr_cbi.set_ylabel("CBI")
    ax_thr_cbi.legend()
    fig_thr_cbi.savefig(fig_dir / f"{safe_name}_threshold_cbi.png", dpi=150)
    plt.close(fig_thr_cbi)

    if boyce_dfs:
        boyce_df = pd.concat(boyce_dfs, ignore_index=True)
        boyce_df = boyce_df.sort_values(
            ["dataset", "method", "habitat_suitability"]
        ).reset_index(drop=True)
        boyce_path = stats_dir / f"{safe_name}_boyce_curve.csv"
        boyce_df.to_csv(boyce_path, index=False)

    rows = [
        [
            "train",
            train_res["auc"],
            train_res["proc"],
            train_res["sedi"],
            train_res["or10"],
            train_res["auc_diff"],
            train_res["aicc"],
            train_res["cbi"],
            train_res.get("cbi_pearson"),
            train_res.get("cbi_spearman"),
            train_res["topsis"],
            train_res["precision"],
            train_res["recall"],
            train_res["f1"],
            train_res["pr_auc"],
            train_res["omission"],
        ]
    ]
    if val_res:
        rows.append([
            "validation",
            val_res["auc"],
            val_res["proc"],
            val_res["sedi"],
            val_res["or10"],
            val_res["auc_diff"],
            val_res["aicc"],
            val_res["cbi"],
            val_res.get("cbi_pearson"),
            val_res.get("cbi_spearman"),
            val_res["topsis"],
            val_res["precision"],
            val_res["recall"],
            val_res["f1"],
            val_res["pr_auc"],
            val_res["omission"],
        ])
    if test_res:
        rows.append([
            "test",
            test_res["auc"],
            test_res["proc"],
            test_res["sedi"],
            test_res["or10"],
            test_res["auc_diff"],
            test_res["aicc"],
            test_res["cbi"],
            test_res.get("cbi_pearson"),
            test_res.get("cbi_spearman"),
            test_res["topsis"],
            test_res["precision"],
            test_res["recall"],
            test_res["f1"],
            test_res["pr_auc"],
            test_res["omission"],
        ])

    df_disp = pd.DataFrame(
        rows,
        columns=[
            "split",
            "auc",
            "proc",
            "sedi",
            "or10",
            "auc_diff",
            "aicc",
            "cbi",
            "cbi_pearson",
            "cbi_spearman",
            "topsis",
            "precision",
            "recall",
            "f1",
            "pr_auc",
            "omission",
        ],
    )
    print(df_disp.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    log.info(
        "Train metrics - CBI_pearson %.3f CBI_spearman %.3f OR10 %.3f AUC %.3f pROC %.3f SEDI %.3f AUCdiff %.3f AICc %.3f TOPSIS %.3f Precision %.3f Recall %.3f F1 %.3f PR-AUC %.3f Omission %.3f",
        train_res.get("cbi_pearson"),
        train_res.get("cbi_spearman"),
        train_res["or10"],
        train_res["auc"],
        train_res["proc"],
        train_res["sedi"],
        train_res["auc_diff"],
        train_res["aicc"],
        train_res["topsis"],
        train_res["precision"],
        train_res["recall"],
        train_res["f1"],
        train_res["pr_auc"],
        train_res["omission"],
    )
    if val_res:
        log.info(
            "Validation metrics - CBI_pearson %.3f CBI_spearman %.3f OR10 %.3f AUC %.3f pROC %.3f SEDI %.3f AUCdiff %.3f AICc %.3f TOPSIS %.3f Precision %.3f Recall %.3f F1 %.3f PR-AUC %.3f Omission %.3f",
            val_res.get("cbi_pearson"),
            val_res.get("cbi_spearman"),
            val_res["or10"],
            val_res["auc"],
            val_res["proc"],
            val_res["sedi"],
            val_res["auc_diff"],
            val_res["aicc"],
            val_res["topsis"],
            val_res["precision"],
            val_res["recall"],
            val_res["f1"],
            val_res["pr_auc"],
            val_res["omission"],
        )
        print(
            "    -> Validation AUC "
            f"{val_res['auc']:.3f} CBI (Pearson) {val_res.get('cbi_pearson', float('nan')):.3f} "
            f"CBI (Spearman) {val_res.get('cbi_spearman', float('nan')):.3f} AICc {val_res['aicc']:.3f}"
        )
    if test_res:
        log.info(
            "Test metrics - CBI_pearson %.3f CBI_spearman %.3f OR10 %.3f AUC %.3f pROC %.3f SEDI %.3f AUCdiff %.3f AICc %.3f TOPSIS %.3f Precision %.3f Recall %.3f F1 %.3f PR-AUC %.3f Omission %.3f",
            test_res.get("cbi_pearson"),
            test_res.get("cbi_spearman"),
            test_res["or10"],
            test_res["auc"],
            test_res["proc"],
            test_res["sedi"],
            test_res["auc_diff"],
            test_res["aicc"],
            test_res["topsis"],
            test_res["precision"],
            test_res["recall"],
            test_res["f1"],
            test_res["pr_auc"],
            test_res["omission"],
        )
        print(
            "    -> Test AUC "
            f"{test_res['auc']:.3f} CBI (Pearson) {test_res.get('cbi_pearson', float('nan')):.3f} "
            f"CBI (Spearman) {test_res.get('cbi_spearman', float('nan')):.3f} AICc {test_res['aicc']:.3f}"
        )

    print(
        "    -> Train AUC "
        f"{train_res['auc']:.3f} CBI (Pearson) {train_res.get('cbi_pearson', float('nan')):.3f} "
        f"CBI (Spearman) {train_res.get('cbi_spearman', float('nan')):.3f} AICc {train_res['aicc']:.3f}"
    )

    df_disp.to_csv(stats_dir / f"{safe_name}_evaluation.csv", index=False)
    log.info("Finished evaluation for %s", run_name)
