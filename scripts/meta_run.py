"""Meta script to run seasonal multitemporal and monotemporal models.

The script executes four scenarios (winter/summer x multi/mono) with
comparable parameters, collects evaluation metrics and generates summary
boxplots comparing multitemporal and monotemporal approaches. Spatial
cross-validation of the best model is optional and only executed when the
configuration enables it.

Scenarios are defined in a single YAML configuration (default:
``configs/experiments/config_gbm-compare.yaml``) that bundles the individual seasonal settings for
GBM vs. MaxEnt comparison runs.
"""
from __future__ import annotations

import argparse
import subprocess
from datetime import datetime
import tempfile
from pathlib import Path
from copy import deepcopy

import yaml
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np

from scripts.cross_validation import run_cv

COLOR_MAP = {
    ("winter", "multi"): "#1f77b4",
    ("winter", "mono"): "#ff7f0e",
    ("summer", "multi"): "#2ca02c",
    ("summer", "mono"): "#d62728",
}


def _collect_boyce_curves(experiments: list[tuple[dict, Path, dict]]) -> pd.DataFrame:
    records: list[pd.DataFrame] = []
    for res, exp_dir, _cfg in experiments:
        season = res.get("season")
        temporal = res.get("temporal")
        if not season or not temporal:
            continue
        pattern = exp_dir.glob(
            f"output/{season}/test_*/**/stats/*_boyce_curve.csv"
        )
        for csv_path in pattern:
            test_dir = next(
                (p for p in csv_path.parents if p.name.startswith("test_")), None
            )
            if test_dir is not None:
                try:
                    test_year = int(test_dir.name.replace("test_", ""))
                except ValueError:
                    test_year = np.nan
            else:
                test_year = np.nan
            engine_dir = csv_path.parents[1]
            engine_name = engine_dir.name
            if engine_name.startswith("test_"):
                engine_name = "elapid"
            df = pd.read_csv(csv_path)
            if df.empty:
                continue
            df["test_year"] = test_year
            df["season"] = season
            df["temporal"] = temporal
            df["engine"] = engine_name
            records.append(df)
    if records:
        return pd.concat(records, ignore_index=True)
    return pd.DataFrame()


def _find_pe_crossing(hs: np.ndarray, pe: np.ndarray) -> tuple[float, float]:
    hs = np.asarray(hs, dtype=float)
    pe = np.asarray(pe, dtype=float)
    if hs.size == 0 or pe.size == 0:
        return float("nan"), float("nan")
    diff = pe - 1.0
    for i in range(len(diff) - 1):
        if np.isnan(diff[i]) or np.isnan(diff[i + 1]):
            continue
        if diff[i] == 0:
            return float(hs[i]), 1.0
        if diff[i + 1] == 0:
            return float(hs[i + 1]), 1.0
        if diff[i] * diff[i + 1] < 0:
            ratio = diff[i] / (diff[i] - diff[i + 1])
            x_cross = hs[i] + (hs[i + 1] - hs[i]) * ratio
            return float(x_cross), 1.0
    if np.all(np.isnan(diff)):
        return float("nan"), float("nan")
    idx = int(np.nanargmin(np.abs(diff)))
    return float(hs[idx]), float(pe[idx])


def _plot_boyce_summary(curves: pd.DataFrame, out_dir: Path) -> None:
    if curves.empty:
        return
    curves = curves.copy()
    curves["season"] = curves["season"].str.lower()
    curves["temporal"] = curves["temporal"].str.lower()
    if "engine" not in curves.columns:
        curves["engine"] = "elapid"
    for season in sorted(curves["season"].dropna().unique()):
        season_df = curves[curves["season"] == season]
        if season_df.empty:
            continue
        for engine in sorted(season_df["engine"].dropna().unique()):
            engine_df = season_df[
                (season_df["engine"] == engine) & (season_df["dataset"] == "test")
            ]
            if engine_df.empty:
                continue
            for method in ["pearson", "spearman"]:
                method_df = engine_df[engine_df["method"] == method]
                if method_df.empty:
                    continue
                fig, ax = plt.subplots(figsize=(7, 5))
                ax.axhline(1.0, color="black", linestyle="--", linewidth=1)
                handles = []
                for temporal, temp_df in method_df.groupby("temporal"):
                    if temp_df.empty:
                        continue
                    color = COLOR_MAP.get((season, temporal), "#7f7f7f")
                    cbi_by_year = temp_df.groupby("test_year")["cbi_value"].mean()
                    cbi_by_year = cbi_by_year.dropna()
                    if cbi_by_year.empty:
                        continue
                    best_year = cbi_by_year.idxmax()
                    for test_year, year_df in temp_df.groupby("test_year"):
                        if pd.isna(test_year):
                            continue
                        year_df = year_df.sort_values("habitat_suitability")
                        is_best = test_year == best_year
                        alpha = 1.0 if is_best else 0.35
                        linewidth = 2.5 if is_best else 1.3
                        label = f"{temporal.capitalize()} {int(test_year)}"
                        label += f" (CBI={cbi_by_year.get(test_year, np.nan):.2f})"
                        line, = ax.plot(
                            year_df["habitat_suitability"],
                            year_df["pe_ratio"],
                            color=color,
                            alpha=alpha,
                            linewidth=linewidth,
                        )
                        handles.append((label, line))
                        if is_best:
                            x_cross, y_cross = _find_pe_crossing(
                                year_df["habitat_suitability"].to_numpy(),
                                year_df["pe_ratio"].to_numpy(),
                            )
                            if not np.isnan(x_cross) and not np.isnan(y_cross):
                                ax.scatter(
                                    [x_cross],
                                    [y_cross],
                                    color=color,
                                    edgecolors="black",
                                    zorder=5,
                                )
                                ax.plot(
                                    [x_cross, x_cross],
                                    [0, y_cross],
                                    color=color,
                                    alpha=0.7,
                                    linestyle=":",
                                    linewidth=1.2,
                                )
                if not handles:
                    plt.close(fig)
                    continue
                legend_items = {}
                for label, handle in handles:
                    legend_items[label] = handle
                ax.legend(legend_items.values(), legend_items.keys(), frameon=False)
                ax.set_xlabel("Habitat suitability")
                ax.set_ylabel("P/E ratio")
                ax.set_ylim(bottom=0)
                ax.set_xlim(0, 1)
                ax.set_title(
                    f"{season.title()} ({engine.title()}) Boyce curves ({method.title()} CBI)"
                )
                plt.tight_layout()
                out_file = out_dir / f"boyce_curves_{season}_{engine}_{method}.png"
                fig.savefig(out_file, dpi=150)
                plt.close(fig)


def _locate_experiment_dir(exp_cfg: dict, script: str) -> Path:
    """Return path to the most recent experiment directory for ``exp_cfg``."""

    name = exp_cfg.get("name")
    if not name:
        raise ValueError("Configuration missing 'experiment.name'")

    base_dir = Path(exp_cfg.get("base_dir", "exps"))
    prefix = exp_cfg.get("dir_prefix")
    if not prefix:
        prefix = "multitemp" if script == "multitemporal" else "monotemp"

    pattern = f"{prefix}-{name}*"
    candidates = [p for p in base_dir.glob(pattern) if p.is_dir()]
    if not candidates:
        raise FileNotFoundError(
            f"Experiment directory matching '{pattern}' not found under {base_dir}"
        )

    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def _run_model(cfg: dict, script: str) -> tuple[list[dict], Path, dict]:
    """Execute modelling script with configuration ``cfg`` and return metrics."""

    cfg = deepcopy(cfg)
    exp_cfg = cfg.setdefault("experiment", {})
    base_name = exp_cfg.get("name", "")
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    exp_cfg["name"] = f"{base_name}-{ts}" if base_name else ts

    model_cfg = cfg.setdefault("model", {})
    engines = [str(e).lower() for e in model_cfg.get("engines", [])]
    if engines != ["elapid", "gbm"]:
        model_cfg["engines"] = ["elapid", "gbm"]

    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as tmp:
        yaml.safe_dump(cfg, tmp, sort_keys=False)
        tmp_path = Path(tmp.name)

    try:
        subprocess.run([
            "python",
            "-m",
            f"scripts.{script}",
            "--config",
            str(tmp_path),
        ], check=True)
    finally:
        tmp_path.unlink(missing_ok=True)

    season = cfg["multitemporal"]["seasons"][0]
    exp_dir = _locate_experiment_dir(exp_cfg, script)
    summary_file = exp_dir / "output" / season / "temporal_cv_summary.csv"
    summary = pd.read_csv(summary_file)
    if "engine" not in summary.columns:
        summary["engine"] = "elapid"
    results: list[dict] = []
    for engine in summary["engine"].dropna().unique():
        engine_rows = summary[
            (summary["engine"] == engine) & (summary["split"] == "test")
        ]
        if engine_rows.empty:
            continue
        test_row = engine_rows.iloc[0]
        cbi_spear = test_row.get("cbi_spearman_mean")
        cbi_pear = test_row.get("cbi_mean")
        cbi_primary = cbi_spear
        if cbi_primary is None or pd.isna(cbi_primary):
            cbi_primary = cbi_pear
        result = {
            "experiment": base_name or exp_cfg["name"],
            "season": season.lower(),
            "temporal": "multi" if script == "multitemporal" else "mono",
            "engine": str(engine).lower(),
            "auc": test_row.get("auc_mean"),
            "cbi": cbi_primary,
            "cbi_spearman": cbi_spear,
            "cbi_pearson": cbi_pear,
            "omission": test_row.get("omission_mean"),
        }
        results.append(result)
    if not results:
        raise ValueError(f"No test rows found in summary {summary_file}")
    return results, exp_dir, cfg


def _spatial_cv(exp_dir: Path, cfg: dict) -> None:
    """Run spatial cross-validation for the final training set."""
    if not cfg.get("evaluation", {}).get("cross_validation", {}).get("enable", False):
        print(f"Skipping cross-validation for {exp_dir}: disabled in config")
        return

    season = cfg["multitemporal"]["seasons"][0]
    latest_year = max(cfg["multitemporal"]["years"])
    pres_file = exp_dir / f"input-data-test-{latest_year}" / f"{season}_train_points.gpkg"
    bg_file = exp_dir / f"input-data-test-{latest_year}" / f"{season}_background_points.gpkg"
    stack_path = exp_dir / "data-stack" / f"{season}_master.zarr"

    # locate list of predictors used during the training phase
    candidate_vars = [
        exp_dir / "outputs" / "selected_predictors.txt",
        exp_dir / "output" / "selected_predictors.txt",
        exp_dir / "output" / season / "selected_predictors.txt",
    ]
    vars_file = next((p for p in candidate_vars if p.exists()), None)
    if not vars_file:
        matches = list(exp_dir.rglob("selected_predictors.txt"))
        vars_file = matches[0] if matches else None

    if not (pres_file.exists() and bg_file.exists() and stack_path.exists()):
        print(f"Skipping cross-validation for {exp_dir}: missing data")
        return

    pres_gdf = gpd.read_file(pres_file)
    bg_gdf = gpd.read_file(bg_file)
    if not vars_file:
        raise FileNotFoundError(f"selected_predictors.txt not found in {exp_dir}")
    vars_used = [v.strip() for v in vars_file.read_text().splitlines() if v.strip()]

    # ensure required experiment directories for CV outputs and visuals
    cfg = dict(cfg)  # shallow copy
    exp_cfg = cfg.setdefault("experiment", {})
    exp_cfg["stats_dir"] = str(exp_dir / "cv_stats")
    exp_cfg.setdefault("vis_dir", str(exp_dir / "cv_vis"))

    run_cv(cfg, pres_gdf=pres_gdf, bg_gdf=bg_gdf, stack_path=stack_path, vars_used=vars_used)


def _boxplots(df: pd.DataFrame, out_dir: Path) -> None:
    """Create boxplots comparing multitemporal and monotemporal models."""
    metrics = ["auc", "cbi", "omission"]
    if "engine" not in df.columns:
        df = df.copy()
        df["engine"] = "elapid"
    for season in df["season"].dropna().unique():
        season_subset = df[df["season"] == season]
        for engine in season_subset["engine"].dropna().unique():
            subset = season_subset[season_subset["engine"] == engine]
            if subset.empty:
                continue
            for metric in metrics:
                ax = subset.boxplot(column=metric, by="temporal")
                ax.set_ylabel(metric)
                ax.set_title(f"{metric} ({season}, {engine})")
                plt.suptitle("")
                out_file = out_dir / f"boxplot_{metric}_{season}_{engine}.png"
                plt.savefig(out_file, dpi=150, bbox_inches="tight")
                plt.close()


def _load_scenarios(config_path: Path) -> list[tuple[str, str, dict]]:
    """Return (label, script, config) tuples loaded from ``config_path``."""

    data = yaml.safe_load(config_path.read_text()) or {}
    raw_scenarios = data.get("scenarios")
    if not raw_scenarios:
        raise ValueError(f"No scenarios defined in {config_path}")

    entries: list[tuple[str, str, dict]] = []
    if isinstance(raw_scenarios, dict):
        iterator = raw_scenarios.values()
    else:
        iterator = raw_scenarios

    for item in iterator:
        if not isinstance(item, dict):
            raise TypeError("Scenario entries must be mappings")
        cfg = item.get("config")
        if not isinstance(cfg, dict):
            raise ValueError("Scenario missing 'config' mapping")
        script = str(item.get("script", "multitemporal"))
        label = str(
            item.get("label")
            or item.get("key")
            or cfg.get("experiment", {}).get("name")
            or script
        )
        entries.append((label, script, cfg))

    return entries


def main() -> None:
    ap = argparse.ArgumentParser(description="Run GBM vs. MaxEnt seasonal scenarios")
    ap.add_argument(
        "--config",
        default="configs/experiments/config_gbm-compare.yaml",
        help="Aggregate YAML containing scenario definitions",
    )
    args = ap.parse_args()

    config_path = Path(args.config)
    scenarios = _load_scenarios(config_path)
    results = []
    experiments = []
    for label, script, cfg in scenarios:
        print(f"Running scenario {label} ({script})")
        res_list, exp_dir, cfg = _run_model(cfg, script)
        for res in res_list:
            res.setdefault("scenario", label)
            results.append(res)
            experiments.append((res, exp_dir, cfg))

    df = pd.DataFrame(results)
    out_dir = Path("exps")
    out_dir.mkdir(exist_ok=True)
    df.to_csv(out_dir / "metrics_summary.csv", index=False)
    _boxplots(df, out_dir)

    curves_df = _collect_boyce_curves(experiments)
    if not curves_df.empty:
        _plot_boyce_summary(curves_df, out_dir)

    if not df.empty:
        best_res, best_exp_dir, best_cfg = max(
            experiments, key=lambda x: x[0].get("auc", float("nan"))
        )
        if best_res.get("engine") != "elapid":
            print(
                "Best model is not an Elapid run; skipping spatial cross-validation"
            )
        elif best_cfg.get("evaluation", {}).get("cross_validation", {}).get(
            "enable", False
        ):
            print(
                f"Running cross-validation on best model {best_res['experiment']} (AUC {best_res['auc']})"
            )
            _spatial_cv(best_exp_dir, best_cfg)
        else:
            print("Cross-validation disabled in config; skipping best-model CV")


if __name__ == "__main__":
    main()
