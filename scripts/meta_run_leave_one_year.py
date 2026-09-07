"""Meta runner executing all leave-one-year-out scenarios.

The script mirrors :mod:`scripts.meta_run` but executes every available
leave-one-year-out (LOYO) rotation for each configured scenario
(winter/summer × multitemporal/monotemporal).  After the modelling scripts
finish, the resulting metrics are collated into a timestamped analysis
directory which stores CSV summaries, diagnostic boxplots, and a ranking of
scenario/engine combinations based on Spearman Continuous Boyce Index (CBI)
performance.  The best performing model for each combination is subjected to
spatial cross-validation (if supported by the configuration) and a detailed
feature-importance study.

Usage
-----

.. code-block:: console

   python -m scripts.meta_run_leave_one_year --config config_gbm_loyo_compare.yaml

The configuration file follows the same structure as
``config_gbm-compare.yaml`` but must enable ``run_test_year: all`` so the
underlying modelling scripts iterate across every available year.
"""

from __future__ import annotations

import argparse
import pickle
import subprocess
import tempfile
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Iterable

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from scripts.cross_validation import run_cv
from scripts.utils import ensure_dir, sanitize_name
from scripts.variable_importance import gain_importance, permutation_importance


def _load_scenarios(config_path: Path) -> list[tuple[str, str, dict]]:
    """Return ``(label, script, config)`` entries from ``config_path``."""

    data = yaml.safe_load(config_path.read_text()) or {}
    raw_scenarios = data.get("scenarios")
    if not raw_scenarios:
        raise ValueError(f"No scenarios defined in {config_path}")

    if isinstance(raw_scenarios, dict):
        iterator: Iterable = raw_scenarios.values()
    else:
        iterator = raw_scenarios

    scenarios: list[tuple[str, str, dict]] = []
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
        scenarios.append((label, script, cfg))
    return scenarios


def _ensure_engines(cfg: dict) -> None:
    """Ensure both Elapid and GBM engines are configured."""

    model_cfg = cfg.setdefault("model", {})
    raw = [str(engine).lower() for engine in model_cfg.get("engines", [])]
    engines: list[str] = []
    if not raw:
        engines = ["elapid", "gbm"]
    else:
        seen: set[str] = set()
        for item in raw:
            if item in {"elapid", "maxent"} and "elapid" not in seen:
                engines.append("elapid")
                seen.add("elapid")
            elif item == "gbm" and "gbm" not in seen:
                engines.append("gbm")
                seen.add("gbm")
        for required in ("elapid", "gbm"):
            if required not in seen:
                engines.append(required)
                seen.add(required)
    model_cfg["engines"] = engines


def _locate_experiment_dir(exp_cfg: dict, script: str) -> Path:
    """Return the most recent experiment directory for ``exp_cfg``."""

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


def _gather_run_infos(
    label: str, script: str, cfg: dict, exp_dir: Path
) -> list[dict]:
    """Collect metadata describing every engine/test-year run."""

    seasons = cfg.get("multitemporal", {}).get("seasons")
    if not seasons:
        raise ValueError("Configuration missing 'multitemporal.seasons'")
    if len(seasons) != 1:
        raise ValueError("Scenarios must target a single season")
    season = seasons[0]
    season_lower = season.lower()
    temporal = "multitemporal" if script == "multitemporal" else "monotemporal"

    out_root = exp_dir / "output" / season
    run_infos: list[dict] = []
    if not out_root.exists():
        return run_infos

    for test_dir in sorted(out_root.glob("test_*")):
        if not test_dir.is_dir():
            continue
        try:
            test_year = int(test_dir.name.split("_", 1)[1])
        except (IndexError, ValueError):
            continue
        input_dir = exp_dir / f"input-data-test-{test_year}"
        for engine_dir in sorted(test_dir.iterdir()):
            if not engine_dir.is_dir():
                continue
            engine = engine_dir.name.lower()
            if engine not in {"elapid", "gbm"}:
                continue
            run_infos.append(
                {
                    "scenario": label,
                    "script": script,
                    "season": season,
                    "season_lower": season_lower,
                    "temporal": temporal,
                    "engine": engine,
                    "test_year": test_year,
                    "engine_dir": engine_dir,
                    "run_dir": test_dir,
                    "input_dir": input_dir,
                    "exp_dir": exp_dir,
                    "cfg": cfg,
                }
            )
    return run_infos


def _run_scenario(label: str, script: str, base_cfg: dict) -> tuple[pd.DataFrame, Path, dict, list[dict]]:
    """Execute a modelling scenario and return metrics and run metadata."""

    cfg = deepcopy(base_cfg)
    exp_cfg = cfg.setdefault("experiment", {})
    base_name = exp_cfg.get("name") or sanitize_name(label.lower().replace(" ", "_"))
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    exp_cfg["name"] = f"{base_name}-{ts}"
    exp_cfg.setdefault("random_seed", 42)

    _ensure_engines(cfg)

    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as tmp:
        yaml.safe_dump(cfg, tmp, sort_keys=False)
        tmp_path = Path(tmp.name)

    try:
        subprocess.run(
            ["python", "-m", f"scripts.{script}", "--config", str(tmp_path)],
            check=True,
        )
    finally:
        tmp_path.unlink(missing_ok=True)

    exp_dir = _locate_experiment_dir(exp_cfg, script)
    seasons = cfg["multitemporal"]["seasons"]
    if len(seasons) != 1:
        raise ValueError("Each scenario must specify exactly one season")
    season = seasons[0]
    metrics_path = exp_dir / "output" / season / "temporal_cv_metrics.csv"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Metrics file not found: {metrics_path}")
    metrics_df = pd.read_csv(metrics_path)
    metrics_df = metrics_df.copy()
    metrics_df["scenario"] = label
    metrics_df["season"] = metrics_df["season"].str.lower()
    metrics_df["season_label"] = metrics_df["season"].str.title()
    metrics_df["engine"] = metrics_df["engine"].str.lower()
    metrics_df["temporal"] = (
        "multitemporal" if script == "multitemporal" else "monotemporal"
    )
    metrics_df["experiment_dir"] = str(exp_dir)
    metrics_df["experiment_name"] = exp_cfg["name"]

    run_infos = _gather_run_infos(label, script, cfg, exp_dir)
    return metrics_df, exp_dir, cfg, run_infos


def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten MultiIndex columns produced by :func:`pandas.DataFrame.pivot`."""

    new_cols: list[str] = []
    for col in df.columns:
        if isinstance(col, tuple):
            new_cols.append("_".join(str(part) for part in col if part))
        else:
            new_cols.append(str(col))
    df.columns = new_cols
    return df


def _summarise_combinations(metrics: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return long and wide summaries of validation/test performance."""

    relevant = metrics[metrics["split"].isin(["validation", "test"])].copy()
    group_cols = ["scenario", "season", "temporal", "engine", "split"]
    summary = (
        relevant.groupby(group_cols, dropna=False)
        .agg(
            cbi_spearman_mean=("cbi_spearman", "mean"),
            cbi_spearman_std=("cbi_spearman", "std"),
            auc_mean=("auc", "mean"),
            auc_std=("auc", "std"),
            n_runs=("test_year", "nunique"),
        )
        .reset_index()
    )

    pivot = summary.pivot_table(
        index=["scenario", "season", "temporal", "engine"],
        columns="split",
        values=["cbi_spearman_mean", "auc_mean"],
    )
    pivot = _flatten_columns(pivot.reset_index())
    pivot = pivot.rename(
        columns={
            "cbi_spearman_mean_test": "cbi_spearman_mean_test",
            "cbi_spearman_mean_validation": "cbi_spearman_mean_validation",
            "auc_mean_test": "auc_mean_test",
            "auc_mean_validation": "auc_mean_validation",
        }
    )
    return summary, pivot


def _select_best_runs(metrics: pd.DataFrame) -> list[dict]:
    """Return the best test-year row for each scenario/engine combination."""

    best_records: list[dict] = []
    test_rows = metrics[metrics["split"] == "test"]
    group_cols = ["scenario", "season", "temporal", "engine"]
    for combo, group in test_rows.groupby(group_cols):
        if group.empty:
            continue
        idx = group["cbi_spearman"].astype(float).idxmax()
        best = group.loc[idx]
        val_match = metrics[
            (metrics["scenario"] == best["scenario"])
            & (metrics["season"] == best["season"])
            & (metrics["temporal"] == best["temporal"])
            & (metrics["engine"] == best["engine"])
            & (metrics["test_year"] == best["test_year"])
            & (metrics["split"] == "validation")
        ]
        val_score = (
            float(val_match["cbi_spearman"].iloc[0]) if not val_match.empty else np.nan
        )
        best_records.append(
            {
                "scenario": best["scenario"],
                "season": best["season"],
                "temporal": best["temporal"],
                "engine": best["engine"],
                "test_year": int(best["test_year"]),
                "test_cbi_spearman": float(best["cbi_spearman"]),
                "test_auc": float(best["auc"]),
                "validation_cbi_spearman": val_score,
            }
        )
    return best_records


def _find_predictors(run_info: dict) -> Path | None:
    """Locate the ``selected_predictors.txt`` for ``run_info``."""

    season = run_info["season"]
    test_year = run_info["test_year"]
    engine_dir: Path = run_info["engine_dir"]
    run_dir: Path = run_info["run_dir"]
    exp_dir: Path = run_info["exp_dir"]
    input_dir: Path = run_info["input_dir"]

    candidates = [
        engine_dir / "selected_predictors.txt",
        engine_dir / "stats" / "selected_predictors.txt",
        run_dir / "selected_predictors.txt",
        run_dir / "stats" / "selected_predictors.txt",
        exp_dir / "output" / season / "selected_predictors.txt",
        input_dir / f"{season}_stack" / "selected_predictors.txt",
        exp_dir / f"input-data-test-{test_year}" / f"{season}_stack" / "selected_predictors.txt",
    ]
    for cand in candidates:
        if cand.exists():
            return cand
    matches = list(engine_dir.rglob("selected_predictors.txt"))
    if matches:
        return matches[0]
    matches = list(exp_dir.rglob("selected_predictors.txt"))
    return matches[0] if matches else None


def _locate_model_file(engine_dir: Path) -> Path:
    """Return the newest pickled model stored in ``engine_dir``."""

    model_dir = engine_dir / "models"
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory missing: {model_dir}")
    candidates = [p for p in model_dir.glob("*.pkl") if p.is_file()]
    if not candidates:
        raise FileNotFoundError(f"No pickled models found in {model_dir}")
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def _run_spatial_cv(run_info: dict, meta_dir: Path) -> None:
    """Execute spatial CV for the supplied run if enabled."""

    cfg = deepcopy(run_info["cfg"])
    cv_cfg = cfg.setdefault("evaluation", {}).setdefault("cross_validation", {})
    if not cv_cfg or not cv_cfg.get("params"):
        print(
            f"Skipping CV for {run_info['scenario']} {run_info['engine']}: missing configuration"
        )
        return
    cv_cfg["enable"] = True

    season = run_info["season"]
    test_year = run_info["test_year"]
    input_dir: Path = run_info["input_dir"]
    pres_file = input_dir / f"{season}_train_points.gpkg"
    bg_file = input_dir / f"{season}_background_points.gpkg"
    stack_path = input_dir / f"{season}_stack"
    if not (pres_file.exists() and bg_file.exists() and stack_path.exists()):
        print(
            f"Skipping CV for {run_info['scenario']} {run_info['engine']} test {test_year}: missing training data"
        )
        return

    vars_file = _find_predictors(run_info)
    if not vars_file:
        print(
            f"Skipping CV for {run_info['scenario']} {run_info['engine']} test {test_year}: predictors not found"
        )
        return
    vars_used = [line.strip() for line in vars_file.read_text().splitlines() if line.strip()]
    if not vars_used:
        print(
            f"Skipping CV for {run_info['scenario']} {run_info['engine']} test {test_year}: empty predictor list"
        )
        return

    pres_gdf = gpd.read_file(pres_file)
    bg_gdf = gpd.read_file(bg_file)

    combo_name = (
        f"{run_info['season_lower']}_{run_info['temporal']}_{run_info['engine']}_test_{test_year}"
    )
    out_root = meta_dir / "cross_validation" / sanitize_name(combo_name)
    stats_dir = out_root / "stats"
    vis_dir = out_root / "visualizations"
    cfg.setdefault("experiment", {})["stats_dir"] = str(stats_dir)
    cfg["experiment"]["vis_dir"] = str(vis_dir)
    ensure_dir(stats_dir)
    ensure_dir(vis_dir)

    run_cv(
        cfg,
        pres_gdf=pres_gdf,
        bg_gdf=bg_gdf,
        stack_path=stack_path,
        vars_used=vars_used,
    )


def _run_feature_importance(run_info: dict, meta_dir: Path, repeats: int = 10) -> None:
    """Execute permutation (and gain) importance for ``run_info``."""

    cfg = deepcopy(run_info["cfg"])
    perm_cfg = cfg.setdefault("variable_importance", {}).setdefault("permutation", {})
    perm_cfg["enable"] = True
    perm_cfg["n_repeats"] = max(int(perm_cfg.get("n_repeats", 0) or 0), repeats)

    season = run_info["season"]
    test_year = run_info["test_year"]
    input_dir: Path = run_info["input_dir"]
    stack_path = input_dir / f"{season}_stack"
    pres_file = input_dir / f"{season}_train_points.gpkg"
    bg_file = input_dir / f"{season}_background_points.gpkg"
    if not (stack_path.exists() and pres_file.exists() and bg_file.exists()):
        print(
            f"Skipping feature importance for {run_info['scenario']} {run_info['engine']} test {test_year}: missing inputs"
        )
        return

    vars_file = _find_predictors(run_info)
    if not vars_file:
        print(
            f"Skipping feature importance for {run_info['scenario']} {run_info['engine']} test {test_year}: predictors not found"
        )
        return
    vars_used = [line.strip() for line in vars_file.read_text().splitlines() if line.strip()]
    if not vars_used:
        print(
            f"Skipping feature importance for {run_info['scenario']} {run_info['engine']} test {test_year}: empty predictor list"
        )
        return

    model_path = _locate_model_file(run_info["engine_dir"])
    with open(model_path, "rb") as fh:
        model = pickle.load(fh)

    pres_gdf = gpd.read_file(pres_file)
    bg_gdf = gpd.read_file(bg_file)

    combo_name = (
        f"{run_info['season_lower']}_{run_info['temporal']}_{run_info['engine']}_test_{test_year}"
    )
    out_root = meta_dir / "feature_importance" / sanitize_name(combo_name)
    stats_dir = out_root / "stats"
    cfg.setdefault("experiment", {})["stats_dir"] = str(stats_dir)
    cfg["experiment"]["fi_dir"] = str(out_root)
    ensure_dir(out_root)
    ensure_dir(stats_dir)

    permutation_importance(
        cfg,
        model,
        stack_path,
        vars_used,
        pres_gdf,
        bg_gdf,
        out_root,
    )

    try:
        gain_df = gain_importance(model, vars_used)
    except Exception as exc:  # pragma: no cover - best effort logging
        print(
            f"Gain-based importance unavailable for {run_info['scenario']} {run_info['engine']} test {test_year}: {exc}"
        )
    else:
        gain_df.to_csv(out_root / "gain_importance.csv", index=False)


def _plot_boxplots(metrics: pd.DataFrame, out_dir: Path) -> None:
    """Create diagnostic boxplots for Spearman CBI and AUC."""

    subset = metrics[metrics["split"] == "test"].copy()
    if subset.empty:
        return
    temporal_map = {"multitemporal": "Multi", "monotemporal": "Mono"}
    subset["combo"] = (
        subset["temporal"].map(temporal_map).fillna(subset["temporal"]) +
        "-" + subset["engine"].str.upper()
    )

    def _plot(df: pd.DataFrame, metric: str, title: str, suffix: str) -> None:
        if df.empty:
            return
        fig, ax = plt.subplots(figsize=(9, 5))
        df.boxplot(column=metric, by="combo", ax=ax)
        ax.set_xlabel("Scenario")
        ax.set_ylabel(metric.replace("_", " ").title())
        ax.set_title(title)
        plt.suptitle("")
        fig.tight_layout()
        fig.savefig(out_dir / f"boxplot_{metric}_{suffix}.png", dpi=150)
        plt.close(fig)

    ensure_dir(out_dir)
    for metric in ["cbi_spearman", "auc"]:
        _plot(subset, metric, f"{metric.replace('_', ' ').title()} across seasons", "overall")
        for season, season_df in subset.groupby("season_label"):
            _plot(
                season_df,
                metric,
                f"{metric.replace('_', ' ').title()} – {season}",
                f"{metric}_{season.lower()}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run all leave-one-year-out seasonal comparison scenarios"
    )
    parser.add_argument(
        "--config",
        default="config_gbm_loyo_compare.yaml",
        help="Aggregate YAML containing scenario definitions",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional output directory. Defaults to exps/meta_loyo_<timestamp>",
    )
    parser.add_argument(
        "--feature-importance-repeats",
        type=int,
        default=10,
        help="Minimum number of permutation repeats for best-run feature importance",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    scenarios = _load_scenarios(config_path)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = Path(args.output_dir) if args.output_dir else Path("exps") / f"meta_loyo_{timestamp}"
    ensure_dir(out_dir)

    metrics_frames: list[pd.DataFrame] = []
    run_infos: list[dict] = []
    for label, script, cfg in scenarios:
        print(f"Running scenario {label} ({script})")
        metrics_df, exp_dir, cfg_used, infos = _run_scenario(label, script, cfg)
        metrics_frames.append(metrics_df)
        for info in infos:
            info["cfg"] = cfg_used
        run_infos.extend(infos)

    if not metrics_frames:
        raise RuntimeError("No scenario metrics collected")

    metrics = pd.concat(metrics_frames, ignore_index=True)
    metrics.to_csv(out_dir / "temporal_cv_metrics_all.csv", index=False)

    summary_long, summary_wide = _summarise_combinations(metrics)
    summary_long.to_csv(out_dir / "combination_summary_long.csv", index=False)
    summary_wide = summary_wide.fillna(np.nan)
    summary_wide["ranking_score"] = summary_wide.get(
        "cbi_spearman_mean_test", pd.Series(dtype=float)
    )
    summary_wide = summary_wide.sort_values(
        ["ranking_score", "cbi_spearman_mean_validation"], ascending=[False, False]
    )
    summary_wide.to_csv(out_dir / "combination_summary_wide.csv", index=False)

    best_runs = _select_best_runs(metrics)
    best_df = pd.DataFrame(best_runs)
    best_df.to_csv(out_dir / "best_runs.csv", index=False)

    info_index = {
        (info["scenario"], info["temporal"], info["engine"], info["test_year"]): info
        for info in run_infos
    }

    for record in best_runs:
        key = (record["scenario"], record["temporal"], record["engine"], record["test_year"])
        info = info_index.get(key)
        if not info:
            print(
                f"Run directory for {record['scenario']} {record['engine']} test {record['test_year']} not found"
            )
            continue
        if info["engine"] == "elapid":
            _run_spatial_cv(info, out_dir)
        _run_feature_importance(info, out_dir, repeats=args.feature_importance_repeats)

    _plot_boxplots(metrics, out_dir)

    if not summary_wide.empty:
        best_combo = summary_wide.iloc[0]
        print(
            "Top-performing combination (by Spearman CBI test mean): "
            f"{best_combo['scenario']} / {best_combo['engine']} with "
            f"CBI_test={best_combo.get('cbi_spearman_mean_test', float('nan')):.3f}"
        )


if __name__ == "__main__":
    main()

