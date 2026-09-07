from __future__ import annotations

import argparse
import logging
import os
import pickle
import sys
from pathlib import Path
from typing import Any

import geopandas as gpd
import pandas as pd

from review.scripts.common import load_yaml, now_iso, write_json


def _first_existing(candidates: list[Path]) -> Path:
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError("None of these files exist:\n" + "\n".join(map(str, candidates)))


def _find_model(run_dir: Path, engine: str) -> Path:
    model_dir = run_dir / engine / "models"
    candidates = sorted(model_dir.rglob("*.pkl"))
    if not candidates:
        candidates = sorted((run_dir / engine).rglob("*.pkl"))
    if not candidates:
        raise FileNotFoundError(f"No pickle model found below {run_dir / engine}")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _load_selected(stack_dir: Path) -> list[str]:
    path = stack_dir / "selected_predictors.txt"
    if not path.exists():
        return []
    values = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        values.extend(v.strip() for v in line.split(",") if v.strip())
    return list(dict.fromkeys(values))


def _model_predictors(model: Any) -> list[str]:
    """Return the ordered predictor names recorded by a fitted model, if any."""

    booster = getattr(model, "booster", None)
    feature_name = getattr(booster, "feature_name", None)
    if callable(feature_name):
        names = feature_name()
        if names:
            return [str(name) for name in names]

    names = getattr(model, "feature_names_in_", None)
    if names is not None and len(names):
        return [str(name) for name in names]
    return []


def _predictors_for_model(model: Any, selected: list[str]) -> list[str]:
    """Reconcile the stack selection with the predictors used during fitting."""

    fitted = _model_predictors(model)
    if not fitted:
        return selected

    available = set(selected)
    missing = [name for name in fitted if name not in available]
    if missing:
        raise RuntimeError(
            "Model predictors are absent from selected_predictors.txt: "
            + ", ".join(missing)
        )
    return fitted


def _best_test_rows(metrics: pd.DataFrame) -> pd.DataFrame:
    frame = metrics.copy()
    if "split" in frame.columns:
        frame = frame[frame["split"].astype(str).str.casefold() == "test"]
    score_col = next(
        (c for c in ("cbi_spearman", "cbi", "boyce_spearman", "auc") if c in frame.columns),
        None,
    )
    if score_col is None:
        raise KeyError("No CBI/AUC score column in temporal_cv_metrics.csv")
    frame[score_col] = pd.to_numeric(frame[score_col], errors="coerce")
    if "engine" not in frame.columns:
        frame["engine"] = "elapid"
    return (
        frame.sort_values(score_col, ascending=False, na_position="last")
        .groupby("engine", as_index=False, sort=False)
        .head(1)
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Compute best-fold feature importance.")
    parser.add_argument("--repo-root", required=True, type=Path)
    parser.add_argument("--experiment-dir", required=True, type=Path)
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--season", default="Winter")
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--n-repeats", type=int, default=10)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    logger = logging.getLogger("review_corrections.feature_importance")
    repo_root = args.repo_root.resolve()
    os.chdir(repo_root)
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from scripts.variable_importance import gain_importance, permutation_importance

    exp_dir = args.experiment_dir.resolve()
    scenario_dir = exp_dir.parent.parent
    resolved_config_path = scenario_dir / "resolved_config.yaml"
    if not resolved_config_path.exists():
        raise FileNotFoundError(f"Resolved scenario config not found: {resolved_config_path}")

    metrics_path = exp_dir / "output" / args.season / "temporal_cv_metrics.csv"
    metrics = pd.read_csv(metrics_path)
    best_rows = _best_test_rows(metrics)
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    index_rows: list[dict[str, Any]] = []

    for _, row in best_rows.iterrows():
        engine = str(row.get("engine", "elapid")).lower()
        year_value = row.get("test_year", row.get("year", row.get("held_out_year")))
        if pd.isna(year_value):
            raise ValueError(f"Best row for {engine} lacks held-out year")
        year = int(year_value)
        fold_dir = exp_dir / "output" / args.season / f"test_{year}"
        input_dir = exp_dir / f"input-data-test-{year}"
        stack_dir = input_dir / f"{args.season}_stack"
        model_path = _find_model(fold_dir, engine)
        train_points = _first_existing(
            [
                input_dir / f"{args.season}_train_points.gpkg",
                fold_dir / "points" / "train.gpkg",
                fold_dir / engine / "points" / "train.gpkg",
            ]
        )
        background_points = _first_existing(
            [
                input_dir / f"{args.season}_background_points.gpkg",
                fold_dir / "points" / "background.gpkg",
                fold_dir / engine / "points" / "background.gpkg",
            ]
        )
        selected = _load_selected(stack_dir)
        out_dir = output_root / engine / f"test_{year}"
        out_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            "Feature importance: scenario=%s engine=%s year=%s model=%s",
            args.scenario,
            engine,
            year,
            model_path,
        )
        with model_path.open("rb") as handle:
            model = pickle.load(handle)

        if not selected:
            raise RuntimeError(f"No selected predictors found in {stack_dir}")
        model_selected = _predictors_for_model(model, selected)
        if model_selected != selected:
            logger.info(
                "Using %d fitted predictors instead of %d stack predictors for %s",
                len(model_selected),
                len(selected),
                engine,
            )

        pres_gdf = gpd.read_file(train_points)
        bg_gdf = gpd.read_file(background_points)

        perm_out = out_dir / "permutation"
        perm_out.mkdir(parents=True, exist_ok=True)
        perm_cfg = load_yaml(resolved_config_path)
        perm_cfg.setdefault("variable_importance", {}).setdefault("permutation", {})
        perm_cfg["variable_importance"]["permutation"]["enable"] = True
        perm_cfg["variable_importance"]["permutation"]["n_repeats"] = args.n_repeats
        perm_cfg.setdefault("experiment", {})
        perm_cfg["experiment"]["random_seed"] = args.random_state
        perm_cfg["experiment"]["stats_dir"] = str(perm_out / "stats")
        # Importance parallelises across predictors; keep the worker count within
        # a practical Windows process budget.
        perm_cfg.setdefault("maxent", {})
        perm_cfg["maxent"]["threads"] = min(32, max(1, len(model_selected)))

        permutation_importance(
            perm_cfg,
            model,
            stack_dir,
            model_selected,
            pres_gdf,
            bg_gdf,
            perm_out,
        )
        perm_csv = perm_out / "permutation_importance.csv"
        if not perm_csv.exists():
            raise FileNotFoundError(f"Permutation importance produced no CSV: {perm_csv}")

        gain_out = out_dir / "gain"
        gain_out.mkdir(parents=True, exist_ok=True)
        try:
            gain_df = gain_importance(model, model_selected)
            gain_csv = gain_out / "gain_importance.csv"
            gain_df.to_csv(gain_csv, index=False)
        except Exception as exc:
            logger.warning("Gain importance unavailable for %s: %s", engine, exc)
            gain_csv = None

        index_rows.append(
            {
                "scenario": args.scenario,
                "engine": engine,
                "test_year": year,
                "selection_score": row.get("cbi_spearman", row.get("auc")),
                "model": str(model_path),
                "stack": str(stack_dir),
                "training_points": str(train_points),
                "background_points": str(background_points),
                "selected_predictors": "|".join(model_selected),
                "permutation_importance": str(perm_csv),
                "gain_importance": str(gain_csv) if gain_csv else None,
            }
        )

    pd.DataFrame(index_rows).to_csv(output_root / "feature_importance_index.csv", index=False)
    write_json(
        {
            "status": "complete",
            "scenario": args.scenario,
            "finished_utc": now_iso(),
            "records": index_rows,
        },
        output_root / "feature_importance_state.json",
    )
    (output_root / "COMPLETE").write_text(now_iso() + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
