#!/usr/bin/env python3
"""
Post-process an already completed winter hunting-correction review run.

This script deliberately DOES NOT train, fit, rebuild, or rerun any SDM model.
It consumes the saved models/stacks/results from an existing review run and
writes a NEW finalized publication-output folder.

It fixes the main post-run issues identified in the first correction bundle:

1. creates a clean successful output/archive without stale FAILURE.json/RUNNING.lock;
2. filters publication metrics to corrected/reference scenarios only, so optional
   no-hunting controls can never contaminate the main corrected summaries;
3. recomputes 95% CIs with Student's t for the seven LOYO folds;
4. computes MaxEnt hunting-bag permutation importance across ALL LOYO folds from
   saved fitted models (cheap relative to model training);
5. keeps the existing full best-fold feature-importance exports as descriptive
   material, while adding the all-fold hunting result as the correction evidence;
6. reclassifies output-path differences as runtime-only and the DEM Aspect
   difference as a pre-existing preserved configuration difference rather than
   an inherent "temporal-method" difference;
7. generates an explicit manuscript-text replacement/attention register;
8. performs much stricter final validation (scenario/engine/fold coverage,
   hunting retention, source rasters, maps, configuration audit, archive state);
9. writes repository-hygiene findings without mutating the repository.

Typical use
-----------
From the maxent-repo root on branch review-iteration:

    python review-finalize-existing-run.py ^
      --source-run review/output/20260826_184244_EEST_hunting-correction

The new finalized folder is written next to the source run by default.

Optional:
    --n-repeats 10
    --predict-threads 1
    --skip-hunting-importance
    --output-root review/output
    --archive-max-member-mb 100
    --resume-output
    --force

Scientific scope
----------------
The all-fold importance calculation permutes ONLY the hunting-bag predictor by
default. This directly tests the reviewer-identified correction at low cost.
It does not refit models. Existing full best-fold feature-importance outputs are
retained as descriptive material. A full all-feature/all-fold permutation pass
would be much more expensive and is intentionally not performed here.
"""

from __future__ import annotations

# Keep numerical libraries from exploding thread counts during repeated predictions.
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import argparse
import csv
import hashlib
import json
import logging
import math
import pickle
import re
import shutil
import subprocess
import sys
import traceback
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

LOG = logging.getLogger("review_finalize")

SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parent
SCRIPT_VERSION = "2026-08-28.2-maxent-importance-only"

CORE_ROLES = {"corrected", "reference"}
CORE_DATASETS = ("allpoints", "gbif")
CORE_TEMPORAL = ("monotemporal", "multitemporal")
CORE_ENGINES = ("elapid", "gbm")
IMPORTANCE_ENGINES = ("elapid",)
EXPECTED_YEARS = tuple(range(2017, 2024))
EXPECTED_SPLITS = ("train", "validation", "test")

METRICS = (
    "cbi_spearman",
    "cbi_pearson",
    "auc",
    "proc",
    "pr_auc",
    "omission",
    "aicc",
)

STALE_NAMES = {
    "FAILURE.json",
    "RUNNING.lock",
}

KNOWN_REPO_HYGIENE_FILES = (
    "4",
    "PACKAGE_MANIFEST.csv",
    "PACKAGE_VALIDATION.txt",
    "SHA256SUMS.txt",
    "VALIDATION_SUCCESS",
)


# ---------------------------------------------------------------------------
# Basic I/O
# ---------------------------------------------------------------------------

def configure_logging(log_file: Path, level: str = "INFO") -> None:
    numeric = getattr(logging, level.upper(), logging.INFO)
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(numeric)
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    root.addHandler(console)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setFormatter(formatter)
    root.addHandler(fh)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )


def write_csv_rows(rows: Iterable[dict[str, Any]], path: Path) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows generated._\n"
    display = frame.copy().replace({np.nan: ""})
    headers = [str(c) for c in display.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for _, row in display.iterrows():
        cells = [
            str(v).replace("|", r"\|").replace("\n", " ")
            for v in row
        ]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def write_table_bundle(
    frame: pd.DataFrame,
    stem: Path,
    *,
    index: bool = False,
) -> list[Path]:
    stem.parent.mkdir(parents=True, exist_ok=True)
    outputs = [
        stem.with_suffix(".csv"),
        stem.with_suffix(".tsv"),
        stem.with_suffix(".md"),
        stem.with_suffix(".tex"),
    ]
    frame.to_csv(outputs[0], index=index)
    frame.to_csv(outputs[1], sep="\t", index=index)
    outputs[2].write_text(markdown_table(frame), encoding="utf-8")
    try:
        tex = frame.to_latex(index=index, escape=True, na_rep="")
    except Exception as exc:
        tex = f"% LaTeX export failed: {exc}\n"
    outputs[3].write_text(tex, encoding="utf-8")
    return outputs


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def hardlink_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        destination.unlink()
    try:
        os.link(source, destination)
    except (OSError, AttributeError):
        shutil.copy2(source, destination)


def copy_tree_lightweight(
    source: Path,
    destination: Path,
    *,
    excluded_names: set[str] | None = None,
) -> list[Path]:
    """Copy a tree using hardlinks where possible; skip stale run-state files."""
    excluded_names = excluded_names or set()
    copied: list[Path] = []
    if not source.exists():
        return copied
    for path in sorted(source.rglob("*")):
        rel = path.relative_to(source)
        target = destination / rel
        if path.is_dir():
            target.mkdir(parents=True, exist_ok=True)
            continue
        if path.name in excluded_names:
            continue
        hardlink_or_copy(path, target)
        copied.append(target)
    return copied


def safe_unlink(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        pass


# ---------------------------------------------------------------------------
# Run and repository discovery
# ---------------------------------------------------------------------------

def require_repo_root() -> None:
    required = [
        REPO_ROOT / "scripts" / "utils.py",
        REPO_ROOT / "review" / "scripts" / "run_feature_importance.py",
        REPO_ROOT / "review" / "config" / "review-corrections.yaml",
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Place this script in the maxent-repo root. Missing: "
            + ", ".join(missing)
        )


def resolve_source_run(value: Path) -> Path:
    source = value if value.is_absolute() else REPO_ROOT / value
    source = source.resolve()
    if not source.exists():
        raise FileNotFoundError(f"Source review run does not exist: {source}")
    registry = source / "00_RUN" / "scenario_registry.json"
    if not registry.exists():
        raise FileNotFoundError(f"Source scenario registry missing: {registry}")
    return source


def git_info() -> dict[str, Any]:
    result: dict[str, Any] = {}
    try:
        branch = subprocess.run(
            ["git", "branch", "--show-current"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        result["branch"] = branch.stdout.strip()
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        result["commit"] = commit.stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        result["dirty"] = bool(status.stdout.strip())
        result["status_porcelain"] = status.stdout.splitlines()
    except Exception as exc:
        result["error"] = str(exc)
    return result


def make_run_id(timezone_name: str = "Europe/Zurich") -> str:
    now = datetime.now(ZoneInfo(timezone_name))
    zone = now.tzname() or "LOCAL"
    return now.strftime("%Y%m%d_%H%M%S_") + zone + "_hunting-correction-finalized"


def prepare_final_dir(
    source_run: Path,
    output_root: Path | None,
    run_id: str | None,
    force: bool,
    *,
    resume_existing: bool = False,
) -> Path:
    root = output_root
    if root is None:
        root = source_run.parent
    elif not root.is_absolute():
        root = (REPO_ROOT / root).resolve()
    final_dir = root / (run_id or make_run_id())
    if final_dir.exists():
        if force and resume_existing:
            raise ValueError("--force and --resume-output are mutually exclusive")
        if resume_existing:
            # Keep checkpoint and existing publication-layer work. Individual files
            # are refreshed idempotently later in the workflow.
            pass
        elif force:
            shutil.rmtree(final_dir)
        else:
            raise FileExistsError(
                f"Final output folder already exists: {final_dir}. "
                "Use --resume-output to continue it or --force to replace it."
            )
    for rel in (
        "00_RUN",
        "01_SOURCE",
        "02_DIAGNOSTICS/hunting_all_fold_importance",
        "02_DIAGNOSTICS/configuration_audit",
        "03_MANUSCRIPT",
        "04_SUPPLEMENT",
        "06_MACHINE_READABLE",
        "07_ARCHIVE",
    ):
        (final_dir / rel).mkdir(parents=True, exist_ok=True)
    return final_dir


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def find_metrics_file(record: dict[str, Any]) -> Path:
    exp_dir = Path(record["experiment_dir"])
    season = str(record.get("season", "Winter"))
    preferred = exp_dir / "output" / season / "temporal_cv_metrics.csv"
    if preferred.exists():
        return preferred
    candidates = list(exp_dir.rglob("temporal_cv_metrics.csv"))
    if not candidates:
        raise FileNotFoundError(
            f"No temporal_cv_metrics.csv for scenario {record.get('id')}"
        )
    return max(candidates, key=lambda p: p.stat().st_mtime)


def collect_core_metrics(
    registry: list[dict[str, Any]],
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Collect corrected/reference metrics only. Controls are intentionally excluded."""
    frames: list[pd.DataFrame] = []
    used_registry: list[dict[str, Any]] = []
    for record in registry:
        if str(record.get("role", "")).lower() not in CORE_ROLES:
            continue
        path = find_metrics_file(record)
        frame = pd.read_csv(path)
        frame["scenario"] = record["id"]
        frame["dataset"] = record["dataset"]
        frame["temporal"] = record["temporal"]
        frame["scenario_role"] = record["role"]
        frame["season_review"] = record.get("season", "Winter")
        if "engine" not in frame.columns:
            frame["engine"] = "elapid"
        if "test_year" not in frame.columns:
            for candidate in ("year", "held_out_year"):
                if candidate in frame.columns:
                    frame["test_year"] = frame[candidate]
                    break
        # aliases
        if "cbi_spearman" not in frame.columns and "cbi" in frame.columns:
            frame["cbi_spearman"] = frame["cbi"]
        if "cbi_pearson" not in frame.columns:
            frame["cbi_pearson"] = np.nan
        if "omission" not in frame.columns and "omission_rate" in frame.columns:
            frame["omission"] = frame["omission_rate"]
        frames.append(frame)
        used_registry.append(record)
    if not frames:
        raise RuntimeError("No corrected/reference scenario metrics found.")
    return pd.concat(frames, ignore_index=True, sort=False), used_registry


def student_t_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    """Summarize test folds with Student-t 95% CIs (df=n-1)."""
    try:
        from scipy.stats import t as student_t
    except ImportError as exc:
        raise RuntimeError("scipy is required for Student-t confidence intervals") from exc

    work = metrics.copy()
    if "split" in work.columns:
        work = work[work["split"].astype(str).str.casefold() == "test"]
    group_cols = [
        "scenario",
        "scenario_role",
        "dataset",
        "temporal",
        "engine",
        "season_review",
    ]
    rows: list[dict[str, Any]] = []
    for keys, group in work.groupby(group_cols, dropna=False, sort=False):
        row = dict(zip(group_cols, keys))
        years = pd.to_numeric(group["test_year"], errors="coerce").dropna()
        row["n_folds"] = int(years.nunique())
        row["ci_method"] = "Student-t 95% CI of LOYO-fold mean (df=n-1)"
        for metric in METRICS:
            if metric not in group.columns:
                continue
            values = pd.to_numeric(group[metric], errors="coerce").dropna()
            n = len(values)
            mean = float(values.mean()) if n else math.nan
            sd = float(values.std(ddof=1)) if n > 1 else math.nan
            if n > 1:
                crit = float(student_t.ppf(0.975, df=n - 1))
                half = crit * sd / math.sqrt(n)
                low, high = mean - half, mean + half
            else:
                crit = half = low = high = math.nan
            row[f"{metric}_mean"] = mean
            row[f"{metric}_sd"] = sd
            row[f"{metric}_tcrit"] = crit
            row[f"{metric}_ci95_low"] = low
            row[f"{metric}_ci95_high"] = high
        rows.append(row)
    out = pd.DataFrame(rows)
    dataset_order = {"allpoints": 0, "gbif": 1}
    engine_order = {"elapid": 0, "gbm": 1}
    temporal_order = {"monotemporal": 0, "multitemporal": 1}
    out["_d"] = out["dataset"].map(dataset_order).fillna(99)
    out["_e"] = out["engine"].map(engine_order).fillna(99)
    out["_t"] = out["temporal"].map(temporal_order).fillna(99)
    return (
        out.sort_values(["_d", "_e", "_t"])
        .drop(columns=["_d", "_e", "_t"])
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# All-fold hunting-only permutation importance
# ---------------------------------------------------------------------------

def first_existing(candidates: Iterable[Path]) -> Path:
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        "None of the required files exists:\n" + "\n".join(str(p) for p in candidates)
    )


def find_model(fold_dir: Path, engine: str) -> Path:
    preferred = fold_dir / engine / "models"
    candidates = list(preferred.rglob("*.pkl")) if preferred.exists() else []
    if not candidates:
        root = fold_dir / engine
        candidates = list(root.rglob("*.pkl")) if root.exists() else []
    if not candidates:
        raise FileNotFoundError(f"No saved model below {fold_dir / engine}")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def load_selected(stack_dir: Path) -> list[str]:
    path = stack_dir / "selected_predictors.txt"
    if not path.exists():
        return []
    values: list[str] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        values.extend(v.strip() for v in line.split(",") if v.strip())
    return list(dict.fromkeys(values))


def _as_nonempty_names(value: Any) -> list[str]:
    """Return a clean list of predictor labels, or an empty list."""
    if value is None:
        return []
    try:
        names = [str(v) for v in value]
    except Exception:
        return []
    names = [name for name in names if name and name.casefold() != "none"]
    return names


def model_predictor_metadata(model: Any) -> tuple[list[str], str, int | None]:
    """Return exact fitted raw-covariate labels, their source and expected width.

    ELAPID MaxentModel is the important case here. During fit(), the repository
    passes ``labels=vars_used``; ELAPID stores those *raw predictor labels* on
    ``model.transformer.labels_``. This is the authoritative fold-specific list
    after zero-variance predictors have been removed.
    """
    # LightGBM / GBM first: its booster exposes the exact fitted feature names.
    booster = getattr(model, "booster", None)
    feature_name = getattr(booster, "feature_name", None)
    if callable(feature_name):
        try:
            names = _as_nonempty_names(feature_name())
        except Exception:
            names = []
        if names:
            return names, "booster.feature_name()", len(names)

    # ELAPID MaxEnt: exact raw predictor names supplied through fit(labels=...).
    transformer = getattr(model, "transformer", None)
    if transformer is not None:
        labels = _as_nonempty_names(getattr(transformer, "labels_", None))

        expected: int | None = None
        estimators = getattr(transformer, "estimators_", None)
        if isinstance(estimators, dict):
            # Linear is present in the study configs and directly records the
            # number of continuous raw covariates seen during fit.
            linear = estimators.get("linear")
            n_features = getattr(linear, "n_features_in_", None) if linear is not None else None
            if n_features is not None:
                try:
                    expected = int(n_features)
                except Exception:
                    expected = None

        # If categorical data ever enter a future run, labels_ remains the
        # authoritative total raw-covariate list whereas linear n_features_in_
        # may cover continuous variables only. Prefer the label count then.
        if labels:
            expected = len(labels)
            return labels, "model.transformer.labels_", expected

        if expected is None:
            continuous = getattr(transformer, "continuous_", None)
            categorical = getattr(transformer, "categorical_", None)
            try:
                n_con = len(continuous) if continuous is not None else 0
                n_cat = len(categorical) if categorical is not None else 0
                if n_con or n_cat:
                    expected = n_con + n_cat
            except Exception:
                pass

        # No names, but preserve the fitted width so callers cannot silently
        # feed a differently sized selected_predictors.txt matrix.
        if expected is not None:
            return [], "ELAPID transformer fitted width", expected

    # Generic sklearn-style models.
    for attr in ("feature_names_in_", "feature_name_", "feature_names_"):
        names = _as_nonempty_names(getattr(model, attr, None))
        if names:
            return names, f"model.{attr}", len(names)

    n_features = getattr(model, "n_features_in_", None)
    if n_features is not None:
        try:
            return [], "model.n_features_in_", int(n_features)
        except Exception:
            pass

    return [], "unavailable", None


def model_predictors(model: Any) -> list[str]:
    """Compatibility helper returning only the exact fitted predictor names."""
    names, _, _ = model_predictor_metadata(model)
    return names


def predictors_for_model(model: Any, selected: list[str]) -> list[str]:
    """Reconcile stack candidates with the model's exact fitted raw predictors.

    Never truncate or guess by position. A saved model with a fitted width that
    differs from ``selected_predictors.txt`` must either expose its fitted names
    (ELAPID does through ``transformer.labels_``) or fail loudly.
    """
    fitted, source, expected = model_predictor_metadata(model)

    if fitted:
        missing = [v for v in fitted if v not in set(selected)]
        if missing:
            raise RuntimeError(
                f"Saved model predictor labels from {source} are absent from "
                "selected_predictors.txt: " + ", ".join(missing)
            )
        if expected is not None and len(fitted) != expected:
            raise RuntimeError(
                f"Internal model metadata disagreement: {source} supplies "
                f"{len(fitted)} labels but fitted width is {expected}."
            )
        return fitted

    if expected is None:
        # Only safe when no contradictory fitted-width information exists.
        LOG.warning(
            "Saved model exposes neither fitted predictor labels nor fitted width; "
            "using all %d selected predictors as a legacy fallback",
            len(selected),
        )
        return selected

    if len(selected) == expected:
        LOG.warning(
            "Saved model exposes fitted width=%d but no labels; selected predictor "
            "count matches exactly, so existing order is retained",
            expected,
        )
        return selected

    raise RuntimeError(
        "Cannot safely reconstruct fitted predictors: selected_predictors.txt has "
        f"{len(selected)} entries, but the saved model expects {expected}. "
        "The script refuses to truncate/reorder predictors without model labels."
    )


def hunting_predictor_name(predictors: list[str]) -> str:
    candidates = [
        p for p in predictors
        if "hunting" in p.casefold()
        or "hunt" in p.casefold()
        or "tableau" in p.casefold()
    ]
    if len(candidates) == 1:
        return candidates[0]
    if "hunting_bag" in predictors:
        return "hunting_bag"
    if not candidates:
        raise RuntimeError("No hunting-named predictor in fitted predictor list.")
    raise RuntimeError(
        "Multiple hunting-like predictors found; refusing to guess: "
        + ", ".join(candidates)
    )


def limit_model_threads(model: Any, n_threads: int) -> None:
    """Best-effort in-memory thread cap; never refits the model."""
    for obj in (
        model,
        getattr(model, "estimator", None),
        getattr(model, "model", None),
    ):
        if obj is None:
            continue
        set_params = getattr(obj, "set_params", None)
        if callable(set_params):
            for params in (
                {"n_jobs": n_threads},
                {"num_threads": n_threads},
                {"nthread": n_threads},
            ):
                try:
                    set_params(**params)
                except Exception:
                    pass
    booster = getattr(model, "booster", None)
    reset = getattr(booster, "reset_parameter", None)
    if callable(reset):
        try:
            reset({"num_threads": n_threads})
        except Exception:
            pass


def predict_1d(model: Any, X: np.ndarray) -> np.ndarray:
    pred = model.predict(X)
    pred = np.asarray(pred, dtype=float)
    if pred.ndim > 1:
        # Be conservative for probability-like 2-column predictions.
        if pred.shape[1] == 2:
            pred = pred[:, 1]
        else:
            pred = pred.reshape(pred.shape[0], -1)[:, 0]
    return pred.ravel()


def auc_score(y: np.ndarray, score: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, score))


def compute_fold_hunting_importance(
    record: dict[str, Any],
    *,
    year: int,
    engine: str,
    n_repeats: int,
    random_state: int,
    predict_threads: int,
) -> dict[str, Any]:
    """Permutation AUC drop for hunting only, using a saved fitted model."""
    import geopandas as gpd

    # Import extraction only; no training/process_season modules are called.
    from scripts.utils import extract_from_stack

    exp_dir = Path(record["experiment_dir"])
    season = str(record.get("season", "Winter"))
    fold_dir = exp_dir / "output" / season / f"test_{year}"
    input_dir = exp_dir / f"input-data-test-{year}"
    stack_dir = input_dir / f"{season}_stack"

    model_path = find_model(fold_dir, engine)
    train_points = first_existing(
        [
            input_dir / f"{season}_train_points.gpkg",
            fold_dir / "points" / "train.gpkg",
            fold_dir / engine / "points" / "train.gpkg",
        ]
    )
    background_points = first_existing(
        [
            input_dir / f"{season}_background_points.gpkg",
            fold_dir / "points" / "background.gpkg",
            fold_dir / engine / "points" / "background.gpkg",
        ]
    )

    selected = load_selected(stack_dir)
    if not selected:
        raise RuntimeError(f"No selected_predictors.txt in {stack_dir}")

    with model_path.open("rb") as fh:
        model = pickle.load(fh)
    limit_model_threads(model, predict_threads)

    predictors = predictors_for_model(model, selected)
    fitted_names, fitted_source, expected_width = model_predictor_metadata(model)
    LOG.info(
        "Predictor reconciliation: stack-selected=%d, fitted=%d, expected-width=%s, source=%s",
        len(selected),
        len(predictors),
        expected_width if expected_width is not None else "unknown",
        fitted_source,
    )
    if expected_width is not None and len(predictors) != expected_width:
        raise RuntimeError(
            f"Predictor reconciliation failed before extraction: resolved "
            f"{len(predictors)} predictors but model expects {expected_width}."
        )
    if len(predictors) != len(selected):
        removed = [v for v in selected if v not in set(predictors)]
        LOG.info(
            "Using fold-specific fitted predictor subset (%d/%d); excluded before fit: %s",
            len(predictors),
            len(selected),
            ", ".join(removed) if removed else "<order/subset differs>",
        )
    hunting_name = hunting_predictor_name(predictors)
    hunting_idx = predictors.index(hunting_name)

    pres = gpd.read_file(train_points)
    bg = gpd.read_file(background_points)

    X_pres = extract_from_stack(stack_dir, pres, predictors)
    X_bg = extract_from_stack(stack_dir, bg, predictors)
    X = np.vstack([X_pres, X_bg])
    y = np.concatenate(
        [
            np.ones(len(X_pres), dtype=np.uint8),
            np.zeros(len(X_bg), dtype=np.uint8),
        ]
    )

    finite = np.all(np.isfinite(X), axis=1)
    dropped = int((~finite).sum())
    X = X[finite]
    y = y[finite]
    if len(np.unique(y)) < 2:
        raise RuntimeError("Permutation sample lacks both presence/background classes.")

    # Threadpoolctl limits BLAS/OpenMP where supported.
    try:
        from threadpoolctl import threadpool_limits
        limiter = threadpool_limits(limits=predict_threads)
    except Exception:
        limiter = None

    try:
        base_pred = predict_1d(model, X)
        base_auc = auc_score(y, base_pred)

        drops: list[float] = []
        for repeat in range(n_repeats):
            rng = np.random.default_rng(
                random_state + year * 1009 + repeat * 37
                + (0 if engine == "elapid" else 500_000)
            )
            X_perm = X.copy()
            order = rng.permutation(X_perm.shape[0])
            X_perm[:, hunting_idx] = X_perm[order, hunting_idx]
            perm_auc = auc_score(y, predict_1d(model, X_perm))
            drops.append(base_auc - perm_auc)
    finally:
        if limiter is not None:
            limiter.restore_original_limits()

    return {
        "scenario": record["id"],
        "scenario_role": record["role"],
        "dataset": record["dataset"],
        "temporal": record["temporal"],
        "season": season,
        "engine": engine,
        "test_year": year,
        "variable": hunting_name,
        "n_repeats": n_repeats,
        "n_samples": int(len(y)),
        "dropped_nan_rows": dropped,
        "base_auc_training_points": base_auc,
        "importance_mean_auc_drop": float(np.mean(drops)),
        "importance_sd_auc_drop": float(np.std(drops, ddof=1))
        if len(drops) > 1 else 0.0,
        "importance_min_auc_drop": float(np.min(drops)),
        "importance_max_auc_drop": float(np.max(drops)),
        "model_path": str(model_path),
        "stack_dir": str(stack_dir),
        "train_points": str(train_points),
        "background_points": str(background_points),
        "fitted_predictor_count": len(predictors),
        "hunting_predictor_index": hunting_idx,
    }



def preflight_saved_model_predictors(
    registry: list[dict[str, Any]],
    output_dir: Path,
) -> pd.DataFrame:
    """Validate predictor metadata for saved models used by the diagnostic."""
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    failures: list[str] = []
    core = [r for r in registry if str(r.get("role", "")).lower() in CORE_ROLES]

    for record in core:
        exp_dir = Path(record["experiment_dir"])
        season = str(record.get("season", "Winter"))
        years = [int(y) for y in record.get("years", EXPECTED_YEARS)]
        engines = [
            str(e).lower()
            for e in record.get("engines", CORE_ENGINES)
            if str(e).lower() in IMPORTANCE_ENGINES
        ]
        for engine in engines:
            for year in years:
                fold_dir = exp_dir / "output" / season / f"test_{year}"
                input_dir = exp_dir / f"input-data-test-{year}"
                stack_dir = input_dir / f"{season}_stack"
                model_path = find_model(fold_dir, engine)
                selected = load_selected(stack_dir)
                row: dict[str, Any] = {
                    "scenario": record["id"],
                    "engine": engine,
                    "test_year": year,
                    "model_path": str(model_path),
                    "stack_dir": str(stack_dir),
                    "stack_selected_count": len(selected),
                    "status": "pass",
                    "detail": "",
                }
                try:
                    if not selected:
                        raise RuntimeError(f"No selected_predictors.txt in {stack_dir}")
                    with model_path.open("rb") as fh:
                        model = pickle.load(fh)
                    predictors = predictors_for_model(model, selected)
                    _, source, expected = model_predictor_metadata(model)
                    hunting = hunting_predictor_name(predictors)
                    removed = [v for v in selected if v not in set(predictors)]
                    row.update(
                        {
                            "fitted_predictor_count": len(predictors),
                            "expected_width": expected,
                            "metadata_source": source,
                            "hunting_predictor": hunting,
                            "removed_before_fit": "|".join(removed),
                        }
                    )
                    if expected is not None and len(predictors) != expected:
                        raise RuntimeError(
                            f"resolved {len(predictors)} predictors but model expects {expected}"
                        )
                except Exception as exc:
                    row["status"] = "fail"
                    row["detail"] = str(exc)
                    failures.append(
                        f"{record['id']} / {engine} / {year}: {exc}"
                    )
                rows.append(row)

    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / "saved_model_predictor_preflight.csv", index=False)
    write_table_bundle(frame, output_dir / "saved_model_predictor_preflight")

    if failures:
        raise RuntimeError(
            "Saved-model predictor preflight failed before permutation prediction:\n- "
            + "\n- ".join(failures)
        )

    LOG.info(
        "Saved-model predictor preflight passed for %d model folds", len(frame)
    )
    return frame

def compute_allfold_hunting_importance(
    registry: list[dict[str, Any]],
    output_dir: Path,
    *,
    n_repeats: int,
    random_state: int,
    predict_threads: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = output_dir / "hunting_permutation_importance_all_folds.partial.csv"
    rows: list[dict[str, Any]] = []
    completed: set[tuple[str, str, int]] = set()

    if checkpoint.exists() and checkpoint.stat().st_size > 0:
        prior = pd.read_csv(checkpoint)
        required = {"scenario", "engine", "test_year", "n_repeats"}
        if required.issubset(prior.columns):
            # Older checkpoints may contain GBM rows from before this diagnostic
            # was aligned with the manuscript's MaxEnt-only importance scope.
            prior = prior[
                prior["engine"].astype(str).str.lower().isin(IMPORTANCE_ENGINES)
            ].copy()
            prior_repeats = set(
                pd.to_numeric(prior["n_repeats"], errors="coerce")
                .dropna().astype(int).tolist()
            )
            if prior_repeats and prior_repeats != {int(n_repeats)}:
                raise RuntimeError(
                    f"Checkpoint {checkpoint} was created with n_repeats="
                    f"{sorted(prior_repeats)}, but this run requests {n_repeats}. "
                    "Use the same --n-repeats or start a new --run-id."
                )
            rows = prior.to_dict(orient="records")
            completed = {
                (str(r["scenario"]), str(r["engine"]).lower(), int(r["test_year"]))
                for r in rows
            }
            LOG.info(
                "Resuming hunting permutation importance from checkpoint: %d fold(s) complete",
                len(completed),
            )
        else:
            LOG.warning(
                "Ignoring malformed checkpoint %s; missing columns: %s",
                checkpoint,
                sorted(required.difference(prior.columns)),
            )

    core = [r for r in registry if str(r.get("role", "")).lower() in CORE_ROLES]
    total = sum(
        len(r.get("years", EXPECTED_YEARS))
        * sum(
            str(engine).lower() in IMPORTANCE_ENGINES
            for engine in r.get("engines", CORE_ENGINES)
        )
        for r in core
    )
    ordinal = 0

    for record in core:
        years = [int(y) for y in record.get("years", EXPECTED_YEARS)]
        engines = [
            str(e).lower()
            for e in record.get("engines", CORE_ENGINES)
            if str(e).lower() in IMPORTANCE_ENGINES
        ]
        for engine in engines:
            for year in years:
                ordinal += 1
                key = (str(record["id"]), engine, int(year))
                if key in completed:
                    LOG.info(
                        "Hunting all-fold importance [%d/%d]: %s / %s / test %d already complete; skipping",
                        ordinal, total, record["id"], engine, year,
                    )
                    continue
                LOG.info(
                    "Hunting all-fold importance [%d/%d]: %s / %s / test %d",
                    ordinal, total, record["id"], engine, year,
                )
                row = compute_fold_hunting_importance(
                    record,
                    year=year,
                    engine=engine,
                    n_repeats=n_repeats,
                    random_state=random_state,
                    predict_threads=predict_threads,
                )
                rows.append(row)
                completed.add(key)
                # Durable checkpoint after every successful fold.
                pd.DataFrame(rows).to_csv(checkpoint, index=False)

    fold_df = pd.DataFrame(rows)
    fold_df.to_csv(
        output_dir / "hunting_permutation_importance_all_folds.csv",
        index=False,
    )
    safe_unlink(output_dir / "hunting_permutation_importance_all_folds.partial.csv")

    from scipy.stats import t as student_t

    summary_rows: list[dict[str, Any]] = []
    group_cols = [
        "scenario",
        "scenario_role",
        "dataset",
        "temporal",
        "engine",
        "variable",
    ]
    for keys, group in fold_df.groupby(group_cols, sort=False, dropna=False):
        vals = pd.to_numeric(
            group["importance_mean_auc_drop"], errors="coerce"
        ).dropna()
        n = len(vals)
        mean = float(vals.mean()) if n else math.nan
        sd = float(vals.std(ddof=1)) if n > 1 else math.nan
        if n > 1:
            crit = float(student_t.ppf(0.975, n - 1))
            half = crit * sd / math.sqrt(n)
        else:
            crit = half = math.nan
        row = dict(zip(group_cols, keys))
        row.update(
            {
                "n_folds": n,
                "mean_auc_drop": mean,
                "sd_across_folds": sd,
                "ci95_low": mean - half if n > 1 else math.nan,
                "ci95_high": mean + half if n > 1 else math.nan,
                "ci_method": "Student-t across LOYO-fold mean importances",
                "min_fold_mean_auc_drop": float(vals.min()) if n else math.nan,
                "max_fold_mean_auc_drop": float(vals.max()) if n else math.nan,
                "positive_importance_folds": int((vals > 0).sum()) if n else 0,
            }
        )
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(
        output_dir / "hunting_permutation_importance_summary.csv",
        index=False,
    )
    write_table_bundle(
        summary_df,
        output_dir / "hunting_permutation_importance_summary",
    )
    return fold_df, summary_df


def plot_hunting_importance(
    fold_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    output_dir: Path,
    dpi: int = 300,
) -> list[Path]:
    if fold_df.empty:
        return []
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    order = []
    groups = []
    for dataset in CORE_DATASETS:
        for engine in CORE_ENGINES:
            for temporal in CORE_TEMPORAL:
                sub = fold_df[
                    (fold_df["dataset"] == dataset)
                    & (fold_df["engine"] == engine)
                    & (fold_df["temporal"] == temporal)
                ]
                vals = pd.to_numeric(
                    sub["importance_mean_auc_drop"], errors="coerce"
                ).dropna().to_numpy()
                if vals.size:
                    order.append(
                        f"{dataset}\n{engine}\n"
                        + ("mono" if temporal == "monotemporal" else "multi")
                    )
                    groups.append(vals)

    if not groups:
        return []

    fig, ax = plt.subplots(figsize=(12, 6.5))
    try:
        ax.boxplot(groups, tick_labels=order, showmeans=True)
    except TypeError:
        ax.boxplot(groups, labels=order, showmeans=True)
    ax.axhline(0.0, linewidth=1)
    ax.set_ylabel("Hunting-bag permutation importance (AUC drop)")
    ax.set_title("Hunting-bag importance across all winter LOYO folds")
    ax.tick_params(axis="x", labelrotation=25)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    png = output_dir / "hunting_bag_all_LOYO_folds_permutation_importance.png"
    pdf = output_dir / "hunting_bag_all_LOYO_folds_permutation_importance.pdf"
    fig.savefig(png, dpi=dpi, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return [png, pdf]


# ---------------------------------------------------------------------------
# Configuration audit cleanup
# ---------------------------------------------------------------------------

def find_source_configuration_audit(source_run: Path) -> Path:
    preferred = (
        source_run
        / "04_SUPPLEMENT"
        / "01_Supplement_S5_corrected_configuration"
        / "matched_configuration_difference_audit.csv"
    )
    if preferred.exists():
        return preferred
    candidates = list(source_run.rglob("matched_configuration_difference_audit.csv"))
    if not candidates:
        raise FileNotFoundError("matched_configuration_difference_audit.csv missing")
    return candidates[0]


def finalize_configuration_audit(
    source_run: Path,
    final_dir: Path,
) -> pd.DataFrame:
    source = find_source_configuration_audit(source_run)
    frame = pd.read_csv(source)
    frame["source_classification"] = frame["classification"].astype(str)
    frame["final_classification"] = frame["classification"].astype(str)
    frame["final_note"] = ""

    output_mask = frame["path"].astype(str).str.startswith("outputs.model_paths")
    frame.loc[output_mask, "final_classification"] = "runtime_output_difference"
    frame.loc[output_mask, "final_note"] = (
        "Scenario-specific output filenames only; no scientific predictor/model-setting difference."
    )

    dem_mask = frame["path"].astype(str).eq("predictors.layers.dem")
    frame.loc[dem_mask, "final_classification"] = (
        "preexisting_preserved_configuration_difference"
    )
    frame.loc[dem_mask, "final_note"] = (
        "DEM Aspect is present only in the multitemporal source configuration. "
        "This difference predates the hunting correction, is not inherent to temporal "
        "method, and was deliberately preserved to avoid changing scope without a model rerun."
    )

    # Any remaining unresolved review_required entry is a true blocker.
    remaining = (
        (frame["classification"].astype(str) == "review_required")
        & ~output_mask
    )
    frame.loc[remaining, "final_classification"] = "review_required"

    # Present the final classification as the primary classification while
    # retaining the source export's original label for provenance.
    frame["classification"] = frame["final_classification"]

    dest = (
        final_dir
        / "04_SUPPLEMENT"
        / "01_Supplement_S5_corrected_configuration"
        / "matched_configuration_difference_audit.csv"
    )
    write_table_bundle(
        frame,
        dest.with_suffix(""),
    )

    diag = (
        final_dir
        / "02_DIAGNOSTICS"
        / "configuration_audit"
        / "matched_configuration_difference_audit_FINAL"
    )
    write_table_bundle(frame, diag)

    statement = """# Final configuration-scope statement

The hunting correction changes only the reviewer-identified winter-monotemporal
hunting setting and regenerates the matched winter reference runs.

Remaining mono/multitemporal differences are classified explicitly. Scenario
output filenames are runtime-only differences. The DEM predictor list has one
pre-existing source-configuration difference: DEM Aspect is present in the
multitemporal source configuration but absent from the monotemporal source
configuration. This is **not** described as an inherent temporal-method
difference. It is preserved unchanged here because harmonizing it would require
a new scientific model rerun outside the scope of the hunting correction.

No unresolved `review_required` difference should remain in the finalized audit.
"""
    (diag.parent / "configuration_scope_statement.md").write_text(
        statement, encoding="utf-8"
    )
    return frame


# ---------------------------------------------------------------------------
# Manuscript replacement register
# ---------------------------------------------------------------------------

def locate_publication_index(source_run: Path) -> Path | None:
    candidates = [
        source_run
        / "01_INPUT"
        / "02_publication_index"
        / "current_publication_index.csv",
        REPO_ROOT / "review" / "input" / "current_publication_index.csv",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def group_mean(
    test: pd.DataFrame,
    *,
    dataset: str,
    temporal: str,
    engine: str,
    metric: str = "cbi_spearman",
) -> float:
    sub = test[
        (test["dataset"] == dataset)
        & (test["temporal"] == temporal)
        & (test["engine"].astype(str).str.casefold() == engine)
    ]
    return float(pd.to_numeric(sub[metric], errors="coerce").mean())


def build_manuscript_attention_register(
    metrics: pd.DataFrame,
    source_run: Path,
    final_dir: Path,
) -> pd.DataFrame:
    test = metrics[metrics["split"].astype(str).str.casefold() == "test"].copy()
    train = metrics[metrics["split"].astype(str).str.casefold() == "train"].copy()
    test["cbi_spearman"] = pd.to_numeric(test["cbi_spearman"], errors="coerce")
    test["auc"] = pd.to_numeric(test["auc"], errors="coerce")
    train["auc"] = pd.to_numeric(train["auc"], errors="coerce")

    maxent_means = []
    gbm_means = []
    for dataset in CORE_DATASETS:
        for temporal in CORE_TEMPORAL:
            maxent_means.append(
                group_mean(
                    test,
                    dataset=dataset,
                    temporal=temporal,
                    engine="elapid",
                )
            )
            gbm_means.append(
                group_mean(
                    test,
                    dataset=dataset,
                    temporal=temporal,
                    engine="gbm",
                )
            )

    mono_all_gbm = test[
        (test["dataset"] == "allpoints")
        & (test["temporal"] == "monotemporal")
        & (test["engine"].astype(str).str.casefold() == "gbm")
    ]
    min_idx = mono_all_gbm["cbi_spearman"].idxmin()
    min_cbi = float(mono_all_gbm.loc[min_idx, "cbi_spearman"])
    min_year = int(mono_all_gbm.loc[min_idx, "test_year"])

    gbm_multi_all = group_mean(
        test, dataset="allpoints", temporal="multitemporal", engine="gbm"
    )
    gbm_multi_gbif = group_mean(
        test, dataset="gbif", temporal="multitemporal", engine="gbm"
    )
    maxent_multi_all = group_mean(
        test, dataset="allpoints", temporal="multitemporal", engine="elapid"
    )
    maxent_multi_gbif = group_mean(
        test, dataset="gbif", temporal="multitemporal", engine="elapid"
    )

    maxent_negative = int(
        (
            test.loc[
                test["engine"].astype(str).str.casefold() == "elapid",
                "cbi_spearman",
            ]
            < 0
        ).sum()
    )

    gbm_train = train[
        train["engine"].astype(str).str.casefold() == "gbm"
    ]["auc"].dropna()

    recommended_1 = (
        "Across the regenerated winter LOYO folds, MaxEnt remained consistently "
        "high and substantially more stable than GBM. Mean winter CBI (Spearman) "
        f"ranged {min(maxent_means):.3f}–{max(maxent_means):.3f} for MaxEnt and "
        f"{min(gbm_means):.3f}–{max(gbm_means):.3f} for GBM. The corrected "
        "winter monotemporal all-points GBM still contained one negative fold, "
        f"but the minimum was {min_cbi:.3f} (test year {min_year}), not approximately "
        "−0.35. MaxEnt produced "
        f"{maxent_negative} negative winter test folds. Combine these corrected "
        "winter values with the unchanged summer results when revising the full "
        "cross-season paragraph."
    )

    recommended_2 = (
        "The data-source effect was small for winter MaxEnt but configuration-"
        "dependent for GBM. For multitemporal MaxEnt, mean winter CBI changed only "
        f"from {maxent_multi_all:.3f} (GBIF + WVC) to {maxent_multi_gbif:.3f} "
        f"(GBIF-only), whereas multitemporal GBM changed from {gbm_multi_all:.3f} "
        f"to {gbm_multi_gbif:.3f}. Therefore the existing statements that GBM's "
        "best mean CBI was only about 0.75 in monotemporal summer and that data "
        "input had minimal impact are no longer defensible without qualification."
    )

    recommended_3 = (
        "For the regenerated winter GBM runs, training AUC ranged "
        f"{float(gbm_train.min()):.3f}–{float(gbm_train.max()):.3f}. "
        "Any statement that GBM 'frequently' exceeded 0.9 in training should be "
        "checked against the unchanged summer runs rather than generalized from "
        "the corrected winter models."
    )

    recommended_4 = (
        "The current predictor-method wording should not imply that elevation, "
        "slope and aspect were used identically in every temporal mode. The "
        "resolved source configurations show elevation+slope for monotemporal "
        "models and elevation+slope+aspect for multitemporal models. This "
        "pre-existing difference was preserved in the hunting-only correction "
        "because harmonizing it would require a separate model rerun."
    )

    patterns = [
        ("Across all LOYO folds, MaxEnt models exhibited", recommended_1,
         "Winter numeric claims changed after hunting correction."),
        ("Data input (WVCs+GBIF vs. GBIF_only) had minimal impact", recommended_2,
         "GBM dataset sensitivity is now large in winter multitemporal models."),
        ("GBM frequently reached training AUCs", recommended_3,
         "Winter regenerated training AUC distribution should qualify this claim."),
        ("Terrain variables (elevation, slope, aspect)", recommended_4,
         "Resolved mono/multi configurations do not use Aspect identically."),
    ]

    publication_index = locate_publication_index(source_run)
    index = (
        pd.read_csv(publication_index)
        if publication_index is not None
        else pd.DataFrame(columns=["source", "page", "text"])
    )

    rows: list[dict[str, Any]] = []
    for pattern, recommendation, issue in patterns:
        matches = index[
            index.get("text", pd.Series(dtype=str))
            .astype(str)
            .str.contains(pattern, case=False, regex=False, na=False)
        ]
        if matches.empty:
            rows.append(
                {
                    "source": "manuscript",
                    "page": None,
                    "matched": False,
                    "old_text": None,
                    "issue": issue,
                    "recommended_revision_or_action": recommendation,
                }
            )
        else:
            for _, m in matches.iterrows():
                rows.append(
                    {
                        "source": m.get("source", "manuscript"),
                        "page": m.get("page"),
                        "matched": True,
                        "old_text": m.get("text"),
                        "issue": issue,
                        "recommended_revision_or_action": recommendation,
                    }
                )

    frame = pd.DataFrame(rows)
    dest = (
        final_dir
        / "03_MANUSCRIPT"
        / "05_Manuscript_numeric_replacement_register"
        / "affected_manuscript_text"
    )
    write_table_bundle(frame, dest)

    # Plain-language compact note for immediate manuscript editing.
    note_lines = [
        "# Manuscript text requiring attention after the winter hunting correction",
        "",
        "This register is intentionally conservative: the new run contains corrected "
        "winter results only. Summer results were not rerun and must not be silently "
        "replaced by winter-only values.",
        "",
    ]
    for i, row in frame.iterrows():
        note_lines += [
            f"## {i + 1}. {row['issue']}",
            "",
            str(row["recommended_revision_or_action"]),
            "",
        ]
    (dest.parent / "affected_manuscript_text_GUIDANCE.md").write_text(
        "\n".join(note_lines), encoding="utf-8"
    )
    return frame


# ---------------------------------------------------------------------------
# Final validation
# ---------------------------------------------------------------------------

def boolish(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return (
        series.astype(str)
        .str.strip()
        .str.casefold()
        .isin({"true", "1", "yes", "y"})
    )


def source_s7_tables(source_run: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    base = (
        source_run
        / "04_SUPPLEMENT"
        / "03_Supplement_S7_feature_provenance_and_hunting_audit"
    )
    selection = base / "S7_hunting_candidate_and_fold_selection_audit.csv"
    sources = base / "S7_hunting_source_raster_inventory.csv"
    if not selection.exists() or not sources.exists():
        raise FileNotFoundError("Source S7 hunting audit tables are missing.")
    return pd.read_csv(selection), pd.read_csv(sources)


def validate_final(
    source_run: Path,
    final_dir: Path,
    registry: list[dict[str, Any]],
    metrics: pd.DataFrame,
    config_audit: pd.DataFrame,
    hunting_fold_df: pd.DataFrame | None,
    *,
    importance_skipped: bool,
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    def check(name: str, passed: bool, detail: Any) -> None:
        checks.append(
            {"check": name, "pass": bool(passed), "detail": str(detail)}
        )

    core = [r for r in registry if str(r.get("role", "")).lower() in CORE_ROLES]
    check("core_scenario_count", len(core) == 4, f"{len(core)}/4")

    expected_pairs = {
        ("allpoints", "monotemporal"),
        ("allpoints", "multitemporal"),
        ("gbif", "monotemporal"),
        ("gbif", "multitemporal"),
    }
    found_pairs = {(r.get("dataset"), r.get("temporal")) for r in core}
    check("core_scenario_matrix", found_pairs == expected_pairs, found_pairs)

    for record in core:
        complete = Path(record["scenario_dir"]) / "COMPLETE"
        check(
            f"source_scenario_complete:{record['id']}",
            complete.exists(),
            complete,
        )

    # Metric coverage: 4 scenarios x 2 engines x 7 years x 3 splits.
    for record in core:
        for engine in CORE_ENGINES:
            sub = metrics[
                (metrics["scenario"] == record["id"])
                & (metrics["engine"].astype(str).str.casefold() == engine)
            ]
            for split in EXPECTED_SPLITS:
                split_sub = sub[sub["split"].astype(str).str.casefold() == split]
                years = set(
                    pd.to_numeric(
                        split_sub["test_year"], errors="coerce"
                    ).dropna().astype(int)
                )
                check(
                    f"metrics:{record['id']}:{engine}:{split}",
                    years == set(EXPECTED_YEARS),
                    f"years={sorted(years)}",
                )

    selection, sources = source_s7_tables(source_run)
    selected_true = boolish(selection["hunting_present_in_materialized_test_stack"])
    retained = (
        pd.to_numeric(selection["hunting_selected_count"], errors="coerce")
        .fillna(0)
        .astype(int)
        > 0
    )
    check(
        "hunting_fold_audit_rows",
        len(selection) == 28,
        f"{len(selection)}/28",
    )
    check(
        "hunting_materialized_all_folds",
        bool(selected_true.all()),
        f"{int(selected_true.sum())}/{len(selected_true)}",
    )
    check(
        "hunting_retained_all_folds",
        bool(retained.all()),
        f"{int(retained.sum())}/{len(retained)}",
    )

    source_exists = boolish(sources["exists"])
    check(
        "hunting_source_inventory_rows",
        len(sources) == 28,
        f"{len(sources)}/28 scenario-years",
    )
    check(
        "hunting_source_rasters_exist",
        bool(source_exists.all()),
        f"{int(source_exists.sum())}/{len(source_exists)}",
    )

    map_index = (
        final_dir
        / "04_SUPPLEMENT"
        / "06_Supplement_full_corrected_winter_maps"
        / "full_corrected_map_index.csv"
    )
    if map_index.exists():
        maps = pd.read_csv(map_index)
        nonempty_source = maps["source"].notna() & maps["source"].astype(str).ne("")
        check("map_index_rows", len(maps) == 56, f"{len(maps)}/56")
        check(
            "map_index_sources_nonempty",
            bool(nonempty_source.all()),
            f"{int(nonempty_source.sum())}/{len(nonempty_source)}",
        )
    else:
        check("map_index_rows", False, f"missing: {map_index}")

    unresolved = config_audit[
        config_audit["final_classification"].astype(str) == "review_required"
    ]
    check(
        "configuration_audit_no_unresolved",
        unresolved.empty,
        f"unresolved rows={len(unresolved)}",
    )

    dem = config_audit[
        config_audit["path"].astype(str) == "predictors.layers.dem"
    ]
    check(
        "dem_difference_not_called_temporal_inherent",
        (
            not dem.empty
            and (
                dem["final_classification"].astype(str)
                == "preexisting_preserved_configuration_difference"
            ).all()
        ),
        dem["final_classification"].tolist() if not dem.empty else "missing",
    )

    if importance_skipped:
        check(
            "hunting_all_fold_importance",
            True,
            "explicitly skipped by CLI; package marked accordingly",
        )
    else:
        if hunting_fold_df is None:
            check("hunting_all_fold_importance", False, "no result table")
        else:
            coverage = (
                hunting_fold_df.groupby(
                    ["scenario", "engine"], dropna=False
                )["test_year"]
                .nunique()
            )
            expected = {
                (str(record["id"]), engine)
                for record in core
                for engine in IMPORTANCE_ENGINES
            }
            passed = set(coverage.index) == expected and (coverage == 7).all()
            check(
                "hunting_all_fold_importance",
                passed,
                coverage.to_dict(),
            )

    # New final output must not contain stale state files.
    stale_found = [
        str(p.relative_to(final_dir))
        for p in final_dir.rglob("*")
        if p.is_file() and p.name in STALE_NAMES
    ]
    check("no_stale_failure_or_lock", not stale_found, stale_found)

    # Required publication directories.
    required_dirs = [
        "03_MANUSCRIPT/01_Table_2_LOYO_model_performance",
        "03_MANUSCRIPT/05_Manuscript_numeric_replacement_register",
        "04_SUPPLEMENT/01_Supplement_S5_corrected_configuration",
        "04_SUPPLEMENT/02_Supplement_S6_foldwise_performance_and_CI",
        "04_SUPPLEMENT/03_Supplement_S7_feature_provenance_and_hunting_audit",
        "04_SUPPLEMENT/06_Supplement_full_corrected_winter_maps",
    ]
    for rel in required_dirs:
        path = final_dir / rel
        check(
            f"required_publication_tree:{rel}",
            path.exists() and any(p.is_file() for p in path.rglob("*")),
            path,
        )

    passed = all(c["pass"] for c in checks)
    result = {
        "status": "pass" if passed else "fail",
        "source_run": str(source_run),
        "final_dir": str(final_dir),
        "checks": checks,
        "issues": [c for c in checks if not c["pass"]],
    }
    write_json(result, final_dir / "00_RUN" / "validation_report.json")
    pd.DataFrame(checks).to_csv(
        final_dir / "00_RUN" / "validation_checks.csv",
        index=False,
    )
    pd.DataFrame(result["issues"]).to_csv(
        final_dir / "00_RUN" / "validation_issues.csv",
        index=False,
    )
    return result


# ---------------------------------------------------------------------------
# Repo hygiene report
# ---------------------------------------------------------------------------

def repo_hygiene_report(final_dir: Path) -> pd.DataFrame:
    info = git_info()
    rows: list[dict[str, Any]] = []
    for name in KNOWN_REPO_HYGIENE_FILES:
        path = REPO_ROOT / name
        tracked = False
        try:
            proc = subprocess.run(
                ["git", "ls-files", "--error-unmatch", name],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
                check=False,
            )
            tracked = proc.returncode == 0
        except Exception:
            pass
        rows.append(
            {
                "path": name,
                "exists": path.exists(),
                "tracked": tracked,
                "recommendation": (
                    "remove from source branch before merge"
                    if path.exists() and tracked
                    else "none"
                ),
            }
        )

    # Check whether self-test discovers the top-level regression tests.
    rows.append(
        {
            "path": "tests/test_review_feature_importance.py",
            "exists": (REPO_ROOT / "tests/test_review_feature_importance.py").exists(),
            "tracked": None,
            "recommendation": (
                "review-corrections.py self-test discovers review/tests only; "
                "move/include this regression test in the self-test suite"
            ),
        }
    )
    rows.append(
        {
            "path": "tests/test_review_novelty.py",
            "exists": (REPO_ROOT / "tests/test_review_novelty.py").exists(),
            "tracked": None,
            "recommendation": (
                "review-corrections.py self-test discovers review/tests only; "
                "move/include this regression test in the self-test suite"
            ),
        }
    )

    frame = pd.DataFrame(rows)
    frame.to_csv(final_dir / "00_RUN" / "repo_hygiene_findings.csv", index=False)
    write_json(info, final_dir / "00_RUN" / "git_state_at_finalization.json")
    return frame


# ---------------------------------------------------------------------------
# Final publication assembly
# ---------------------------------------------------------------------------

def copy_publication_layer(source_run: Path, final_dir: Path) -> None:
    """Copy/hardlink only publication/source layers, never 02_MODELS wholesale."""
    for rel in ("03_MANUSCRIPT", "04_SUPPLEMENT", "06_MACHINE_READABLE"):
        LOG.info("Reusing publication layer: %s", rel)
        copy_tree_lightweight(
            source_run / rel,
            final_dir / rel,
            excluded_names=STALE_NAMES,
        )

    # Preserve source input/provenance snapshots without copying run-state junk.
    source_input = source_run / "01_INPUT"
    if source_input.exists():
        copy_tree_lightweight(
            source_input,
            final_dir / "01_SOURCE" / "source_input_snapshot",
            excluded_names=STALE_NAMES,
        )

    selected_run_files = (
        "scenario_registry.json",
        "scenario_audit_summary.json",
        "scenario_audit_summary.csv",
        "validation_report.json",
        "disabled_hunting_config_inventory.csv",
        "02_provenance.json",
        "03_environment_pip_freeze.txt",
        "04_input_config_inventory.csv",
    )
    for name in selected_run_files:
        src = source_run / "00_RUN" / name
        if src.exists():
            hardlink_or_copy(
                src,
                final_dir / "01_SOURCE" / "source_run_metadata" / name,
            )


def replace_metric_exports(
    metrics: pd.DataFrame,
    summary: pd.DataFrame,
    final_dir: Path,
) -> None:
    # Machine-readable final tables.
    metrics.to_csv(
        final_dir / "06_MACHINE_READABLE" / "all_corrected_temporal_cv_metrics.csv",
        index=False,
    )
    summary.to_csv(
        final_dir / "06_MACHINE_READABLE" / "all_corrected_temporal_cv_summary.csv",
        index=False,
    )

    # Main table, replacing the previous normal-approximation CI table.
    main_stem = (
        final_dir
        / "03_MANUSCRIPT"
        / "01_Table_2_LOYO_model_performance"
        / "Table_2_corrected_winter_LOYO_performance"
    )
    write_table_bundle(summary, main_stem)

    # S6 summary and foldwise tables.
    s6 = (
        final_dir
        / "04_SUPPLEMENT"
        / "02_Supplement_S6_foldwise_performance_and_CI"
    )
    write_table_bundle(
        summary,
        s6 / "S6_corrected_LOYO_summary_and_95CI",
    )
    write_table_bundle(
        metrics.sort_values(
            ["dataset", "temporal", "engine", "test_year", "split"],
            kind="stable",
        ),
        s6 / "S6_corrected_foldwise_LOYO_metrics",
    )

    # Numeric register remains a machine-friendly exact-value table, now with
    # Student-t CI and explicit scenario role.
    numeric = summary.copy()
    numeric["recommended_rounding"] = (
        "three decimals unless manuscript specifies otherwise"
    )
    write_table_bundle(
        numeric,
        final_dir
        / "03_MANUSCRIPT"
        / "05_Manuscript_numeric_replacement_register"
        / "manuscript_numeric_replacements",
    )


def attach_hunting_importance_to_publication(
    fold_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    final_dir: Path,
) -> None:
    main = (
        final_dir
        / "03_MANUSCRIPT"
        / "04_Figure_winter_feature_importance"
    )
    supp = (
        final_dir
        / "04_SUPPLEMENT"
        / "03_Supplement_S7_feature_provenance_and_hunting_audit"
    )
    machine = final_dir / "06_MACHINE_READABLE"

    write_table_bundle(
        fold_df,
        main / "hunting_bag_all_LOYO_folds_permutation_importance",
    )
    write_table_bundle(
        summary_df,
        main / "hunting_bag_all_LOYO_folds_permutation_importance_summary",
    )
    plot_hunting_importance(fold_df, summary_df, main, dpi=300)

    write_table_bundle(
        fold_df,
        supp / "S7_hunting_bag_all_LOYO_folds_permutation_importance",
    )
    write_table_bundle(
        summary_df,
        supp / "S7_hunting_bag_all_LOYO_folds_permutation_importance_summary",
    )
    fold_df.to_csv(
        machine / "hunting_bag_all_LOYO_folds_permutation_importance.csv",
        index=False,
    )
    summary_df.to_csv(
        machine / "hunting_bag_all_LOYO_folds_permutation_importance_summary.csv",
        index=False,
    )

    note = """# Feature-importance interpretation for the hunting correction

The pre-existing full feature-importance exports are retained as descriptive
best-fold plots. They should not be used as the sole evidence for the hunting
correction because the selected fold differs among engines/scenarios.

The files prefixed `hunting_bag_all_LOYO_folds_` are the correction-specific
evidence. They recompute **only the hunting-bag permutation importance for
MaxEnt** from the already fitted saved model in every held-out-year fold. No
model is refitted. GBM is intentionally excluded because the manuscript limits
permutation feature-importance inference to the consistently better-CBI MaxEnt
models.

Report the across-fold distribution/mean (and its uncertainty) rather than the
single best-fold value when discussing whether the corrected hunting predictor
contributed to model discrimination.
"""
    (main / "FEATURE_IMPORTANCE_INTERPRETATION.md").write_text(
        note, encoding="utf-8"
    )


def write_final_readme(
    source_run: Path,
    final_dir: Path,
    *,
    importance_skipped: bool,
) -> None:
    text = f"""# Finalized winter-monotemporal hunting correction

Source completed model run:

`{source_run}`

Finalized publication layer:

`{final_dir}`

## What this postprocessor changed

- **No models were retrained or rebuilt.**
- Publication metrics include only `corrected` and `reference` scenario roles.
  Optional no-hunting controls are excluded from the primary summaries.
- 95% intervals are recomputed with Student's *t* across the seven LOYO folds.
- Stale `FAILURE.json` and `RUNNING.lock` files are not propagated.
- The configuration-difference audit treats model output filenames as runtime
  differences and identifies the DEM Aspect mismatch as a pre-existing
  preserved configuration difference, not an inherent temporal-method difference.
- An explicit manuscript-text attention/replacement register is generated.
- Final validation checks scenario, engine, split and year coverage, hunting
  retention/source rasters, map inventory and configuration-audit status.
- Hunting-specific MaxEnt all-fold permutation importance was
  **{"skipped by explicit CLI request" if importance_skipped else "computed from saved fitted models without refitting"}**.

## Feature importance

The older full-feature best-fold outputs are retained as descriptive material.
Correction-specific inference should use the new all-fold hunting-bag importance
tables/figure because they do not select an optimistic held-out year. This
diagnostic is MaxEnt-only, matching the manuscript's stated feature-importance
scope; it does not load or evaluate saved GBM models.

## DEM Aspect

The source configurations differ in the DEM layer set: monotemporal uses
elevation+slope, while multitemporal also contains aspect. This difference
predates the hunting correction. It is preserved here because changing it would
require a separate scientific model rerun. The final audit therefore flags it
transparently rather than misclassifying it as inherent to temporal design.

## Archive state

The final publication ZIP contains only successful finalization state. Earlier
failed/resumed attempts remain part of the source-run history on disk but are not
included as a misleading `FAILURE.json` in this finalized package.
"""
    (final_dir / "README_FINALIZED_CORRECTION.md").write_text(
        text, encoding="utf-8"
    )


def create_clean_archive(
    final_dir: Path,
    *,
    max_member_mb: int,
) -> Path:
    archive = (
        final_dir
        / "07_ARCHIVE"
        / f"{final_dir.name}_publication_bundle_FINAL.zip"
    )
    max_bytes = max_member_mb * 1024 * 1024
    include_roots = [
        final_dir / "00_RUN",
        final_dir / "01_SOURCE",
        final_dir / "02_DIAGNOSTICS",
        final_dir / "03_MANUSCRIPT",
        final_dir / "04_SUPPLEMENT",
        final_dir / "06_MACHINE_READABLE",
        final_dir / "README_FINALIZED_CORRECTION.md",
    ]
    skipped: list[dict[str, Any]] = []

    with zipfile.ZipFile(
        archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6
    ) as zf:
        for root in include_roots:
            paths = [root] if root.is_file() else sorted(root.rglob("*"))
            for path in paths:
                if not path.is_file():
                    continue
                if path.name in STALE_NAMES:
                    continue
                if path == archive:
                    continue
                size = path.stat().st_size
                if size > max_bytes:
                    skipped.append(
                        {
                            "path": str(path.relative_to(final_dir)),
                            "size_bytes": size,
                            "reason": (
                                f"exceeds archive max member size "
                                f"({max_member_mb} MB)"
                            ),
                        }
                    )
                    continue
                zf.write(path, arcname=str(path.relative_to(final_dir)))

    write_csv_rows(
        skipped,
        final_dir / "07_ARCHIVE" / "large_files_excluded_from_zip.csv",
    )
    return archive


def archive_integrity_checks(archive: Path) -> dict[str, Any]:
    with zipfile.ZipFile(archive) as zf:
        names = zf.namelist()
        bad = zf.testzip()
    stale = [
        n for n in names
        if Path(n).name in STALE_NAMES
    ]
    return {
        "zip_test": "pass" if bad is None else f"bad member: {bad}",
        "member_count": len(names),
        "stale_state_members": stale,
        "pass": bad is None and not stale,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Finalize an existing hunting-correction run without model reruns."
        )
    )
    p.add_argument(
        "--source-run",
        type=Path,
        required=True,
        help="Existing completed review/output/<run-id> directory.",
    )
    p.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Destination root. Defaults to the source run's parent folder.",
    )
    p.add_argument("--run-id", default=None)
    p.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {SCRIPT_VERSION}",
    )
    p.add_argument("--n-repeats", type=int, default=10)
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument(
        "--predict-threads",
        type=int,
        default=1,
        help="Threads used inside each saved-model prediction during permutation.",
    )
    p.add_argument(
        "--skip-hunting-importance",
        action="store_true",
        help=(
            "Skip the all-fold hunting permutation diagnostic. "
            "No model training occurs either way."
        ),
    )
    p.add_argument(
        "--archive-max-member-mb",
        type=int,
        default=100,
    )
    p.add_argument("--log-level", default="INFO")
    p.add_argument(
        "--resume-output",
        action="store_true",
        help=(
            "Continue an existing finalization output folder and reuse its "
            "per-fold permutation checkpoint. Must use the same --run-id."
        ),
    )
    p.add_argument("--force", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    require_repo_root()
    source_run = resolve_source_run(args.source_run)
    final_dir = prepare_final_dir(
        source_run,
        args.output_root,
        args.run_id,
        args.force,
        resume_existing=args.resume_output,
    )
    configure_logging(
        final_dir / "00_RUN" / "finalization_console.log",
        args.log_level,
    )

    try:
        LOG.info("Repository: %s", REPO_ROOT)
        LOG.info("Source completed run: %s", source_run)
        LOG.info("Final output: %s", final_dir)
        LOG.info(
            "NO MODEL RERUN: this command only reads saved models/results/stacks."
        )

        registry = load_json(source_run / "00_RUN" / "scenario_registry.json")
        core_metrics, core_registry = collect_core_metrics(registry)

        source_failure = source_run / "00_RUN" / "FAILURE.json"
        source_lock = source_run / "00_RUN" / "RUNNING.lock"
        source_meta = {
            "source_run": str(source_run),
            "source_failure_json_existed": source_failure.exists(),
            "source_running_lock_existed_at_finalization": source_lock.exists(),
            "source_registry_records": len(registry),
            "core_corrected_reference_records": len(core_registry),
            "finalization_started": datetime.now(ZoneInfo("Europe/Zurich")).isoformat(),
            "git": git_info(),
        }
        write_json(source_meta, final_dir / "00_RUN" / "source_run.json")

        LOG.info("Reusing existing publication artifacts with hardlinks/copies")
        copy_publication_layer(source_run, final_dir)

        LOG.info("Recomputing LOYO summaries with Student-t 95%% confidence intervals")
        summary = student_t_summary(core_metrics)
        replace_metric_exports(core_metrics, summary, final_dir)

        LOG.info("Finalizing matched configuration-difference audit")
        config_audit = finalize_configuration_audit(source_run, final_dir)

        LOG.info("Building explicit manuscript text replacement/attention register")
        build_manuscript_attention_register(
            core_metrics, source_run, final_dir
        )

        hunting_fold_df: pd.DataFrame | None = None
        hunting_summary_df: pd.DataFrame | None = None
        if args.skip_hunting_importance:
            LOG.warning(
                "All-fold hunting permutation importance explicitly skipped."
            )
            skip_note = (
                final_dir
                / "02_DIAGNOSTICS"
                / "hunting_all_fold_importance"
                / "SKIPPED.txt"
            )
            skip_note.write_text(
                "Skipped by --skip-hunting-importance. No model rerun performed.\n",
                encoding="utf-8",
            )
        else:
            LOG.info(
                "Preflighting fitted predictor metadata across saved MaxEnt LOYO models"
            )
            preflight_saved_model_predictors(
                registry,
                final_dir / "02_DIAGNOSTICS" / "hunting_all_fold_importance",
            )
            LOG.info(
                "Computing hunting-only permutation importance across saved MaxEnt LOYO models"
            )
            hunting_fold_df, hunting_summary_df = compute_allfold_hunting_importance(
                registry,
                final_dir / "02_DIAGNOSTICS" / "hunting_all_fold_importance",
                n_repeats=max(1, args.n_repeats),
                random_state=args.random_state,
                predict_threads=max(1, args.predict_threads),
            )
            attach_hunting_importance_to_publication(
                hunting_fold_df,
                hunting_summary_df,
                final_dir,
            )

        repo_hygiene_report(final_dir)
        write_final_readme(
            source_run,
            final_dir,
            importance_skipped=args.skip_hunting_importance,
        )

        LOG.info("Running strict final validation")
        validation = validate_final(
            source_run,
            final_dir,
            registry,
            core_metrics,
            config_audit,
            hunting_fold_df,
            importance_skipped=args.skip_hunting_importance,
        )
        if validation["status"] != "pass":
            write_json(
                {
                    "status": "failed",
                    "reason": "strict final validation failed",
                    "issues": validation["issues"],
                },
                final_dir / "00_RUN" / "FINALIZATION_FAILED.json",
            )
            LOG.error(
                "Strict final validation failed; no success archive will be declared."
            )
            return 2

        # Ensure stale source state can never be mistaken for final state.
        for stale in STALE_NAMES:
            for path in final_dir.rglob(stale):
                safe_unlink(path)

        safe_unlink(final_dir / "00_RUN" / "FINALIZATION_FAILED.json")

        success = {
            "status": "success",
            "source_run": str(source_run),
            "final_dir": str(final_dir),
            "completed": datetime.now(ZoneInfo("Europe/Zurich")).isoformat(),
            "model_rerun": False,
            "hunting_all_fold_importance": not args.skip_hunting_importance,
            "ci_method": "Student-t 95% CI across seven LOYO folds",
            "validation_status": validation["status"],
        }
        write_json(success, final_dir / "00_RUN" / "FINALIZATION_SUCCESS.json")
        (final_dir / "00_RUN" / "FINALIZATION_COMPLETE").write_text(
            success["completed"] + "\n", encoding="utf-8"
        )

        # Build a manifest before the ZIP so the clean publication archive
        # contains its own file inventory. The archive itself is intentionally
        # excluded to avoid a circular checksum.
        manifest_path = final_dir / "00_RUN" / "FINAL_MANIFEST.csv"
        manifest_rows: list[dict[str, Any]] = []
        for path in sorted(final_dir.rglob("*")):
            if not path.is_file():
                continue
            if path == manifest_path:
                continue
            if "07_ARCHIVE" in path.parts:
                continue
            manifest_rows.append(
                {
                    "path": str(path.relative_to(final_dir)),
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
        write_csv_rows(manifest_rows, manifest_path)

        LOG.info("Creating clean final publication archive")
        archive = create_clean_archive(
            final_dir,
            max_member_mb=max(1, args.archive_max_member_mb),
        )
        integrity = archive_integrity_checks(archive)
        write_json(
            integrity,
            final_dir / "00_RUN" / "final_archive_integrity.json",
        )
        if not integrity["pass"]:
            LOG.error("Archive integrity/state check failed: %s", integrity)
            return 3

        LOG.info("Finalization complete")
        LOG.info("Final output folder: %s", final_dir)
        LOG.info("Final clean ZIP: %s", archive)
        print()
        print("=" * 78)
        print("FINALIZED WITHOUT MODEL RERUN")
        print(f"Output: {final_dir}")
        print(f"ZIP:    {archive}")
        print("=" * 78)
        return 0

    except Exception:
        LOG.error("Postprocessing failed:\n%s", traceback.format_exc())
        write_json(
            {
                "status": "failed",
                "failed": datetime.now(ZoneInfo("Europe/Zurich")).isoformat(),
                "traceback": traceback.format_exc(),
                "model_rerun": False,
            },
            final_dir / "00_RUN" / "FINALIZATION_FAILED.json",
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
