from __future__ import annotations

import inspect
import json
import logging
import math
import os
import re
import shutil
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from review.scripts.common import (
    file_inventory,
    hardlink_or_copy,
    load_yaml,
    relative_or_absolute,
    sanitize_name,
    sha256_file,
    write_csv,
    write_json,
)
from review.scripts.configuration import compare_candidate_configuration

LOG = logging.getLogger("review_corrections.artifacts")

METRIC_ALIASES = {
    "cbi_spearman": ["cbi_spearman", "boyce_spearman", "cbi"],
    "cbi_pearson": ["cbi_pearson", "boyce_pearson"],
    "auc": ["auc", "roc_auc"],
    "pr_auc": ["pr_auc", "average_precision"],
    "omission_rate": ["omission_rate", "omission"],
    "aicc": ["aicc", "AICc"],
}


def _column(frame: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    for name in candidates:
        if name in frame.columns:
            return name
    return None


def _markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows generated._\n"
    display = frame.copy()
    display = display.replace({np.nan: ""})
    headers = [str(c) for c in display.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for _, row in display.iterrows():
        cells = []
        for value in row:
            text = str(value).replace("|", "\\|").replace("\n", " ")
            cells.append(text)
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def write_table_bundle(frame: pd.DataFrame, stem: Path, *, index: bool = False) -> list[Path]:
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    outputs = [
        stem.with_suffix(".csv"),
        stem.with_suffix(".tsv"),
        stem.with_suffix(".md"),
        stem.with_suffix(".tex"),
    ]
    frame.to_csv(outputs[0], index=index)
    frame.to_csv(outputs[1], sep="\t", index=index)
    outputs[2].write_text(_markdown_table(frame), encoding="utf-8")
    try:
        tex = frame.to_latex(index=index, escape=True, na_rep="")
    except Exception as exc:
        tex = f"% LaTeX export failed: {exc}\n"
    outputs[3].write_text(tex, encoding="utf-8")
    return outputs


def load_registry(run_dir: Path) -> list[dict[str, Any]]:
    path = Path(run_dir) / "00_RUN" / "scenario_registry.json"
    return json.loads(path.read_text(encoding="utf-8"))


def collect_metrics(registry: list[dict[str, Any]]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for record in registry:
        exp_dir = Path(record["experiment_dir"])
        path = exp_dir / "output" / record["season"] / "temporal_cv_metrics.csv"
        if not path.exists():
            candidates = list(exp_dir.rglob("temporal_cv_metrics.csv"))
            if candidates:
                path = max(candidates, key=lambda p: p.stat().st_mtime)
        if not path.exists():
            LOG.warning("Metrics missing for %s", record["id"])
            continue
        frame = pd.read_csv(path)
        frame["scenario"] = record["id"]
        frame["dataset"] = record["dataset"]
        frame["temporal"] = record["temporal"]
        frame["scenario_role"] = record["role"]
        frame["season_review"] = record["season"]
        if "engine" not in frame.columns:
            frame["engine"] = "elapid"
        year_col = _column(frame, ["test_year", "held_out_year", "year"])
        if year_col and year_col != "test_year":
            frame["test_year"] = frame[year_col]
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True, sort=False)


def _normalise_metric_columns(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for canonical, candidates in METRIC_ALIASES.items():
        found = _column(out, candidates)
        if found and found != canonical:
            out[canonical] = out[found]
        if canonical in out.columns:
            out[canonical] = pd.to_numeric(out[canonical], errors="coerce")
    return out


def summarise_test_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    work = _normalise_metric_columns(frame)
    if "split" in work.columns:
        work = work[work["split"].astype(str).str.casefold() == "test"]
    group_cols = ["dataset", "temporal", "engine", "season_review"]
    rows: list[dict[str, Any]] = []
    for keys, group in work.groupby(group_cols, dropna=False, sort=False):
        row = dict(zip(group_cols, keys))
        row["n_folds"] = int(group["test_year"].nunique()) if "test_year" in group else len(group)
        for metric in METRIC_ALIASES:
            if metric not in group:
                continue
            values = pd.to_numeric(group[metric], errors="coerce").dropna()
            n = len(values)
            mean = float(values.mean()) if n else math.nan
            sd = float(values.std(ddof=1)) if n > 1 else math.nan
            ci = float(1.96 * sd / math.sqrt(n)) if n > 1 else math.nan
            row[f"{metric}_mean"] = mean
            row[f"{metric}_sd"] = sd
            row[f"{metric}_ci95_low"] = mean - ci if n > 1 else math.nan
            row[f"{metric}_ci95_high"] = mean + ci if n > 1 else math.nan
        rows.append(row)
    summary = pd.DataFrame(rows)
    if summary.empty:
        return summary
    dataset_order = {"allpoints": 0, "gbif": 1, "gbif_grid": 2}
    temporal_order = {"monotemporal": 0, "multitemporal": 1}
    engine_order = {"elapid": 0, "gbm": 1}
    summary["_dataset_order"] = summary["dataset"].map(dataset_order).fillna(99)
    summary["_temporal_order"] = summary["temporal"].map(temporal_order).fillna(99)
    summary["_engine_order"] = summary["engine"].map(engine_order).fillna(99)
    summary = summary.sort_values(
        ["_dataset_order", "_engine_order", "_temporal_order"]
    ).drop(columns=["_dataset_order", "_temporal_order", "_engine_order"])
    return summary.reset_index(drop=True)


def _save_figure(fig: Any, stem: Path, dpi: int) -> list[Path]:
    stem.parent.mkdir(parents=True, exist_ok=True)
    png = stem.with_suffix(".png")
    pdf = stem.with_suffix(".pdf")
    fig.savefig(png, dpi=dpi, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    return [png, pdf]


def _boxplot_with_tick_labels(
    ax: Any,
    groups: list[np.ndarray],
    tick_labels: list[str],
    **kwargs: Any,
) -> Any:
    """Call ``Axes.boxplot`` across Matplotlib's label API transition.

    Matplotlib 3.9 renamed ``labels`` to ``tick_labels`` and Matplotlib 3.10
    removed the old keyword.  Inspecting the bound method keeps the review
    exporter compatible with both the older environments used for the original
    analysis and current environments, without hiding unrelated plotting
    errors behind a broad exception handler.
    """
    parameters = inspect.signature(ax.boxplot).parameters
    label_keyword = "tick_labels" if "tick_labels" in parameters else "labels"
    return ax.boxplot(groups, **{label_keyword: tick_labels}, **kwargs)


def _sort_by_available_columns(
    frame: pd.DataFrame,
    columns: Iterable[str],
) -> pd.DataFrame:
    """Sort by columns present in *frame*, or return an unchanged copy.

    Publication export deliberately continues when an upstream scenario has no
    metrics.  Pandas rejects ``sort_values(by=[])``, so empty/malformed inputs
    must not turn a reportable "missing" artifact into a pipeline crash.
    """
    available = [column for column in columns if column in frame.columns]
    if not available:
        return frame.copy()
    return frame.sort_values(available)


def plot_performance(
    metrics: pd.DataFrame,
    output_dir: Path,
    *,
    dpi: int,
) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    outputs: list[Path] = []
    work = _normalise_metric_columns(metrics)
    if "split" in work.columns:
        work = work[work["split"].astype(str).str.casefold() == "test"]
    for metric, label in [
        ("cbi_spearman", "Continuous Boyce index (Spearman)"),
        ("auc", "Area under the ROC curve"),
    ]:
        if metric not in work.columns or work[metric].dropna().empty:
            continue
        fig, ax = plt.subplots(figsize=(11, 6.5))
        labels: list[str] = []
        groups: list[np.ndarray] = []
        for dataset in ["allpoints", "gbif", "gbif_grid"]:
            for engine in sorted(work["engine"].dropna().astype(str).unique()):
                for temporal in ["monotemporal", "multitemporal"]:
                    subset = work[
                        (work["dataset"] == dataset)
                        & (work["engine"].astype(str) == engine)
                        & (work["temporal"] == temporal)
                    ]
                    values = pd.to_numeric(subset[metric], errors="coerce").dropna().to_numpy()
                    if values.size:
                        groups.append(values)
                        labels.append(
                            f"{dataset}\n{engine}\n"
                            + ("mono" if temporal == "monotemporal" else "multi")
                        )
        if groups:
            _boxplot_with_tick_labels(ax, groups, labels, showmeans=True)
            ax.set_ylabel(label)
            ax.set_title(f"Winter LOYO {label}")
            ax.tick_params(axis="x", labelrotation=30)
            ax.grid(axis="y", alpha=0.25)
            outputs.extend(
                _save_figure(
                    fig,
                    output_dir / f"winter_loyo_{metric}",
                    dpi,
                )
            )
        plt.close(fig)
    return outputs


def _prediction_tif(
    record: dict[str, Any],
    *,
    year: int,
    engine: str,
) -> Path | None:
    root = (
        Path(record["experiment_dir"])
        / "output"
        / record["season"]
        / f"test_{year}"
        / engine
        / "maps"
    )
    candidates = []
    if root.exists():
        for path in root.rglob("*.tif"):
            name = path.name.casefold()
            if any(token in name for token in ("stack", "bias", "mess", "limiting", "nt2")):
                continue
            score = 0
            for token, weight in [
                ("test", 5),
                ("prediction", 4),
                ("suitability", 3),
                ("cloglog", 2),
                (str(year), 1),
            ]:
                if token in name:
                    score += weight
            candidates.append((score, path.stat().st_mtime, path))
    if not candidates:
        return None
    candidates.sort(reverse=True)
    return candidates[0][2]


def plot_prediction_grid(
    registry: list[dict[str, Any]],
    output_dir: Path,
    *,
    year: int,
    engine: str,
    dpi: int,
) -> tuple[list[Path], list[dict[str, Any]]]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import rasterio

    records = {
        (rec["dataset"], rec["temporal"]): rec
        for rec in registry
        if rec.get("role") != "control"
    }
    order = [
        ("allpoints", "monotemporal"),
        ("allpoints", "multitemporal"),
        ("gbif", "monotemporal"),
        ("gbif", "multitemporal"),
    ]
    inventory: list[dict[str, Any]] = []
    arrays = []
    for dataset, temporal in order:
        record = records.get((dataset, temporal))
        path = _prediction_tif(record, year=year, engine=engine) if record else None
        inventory.append(
            {
                "dataset": dataset,
                "temporal": temporal,
                "year": year,
                "engine": engine,
                "path": str(path) if path else None,
                "exists": bool(path and path.exists()),
            }
        )
        if path:
            with rasterio.open(path) as src:
                arr = src.read(1, masked=True)
                arrays.append(arr)
        else:
            arrays.append(None)

    write_csv(inventory, output_dir / "prediction_map_source_inventory.csv")
    if any(arr is None for arr in arrays):
        return [], inventory

    fig, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)
    image = None
    for ax, arr, (dataset, temporal) in zip(axes.flat, arrays, order):
        image = ax.imshow(arr, vmin=0, vmax=1)
        ax.set_title(
            f"{dataset.upper() if dataset == 'gbif' else 'GBIF + WVC'} — "
            f"{'monotemporal' if temporal == 'monotemporal' else 'multitemporal'}"
        )
        ax.set_axis_off()
    if image is not None:
        fig.colorbar(image, ax=list(axes.flat), shrink=0.8, label="Relative habitat suitability")
    fig.suptitle(f"Winter {year} prediction comparison ({engine})")
    outputs = _save_figure(
        fig,
        output_dir / f"winter_{year}_{engine}_prediction_comparison",
        dpi,
    )
    plt.close(fig)
    return outputs, inventory


def _copy_configs(
    run_dir: Path,
    registry: list[dict[str, Any]],
    destination: Path,
    allowed_difference_prefixes: list[str],
) -> list[Path]:
    destination.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    for index, record in enumerate(sorted(registry, key=lambda r: r["order"]), start=1):
        src_cfg = Path(record["scenario_dir"]) / "source_config.yaml"
        resolved = Path(record["resolved_config"])
        diff = Path(record["scenario_dir"]) / "config.diff"
        for source, suffix in [
            (src_cfg, "source"),
            (resolved, "resolved"),
            (diff, "diff"),
        ]:
            if source.exists():
                target = destination / f"{index:02d}_{record['id']}_{suffix}{source.suffix}"
                shutil.copy2(source, target)
                outputs.append(target)

    comparison_rows: list[dict[str, Any]] = []
    for dataset in sorted({r["dataset"] for r in registry}):
        mono = next(
            (r for r in registry if r["dataset"] == dataset and r["temporal"] == "monotemporal" and r.get("role") != "control"),
            None,
        )
        multi = next(
            (r for r in registry if r["dataset"] == dataset and r["temporal"] == "multitemporal"),
            None,
        )
        if not mono or not multi:
            continue
        rows = compare_candidate_configuration(
            Path(mono["resolved_config"]),
            Path(multi["resolved_config"]),
            allowed_difference_prefixes,
        )
        for row in rows:
            row["dataset"] = dataset
        comparison_rows.extend(rows)
    comparison_path = destination / "matched_configuration_difference_audit.csv"
    write_csv(comparison_rows, comparison_path)
    outputs.append(comparison_path)

    confound_rows = []
    for dataset in sorted({r["dataset"] for r in registry}):
        mono = next(
            (r for r in registry if r["dataset"] == dataset and r["temporal"] == "monotemporal" and r.get("role") != "control"),
            None,
        )
        multi = next(
            (r for r in registry if r["dataset"] == dataset and r["temporal"] == "multitemporal"),
            None,
        )
        if mono:
            confound_rows.append(
                {
                    "dataset": dataset,
                    "monotemporal_source_include_hunting": mono.get("include_hunting_before"),
                    "monotemporal_corrected_include_hunting": mono.get("include_hunting_after"),
                    "multitemporal_include_hunting": multi.get("include_hunting_after") if multi else None,
                    "matched_after_correction": (
                        bool(multi)
                        and mono.get("include_hunting_after") == multi.get("include_hunting_after")
                    ),
                }
            )
    confound_path = destination / "hunting_confound_resolution.csv"
    write_csv(confound_rows, confound_path)
    outputs.append(confound_path)

    summary = [
        "# Supplement S5 — corrected winter configuration",
        "",
        "The review run resolves the winter monotemporal hunting-predictor confound without editing the repository's source configurations in place.",
        "",
        "For each dataset, the resolved winter monotemporal and multitemporal runs use `multitemporal.include_hunting: true`.",
        "Fresh scenario-specific stacks are built, and the fold audit records whether the hunting candidate survives correlation/VIF selection.",
        "",
        "The configuration-difference audit classifies remaining differences as declared temporal-method differences or items requiring review.",
        "",
    ]
    summary_path = destination / "S5_corrected_configuration.md"
    summary_path.write_text("\n".join(summary), encoding="utf-8")
    outputs.append(summary_path)
    return outputs


def _collect_audit_tables(run_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    selection_frames = []
    source_frames = []
    audit_root = run_dir / "05_DIAGNOSTICS" / "scenario_audits"
    for scenario_dir in sorted(audit_root.glob("*")):
        selection = scenario_dir / "hunting_fold_selection_audit.csv"
        source = scenario_dir / "hunting_source_inventory.csv"
        if selection.exists():
            frame = pd.read_csv(selection)
            frame["audit_scenario_dir"] = scenario_dir.name
            selection_frames.append(frame)
        if source.exists():
            frame = pd.read_csv(source)
            frame["scenario"] = scenario_dir.name.split("_", 1)[-1]
            source_frames.append(frame)
    return (
        pd.concat(selection_frames, ignore_index=True, sort=False)
        if selection_frames
        else pd.DataFrame(),
        pd.concat(source_frames, ignore_index=True, sort=False)
        if source_frames
        else pd.DataFrame(),
    )


def _collect_feature_importance(run_dir: Path) -> pd.DataFrame:
    frames = []
    root = run_dir / "05_DIAGNOSTICS" / "feature_importance"
    for index_path in root.rglob("feature_importance_index.csv"):
        index = pd.read_csv(index_path)
        for _, meta in index.iterrows():
            path = Path(str(meta.get("permutation_importance", "")))
            if not path.exists():
                continue
            frame = pd.read_csv(path)
            variable_col = _column(frame, ["variable", "feature", "predictor"])
            importance_col = _column(
                frame,
                ["importance_mean", "mean_importance", "permutation_importance", "importance"],
            )
            if not variable_col or not importance_col:
                continue
            normal = pd.DataFrame(
                {
                    "variable": frame[variable_col].astype(str),
                    "importance_mean": pd.to_numeric(frame[importance_col], errors="coerce"),
                }
            )
            normal["scenario"] = meta.get("scenario")
            normal["engine"] = meta.get("engine")
            normal["test_year"] = meta.get("test_year")
            frames.append(normal)
    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()


def _plot_feature_importance(
    frame: pd.DataFrame,
    output_dir: Path,
    *,
    dpi: int,
) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    outputs: list[Path] = []
    if frame.empty:
        return outputs
    for (scenario, engine), group in frame.groupby(["scenario", "engine"], dropna=False):
        group = group.dropna(subset=["importance_mean"]).sort_values(
            "importance_mean", ascending=False
        ).head(20)
        if group.empty:
            continue
        fig, ax = plt.subplots(figsize=(9, 7))
        ordered = group.sort_values("importance_mean")
        ax.barh(ordered["variable"], ordered["importance_mean"])
        ax.set_xlabel("Permutation importance")
        ax.set_title(f"{scenario} — {engine}")
        outputs.extend(
            _save_figure(
                fig,
                output_dir / f"{sanitize_name(str(scenario))}_{sanitize_name(str(engine))}_top20",
                dpi,
            )
        )
        plt.close(fig)
    return outputs


def _copy_response_curves(
    registry: list[dict[str, Any]],
    destination: Path,
    *,
    year: int,
) -> list[Path]:
    outputs: list[Path] = []
    destination.mkdir(parents=True, exist_ok=True)
    counter = 1
    for record in sorted(registry, key=lambda r: r["order"]):
        for engine in record["engines"]:
            root = (
                Path(record["experiment_dir"])
                / "output"
                / record["season"]
                / f"test_{year}"
                / engine
            )
            candidates = [
                p
                for p in root.rglob("*")
                if p.is_file()
                and "response" in p.name.casefold()
                and p.suffix.casefold() in {".pdf", ".png", ".csv"}
            ]
            for source in sorted(candidates):
                target = destination / (
                    f"{counter:03d}_{record['id']}_{engine}_{source.name}"
                )
                hardlink_or_copy(source, target)
                outputs.append(target)
                counter += 1
    return outputs


def _copy_novelty(run_dir: Path, destination: Path) -> list[Path]:
    outputs: list[Path] = []
    source_root = run_dir / "05_DIAGNOSTICS" / "novelty"
    destination.mkdir(parents=True, exist_ok=True)
    counter = 1
    for source in sorted(source_root.rglob("*")):
        if not source.is_file():
            continue
        if source.name in {"COMPLETE", "novelty_state.json"}:
            continue
        if source.suffix.casefold() not in {
            ".csv", ".json", ".jpg", ".jpeg", ".png", ".pdf", ".tif", ".qml"
        }:
            continue
        rel = source.relative_to(source_root)
        target = destination / f"{counter:03d}_{sanitize_name(str(rel))}"
        hardlink_or_copy(source, target)
        outputs.append(target)
        counter += 1
    return outputs


def _copy_full_maps(
    registry: list[dict[str, Any]],
    destination: Path,
) -> tuple[list[Path], pd.DataFrame]:
    outputs: list[Path] = []
    rows: list[dict[str, Any]] = []
    destination.mkdir(parents=True, exist_ok=True)
    counter = 1
    for record in sorted(registry, key=lambda r: r["order"]):
        for year in record["years"]:
            for engine in record["engines"]:
                path = _prediction_tif(record, year=year, engine=engine)
                row = {
                    "scenario": record["id"],
                    "dataset": record["dataset"],
                    "temporal": record["temporal"],
                    "year": year,
                    "engine": engine,
                    "source": str(path) if path else None,
                    "export": None,
                }
                if path:
                    target = destination / (
                        f"{counter:03d}_{record['id']}_{engine}_test_{year}{path.suffix}"
                    )
                    hardlink_or_copy(path, target)
                    outputs.append(target)
                    row["export"] = str(target)
                    counter += 1
                rows.append(row)
    index = pd.DataFrame(rows)
    write_table_bundle(index, destination / "full_corrected_map_index")
    return outputs, index


def _replacement_register(
    artifact_rows: list[dict[str, Any]],
    destination: Path,
) -> list[Path]:
    frame = pd.DataFrame(artifact_rows).sort_values(
        ["section_order", "artifact_order"], kind="stable"
    )
    return write_table_bundle(frame, destination / "replacement_register")


def _artifact_inventory(run_dir: Path, roots: list[Path]) -> pd.DataFrame:
    rows = []
    for root in roots:
        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue
            row = file_inventory(path)
            row["relative_path"] = relative_or_absolute(path, run_dir)
            rows.append(row)
    return pd.DataFrame(rows)


def export_publication_artifacts(
    run_dir: Path,
    registry: list[dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, Any]:
    """Generate manuscript-ordered and supplement-ordered replacement artifacts."""
    run_dir = Path(run_dir)
    manuscript_root = run_dir / "03_MANUSCRIPT"
    supplement_root = run_dir / "04_SUPPLEMENT"
    machine_root = run_dir / "06_MACHINE_READABLE"
    for directory in [manuscript_root, supplement_root, machine_root]:
        directory.mkdir(parents=True, exist_ok=True)

    dpi = int(config.get("review", {}).get("publication", {}).get("figure_dpi", 300))
    map_year = int(config.get("review", {}).get("publication", {}).get("map_year", 2019))
    map_engine = str(
        config.get("review", {}).get("publication", {}).get("map_engine", "elapid")
    )
    allowed_diffs = list(
        config.get("review", {})
        .get("configuration_audit", {})
        .get("allowed_difference_prefixes", [])
    )

    metrics = collect_metrics(registry)
    metrics = _normalise_metric_columns(metrics)
    summary = summarise_test_metrics(metrics)
    metrics.to_csv(machine_root / "all_corrected_temporal_cv_metrics.csv", index=False)
    summary.to_csv(machine_root / "all_corrected_temporal_cv_summary.csv", index=False)

    artifacts: list[dict[str, Any]] = []
    generated: dict[str, list[str]] = {}

    # MAIN 01 — central quantitative replacement table.
    main_01 = manuscript_root / "01_Table_2_LOYO_model_performance"
    outputs = write_table_bundle(summary, main_01 / "Table_2_corrected_winter_LOYO_performance")
    generated["main_table_performance"] = [str(p) for p in outputs]
    artifacts.append(
        {
            "section": "manuscript",
            "section_order": 1,
            "artifact_order": 1,
            "publication_label": "Table 2 — LOYO model performance",
            "directory": str(main_01),
            "action": "replace winter-monotemporal rows and all derived summary values",
            "affected": True,
            "status": "generated" if not summary.empty else "missing",
        }
    )

    # MAIN 02 — performance figures.
    main_02 = manuscript_root / "02_Figure_LOYO_performance"
    outputs = plot_performance(metrics, main_02, dpi=dpi)
    generated["main_figure_performance"] = [str(p) for p in outputs]
    artifacts.append(
        {
            "section": "manuscript",
            "section_order": 1,
            "artifact_order": 2,
            "publication_label": "Main LOYO performance figure(s)",
            "directory": str(main_02),
            "action": "replace any panel containing winter-monotemporal metrics",
            "affected": True,
            "status": "generated" if outputs else "missing",
        }
    )

    # MAIN 03 — 2019 composite maps.
    main_03 = manuscript_root / "03_Figure_2019_winter_prediction_maps"
    outputs, map_inventory = plot_prediction_grid(
        registry, main_03, year=map_year, engine=map_engine, dpi=dpi
    )
    generated["main_figure_maps"] = [str(p) for p in outputs]
    artifacts.append(
        {
            "section": "manuscript",
            "section_order": 1,
            "artifact_order": 3,
            "publication_label": f"Main {map_year} winter monotemporal/multitemporal map comparison",
            "directory": str(main_03),
            "action": "replace corrected monotemporal panels; retain regenerated matched reference panels",
            "affected": True,
            "status": "generated" if outputs else "missing",
        }
    )

    # MAIN 04 — feature importance.
    feature_frame = _collect_feature_importance(run_dir)
    main_04 = manuscript_root / "04_Figure_winter_feature_importance"
    if not feature_frame.empty:
        feature_outputs = write_table_bundle(
            feature_frame.sort_values(
                ["scenario", "engine", "importance_mean"],
                ascending=[True, True, False],
            ),
            main_04 / "winter_feature_importance_full",
        )
        feature_outputs += _plot_feature_importance(feature_frame, main_04, dpi=dpi)
    else:
        feature_outputs = []
    generated["main_feature_importance"] = [str(p) for p in feature_outputs]
    artifacts.append(
        {
            "section": "manuscript",
            "section_order": 1,
            "artifact_order": 4,
            "publication_label": "Winter feature-importance table/figure",
            "directory": str(main_04),
            "action": "replace because the candidate predictor pool and fitted winter monotemporal models changed",
            "affected": True,
            "status": "generated" if feature_outputs else "skipped_or_missing",
        }
    )

    # MAIN 05 — textual-number replacement register.
    main_05 = manuscript_root / "05_Manuscript_numeric_replacement_register"
    main_05.mkdir(parents=True, exist_ok=True)
    numeric = summary.copy()
    if not numeric.empty:
        numeric["recommended_rounding"] = "three decimals unless manuscript specifies otherwise"
    numeric_outputs = write_table_bundle(
        numeric, main_05 / "manuscript_numeric_replacements"
    )
    generated["main_numeric_register"] = [str(p) for p in numeric_outputs]
    artifacts.append(
        {
            "section": "manuscript",
            "section_order": 1,
            "artifact_order": 5,
            "publication_label": "Results-text numerical values",
            "directory": str(main_05),
            "action": "search and replace all winter-monotemporal CBI/AUC/PR-AUC/omission values",
            "affected": True,
            "status": "generated" if not numeric.empty else "missing",
        }
    )

    # SUPPLEMENT 01 — S5 configs.
    supp_01 = supplement_root / "01_Supplement_S5_corrected_configuration"
    config_outputs = _copy_configs(
        run_dir, registry, supp_01, allowed_difference_prefixes=allowed_diffs
    )
    generated["supplement_s5"] = [str(p) for p in config_outputs]
    artifacts.append(
        {
            "section": "supplement",
            "section_order": 2,
            "artifact_order": 1,
            "publication_label": "Supplement S5 — configuration",
            "directory": str(supp_01),
            "action": "replace false hunting setting, include exact resolved configs and diff",
            "affected": True,
            "status": "generated",
        }
    )

    # SUPPLEMENT 02 — S6 full folds and CI.
    supp_02 = supplement_root / "02_Supplement_S6_foldwise_performance_and_CI"
    s6_outputs = write_table_bundle(
        _sort_by_available_columns(
            metrics, ["dataset", "engine", "temporal", "test_year", "split"]
        ),
        supp_02 / "S6_corrected_foldwise_LOYO_metrics",
    )
    s6_outputs += write_table_bundle(
        summary,
        supp_02 / "S6_corrected_LOYO_summary_and_95CI",
    )
    generated["supplement_s6"] = [str(p) for p in s6_outputs]
    artifacts.append(
        {
            "section": "supplement",
            "section_order": 2,
            "artifact_order": 2,
            "publication_label": "Supplement S6 — fold-wise performance and confidence intervals",
            "directory": str(supp_02),
            "action": "replace all winter-monotemporal fold rows and recomputed aggregate intervals",
            "affected": True,
            "status": "generated" if not metrics.empty else "missing",
        }
    )

    # SUPPLEMENT 03 — S7 feature provenance and hunting audit.
    selection, sources = _collect_audit_tables(run_dir)
    supp_03 = supplement_root / "03_Supplement_S7_feature_provenance_and_hunting_audit"
    s7_outputs = []
    s7_outputs += write_table_bundle(
        selection,
        supp_03 / "S7_hunting_candidate_and_fold_selection_audit",
    )
    s7_outputs += write_table_bundle(
        sources,
        supp_03 / "S7_hunting_source_raster_inventory",
    )
    disabled_inventory = run_dir / "00_RUN" / "disabled_hunting_config_inventory.csv"
    if disabled_inventory.exists():
        target = supp_03 / "S7_repository_wide_disabled_config_inventory.csv"
        shutil.copy2(disabled_inventory, target)
        s7_outputs.append(target)
    generated["supplement_s7"] = [str(p) for p in s7_outputs]
    artifacts.append(
        {
            "section": "supplement",
            "section_order": 2,
            "artifact_order": 3,
            "publication_label": "Supplement S7 — feature provenance",
            "directory": str(supp_03),
            "action": "replace selected-predictor records; explicitly distinguish candidate inclusion from filtering retention",
            "affected": True,
            "status": "generated" if not selection.empty else "missing",
        }
    )

    # SUPPLEMENT 04 — response curves / full importance.
    supp_04 = supplement_root / "04_Supplement_response_curves_and_full_importance"
    response_outputs = _copy_response_curves(registry, supp_04 / "response_curves", year=map_year)
    if not feature_frame.empty:
        response_outputs += write_table_bundle(
            feature_frame,
            supp_04 / "full_permutation_importance",
        )
    generated["supplement_response_importance"] = [str(p) for p in response_outputs]
    artifacts.append(
        {
            "section": "supplement",
            "section_order": 2,
            "artifact_order": 4,
            "publication_label": "Supplementary response curves and feature importance",
            "directory": str(supp_04),
            "action": "replace winter-monotemporal response curves and importance outputs",
            "affected": True,
            "status": "generated" if response_outputs else "skipped_or_missing",
        }
    )

    # SUPPLEMENT 05 — novelty.
    supp_05 = supplement_root / "05_Supplement_MESS_NT1_NT2_novelty"
    novelty_outputs = _copy_novelty(run_dir, supp_05)
    generated["supplement_novelty"] = [str(p) for p in novelty_outputs]
    artifacts.append(
        {
            "section": "supplement",
            "section_order": 2,
            "artifact_order": 5,
            "publication_label": "Supplementary MESS/NT1/NT2 novelty diagnostics",
            "directory": str(supp_05),
            "action": "replace monotemporal novelty maps, summaries, limiting-variable and comparison outputs",
            "affected": True,
            "status": "generated" if novelty_outputs else "skipped_or_missing",
        }
    )

    # SUPPLEMENT 06 — all corrected held-out maps.
    supp_06 = supplement_root / "06_Supplement_full_corrected_winter_maps"
    full_map_outputs, full_map_index = _copy_full_maps(registry, supp_06)
    generated["supplement_maps"] = [str(p) for p in full_map_outputs]
    artifacts.append(
        {
            "section": "supplement",
            "section_order": 2,
            "artifact_order": 6,
            "publication_label": "Full winter held-out-year map exports",
            "directory": str(supp_06),
            "action": "replace every winter-monotemporal raster; regenerated references are indexed alongside them",
            "affected": True,
            "status": "generated" if full_map_outputs else "missing",
        }
    )

    # Replacement register is deliberately last in manuscript order.
    replacement_outputs = _replacement_register(artifacts, main_05)
    generated["replacement_register"] = [str(p) for p in replacement_outputs]

    status_path = run_dir / "00_RUN" / "artifact_status.csv"
    write_csv(artifacts, status_path)
    write_json(generated, run_dir / "00_RUN" / "generated_artifacts.json")

    inventory = _artifact_inventory(
        run_dir, [manuscript_root, supplement_root, machine_root]
    )
    inventory.to_csv(run_dir / "00_RUN" / "artifact_file_manifest.csv", index=False)

    return {
        "artifacts": artifacts,
        "generated": generated,
        "metrics_rows": len(metrics),
        "summary_rows": len(summary),
        "artifact_inventory_rows": len(inventory),
    }
