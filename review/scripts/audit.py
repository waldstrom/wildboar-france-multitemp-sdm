from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from review.scripts.common import (
    deep_get,
    file_inventory,
    load_yaml,
    relative_or_absolute,
    write_csv,
    write_json,
)

LOG = logging.getLogger("review_corrections.audit")
HUNTING_RE = re.compile(r"hunting|hunt_bag|hunting_bag|tableau.?chasse", re.IGNORECASE)


def _text_predictors(path: Path) -> list[str]:
    if not path.exists():
        return []
    values: list[str] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip().strip(",")
        if not line or line.startswith("#"):
            continue
        values.extend(v.strip() for v in line.split(",") if v.strip())
    return list(dict.fromkeys(values))


def _candidate_stack_names(stack_root: Path) -> list[str]:
    """Read Zarr variable names when possible, then fall back to directory names."""
    names: set[str] = set()
    if not stack_root.exists():
        return []
    try:
        import xarray as xr

        for candidate in [stack_root, *stack_root.glob("*.zarr")]:
            if candidate.exists() and candidate.is_dir():
                try:
                    ds = xr.open_zarr(candidate, consolidated=False)
                    names.update(str(v) for v in ds.data_vars)
                    ds.close()
                except Exception:
                    continue
    except Exception:
        pass

    for path in stack_root.rglob("*"):
        if HUNTING_RE.search(path.name):
            names.add(path.name)
    return sorted(names)


def _find_metrics(experiment_dir: Path, season: str) -> Path | None:
    direct = experiment_dir / "output" / season / "temporal_cv_metrics.csv"
    if direct.exists():
        return direct
    candidates = list(experiment_dir.rglob("temporal_cv_metrics.csv"))
    return max(candidates, key=lambda p: p.stat().st_mtime) if candidates else None


def _find_maps(experiment_dir: Path, season: str, year: int, engine: str) -> list[Path]:
    root = experiment_dir / "output" / season / f"test_{year}" / engine / "maps"
    if not root.exists():
        return []
    out = []
    for path in root.rglob("*.tif"):
        name = path.name.casefold()
        if any(token in name for token in ("stack", "bias", "mess", "limiting", "nt2")):
            continue
        out.append(path)
    return sorted(out)


def _resolve_hunting_sources(cfg: dict[str, Any], repo_root: Path, years: list[int]) -> list[dict[str, Any]]:
    layer = deep_get(cfg, "predictors.layers.hunting_bag")
    if not isinstance(layer, dict):
        # Be tolerant of alternate schemas by recursively locating a hunting-named
        # mapping that looks like a raster source declaration.
        candidates: list[dict[str, Any]] = []

        def walk(value: Any, key_hint: str = "") -> None:
            if isinstance(value, dict):
                identity = " ".join(
                    str(value.get(key, ""))
                    for key in ("name", "variable", "id", "description", "pattern", "path", "file")
                )
                if (HUNTING_RE.search(key_hint) or HUNTING_RE.search(identity)) and any(
                    key in value
                    for key in ("pattern", "path", "file", "dir", "base_dir", "directory")
                ):
                    candidates.append(value)
                for key, child in value.items():
                    walk(child, str(key))
            elif isinstance(value, list):
                for child in value:
                    walk(child, key_hint)

        walk(cfg)
        layer = candidates[0] if candidates else None
    if not isinstance(layer, dict):
        return [
            {
                "year": year,
                "configured": False,
                "path": None,
                "exists": False,
                "detail": "No hunting layer mapping under predictors.layers",
            }
            for year in years
        ]

    pattern = layer.get("pattern") or layer.get("path") or layer.get("file")
    base_dir = Path(str(layer.get("dir") or layer.get("directory") or layer.get("base_dir") or "."))
    if not base_dir.is_absolute():
        base_dir = repo_root / base_dir
    rows = []
    for year in years:
        if pattern:
            try:
                rendered = str(pattern).format(year=year, YEAR=year)
            except Exception:
                rendered = str(pattern).replace("{year}", str(year))
            candidate = Path(rendered)
            if not candidate.is_absolute():
                candidate = base_dir / candidate
            candidates = list(candidate.parent.glob(candidate.name)) if any(
                ch in candidate.name for ch in "*?[]"
            ) else [candidate]
            exists = any(p.exists() for p in candidates)
            selected = next((p for p in candidates if p.exists()), candidate)
        else:
            selected = base_dir
            exists = selected.exists()
        rows.append(
            {
                "year": year,
                "configured": True,
                "path": str(selected),
                "exists": exists,
                "detail": layer.get("name") or layer.get("description"),
            }
        )
    return rows


def audit_scenario(
    repo_root: Path,
    record: dict[str, Any],
    *,
    output_dir: Path,
    expected_splits: list[str],
) -> dict[str, Any]:
    scenario_id = record["id"]
    cfg = load_yaml(Path(record["resolved_config"]))
    exp_dir = Path(record["experiment_dir"])
    season = str(record["season"])
    years = [int(v) for v in record["years"]]
    engines = [str(v) for v in record["engines"]]
    expect_hunting = bool(record.get("include_hunting_after", True))
    output_dir.mkdir(parents=True, exist_ok=True)

    hunting_sources = _resolve_hunting_sources(cfg, repo_root, years)
    write_csv(hunting_sources, output_dir / "hunting_source_inventory.csv")

    stack_root = Path(
        deep_get(cfg, "multitemporal.stack_base_dir", exp_dir / "data-stack")
    )
    candidate_names = _candidate_stack_names(stack_root)
    hunting_candidates = [name for name in candidate_names if HUNTING_RE.search(name)]

    selection_rows: list[dict[str, Any]] = []
    map_rows: list[dict[str, Any]] = []
    for year in years:
        stack_dir = exp_dir / f"input-data-test-{year}" / f"{season}_stack"
        selected_file = stack_dir / "selected_predictors.txt"
        selected = _text_predictors(selected_file)
        selected_hunting = [name for name in selected if HUNTING_RE.search(name)]
        stack_hunting_files = [
            str(path)
            for path in stack_dir.rglob("*")
            if HUNTING_RE.search(path.name)
        ] if stack_dir.exists() else []
        selection_rows.append(
            {
                "scenario": scenario_id,
                "year": year,
                "selected_predictor_file": str(selected_file),
                "selected_predictor_file_exists": selected_file.exists(),
                "selected_predictor_count": len(selected),
                "hunting_selected_count": len(selected_hunting),
                "hunting_selected": "|".join(selected_hunting),
                "hunting_present_in_materialized_test_stack": bool(stack_hunting_files),
                "materialized_hunting_files": "|".join(stack_hunting_files),
                "interpretation": (
                    "retained_after_filtering"
                    if selected_hunting
                    else "candidate_considered_but_not_retained_or_stack_missing"
                ),
            }
        )
        for engine in engines:
            maps = _find_maps(exp_dir, season, year, engine)
            map_rows.append(
                {
                    "scenario": scenario_id,
                    "year": year,
                    "engine": engine,
                    "prediction_map_count": len(maps),
                    "prediction_maps": "|".join(str(p) for p in maps),
                }
            )

    write_csv(selection_rows, output_dir / "hunting_fold_selection_audit.csv")
    write_csv(map_rows, output_dir / "prediction_map_inventory.csv")

    metrics_path = _find_metrics(exp_dir, season)
    metric_issues: list[str] = []
    metric_rows = 0
    observed_years: list[int] = []
    observed_engines: list[str] = []
    observed_splits: list[str] = []
    if metrics_path is None:
        metric_issues.append("temporal_cv_metrics.csv not found")
    else:
        metrics = pd.read_csv(metrics_path)
        metric_rows = len(metrics)
        year_col = next(
            (c for c in ("test_year", "year", "held_out_year") if c in metrics.columns),
            None,
        )
        engine_col = "engine" if "engine" in metrics.columns else None
        split_col = "split" if "split" in metrics.columns else None
        if year_col:
            observed_years = sorted(
                {int(v) for v in pd.to_numeric(metrics[year_col], errors="coerce").dropna()}
            )
            missing_years = sorted(set(years) - set(observed_years))
            if missing_years:
                metric_issues.append(f"missing held-out years: {missing_years}")
        else:
            metric_issues.append("no held-out-year column")
        if engine_col:
            observed_engines = sorted({str(v).lower() for v in metrics[engine_col].dropna()})
            missing_engines = sorted(set(engines) - set(observed_engines))
            if missing_engines:
                metric_issues.append(f"missing engines: {missing_engines}")
        else:
            metric_issues.append("no engine column")
        if split_col:
            observed_splits = sorted({str(v).lower() for v in metrics[split_col].dropna()})
            missing_splits = sorted(set(expected_splits) - set(observed_splits))
            if missing_splits:
                metric_issues.append(f"missing splits: {missing_splits}")
        else:
            metric_issues.append("no split column")

    source_missing = [row["year"] for row in hunting_sources if not row["exists"]]
    map_missing = [
        (row["year"], row["engine"])
        for row in map_rows
        if int(row["prediction_map_count"]) == 0
    ]
    stack_has_hunting = bool(hunting_candidates)
    issues = list(metric_issues)
    if expect_hunting and source_missing:
        issues.append(f"hunting source rasters missing for years: {source_missing}")
    if expect_hunting and not stack_has_hunting:
        issues.append(
            "fresh master stack exposes no hunting-named candidate; "
            "inspect featureconfig/drop rules and coverage metadata"
        )
    if expect_hunting and map_missing:
        issues.append(f"prediction maps missing for {map_missing}")

    summary = {
        "scenario": scenario_id,
        "status": "pass" if not issues else "fail",
        "expect_hunting": expect_hunting,
        "config_include_hunting": bool(
            deep_get(cfg, "multitemporal.include_hunting", True)
        ),
        "hunting_source_years_present": len(years) - len(source_missing),
        "hunting_source_years_expected": len(years),
        "master_stack_path": str(stack_root),
        "master_stack_hunting_candidates": hunting_candidates,
        "master_stack_has_hunting": stack_has_hunting,
        "folds_with_hunting_selected": sum(
            1 for row in selection_rows if row["hunting_selected_count"]
        ),
        "folds_audited": len(selection_rows),
        "metrics_path": str(metrics_path) if metrics_path else None,
        "metric_rows": metric_rows,
        "observed_years": observed_years,
        "observed_engines": observed_engines,
        "observed_splits": observed_splits,
        "missing_map_combinations": map_missing,
        "issues": issues,
    }
    write_json(summary, output_dir / "audit_summary.json")

    inventory_rows = []
    for path in [Path(record["resolved_config"]), Path(record["scenario_dir"]) / "config.diff"]:
        if path.exists():
            item = file_inventory(path)
            item["scenario"] = scenario_id
            inventory_rows.append(item)
    if metrics_path and metrics_path.exists():
        item = file_inventory(metrics_path)
        item["scenario"] = scenario_id
        inventory_rows.append(item)
    write_csv(inventory_rows, output_dir / "key_file_inventory.csv")
    return summary


def aggregate_audits(audit_dirs: Iterable[Path], destination: Path) -> list[dict[str, Any]]:
    rows = []
    for directory in audit_dirs:
        path = Path(directory) / "audit_summary.json"
        if path.exists():
            rows.append(json.loads(path.read_text(encoding="utf-8")))
    write_json(rows, destination.with_suffix(".json"))
    write_csv(
        [
            {
                "scenario": row.get("scenario"),
                "status": row.get("status"),
                "config_include_hunting": row.get("config_include_hunting"),
                "master_stack_has_hunting": row.get("master_stack_has_hunting"),
                "hunting_source_years_present": row.get("hunting_source_years_present"),
                "hunting_source_years_expected": row.get("hunting_source_years_expected"),
                "folds_with_hunting_selected": row.get("folds_with_hunting_selected"),
                "folds_audited": row.get("folds_audited"),
                "issues": " | ".join(row.get("issues", [])),
            }
            for row in rows
        ],
        destination.with_suffix(".csv"),
    )
    return rows
