#!/usr/bin/env python3
"""Shift cloglog maps and recompute metrics for selected experiments.

This utility looks in the ``exps`` directory for the experiments listed in
``EXPERIMENTS``. For each experiment it searches the directory tree for
cloglog maps (``*_cloglog.tif``), shifts each raster 1000 km south and
recomputes basic evaluation metrics.

Presence/background point files stored under ``input-data-test-<year>`` are
recreated for the train/validation/test splits. Each subset is sampled against
the shifted rasters and predictions and metrics are written next to the
corrected maps under ``exps/corrected/<experiment>/``.
"""
from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio import Affine

from scripts.evaluation import _compute_metrics

# ---------------------------------------------------------------------------
# Experiments to correct
EXPERIMENTS = [
    "multitemp-20250828-160346",  # summer
    "multitemp-20250828-214427",  # winter
    "monotemp-20250829-030508",   # summer
    "monotemp-20250829-144241",   # winter
]

# Distance to shift southwards in metres (1000 km)
SHIFT_DISTANCE = 1_000_000


# ---------------------------------------------------------------------------
def _shift_raster(src: Path, dst: Path, distance: float) -> None:
    """Shift ``src`` raster south by ``distance`` metres and save to ``dst``."""
    with rasterio.open(src) as ds:
        data = ds.read()
        meta = ds.meta.copy()
        transform = meta["transform"]
        meta["transform"] = Affine(
            transform.a,
            transform.b,
            transform.c,
            transform.d,
            transform.e,
            transform.f - distance,
        )
        dst.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(dst, "w", **meta) as out:
            out.write(data)


def _parse_run_info(map_path: Path) -> tuple[str | None, str | None]:
    """Return ``(season, test_year)`` inferred from ``map_path``."""
    season = None
    test_year = None
    for parent in map_path.parents:
        if parent.name.startswith("test_"):
            test_year = parent.name.split("_", 1)[1]
        if parent.parent and parent.parent.name == "output":
            season = parent.name
    return season, test_year


def _gather_points(exp_dir: Path, map_path: Path) -> dict[str, pd.DataFrame]:
    """Return evaluation point sets grouped by data split.

    The original experiments saved presence and background GeoPackages for the
    different splits under ``input-data-test-<year>``. This function collects
    all available datasets (train/val/test) and returns them in a dictionary
    keyed by the split name. Each DataFrame contains ``x``, ``y`` and ``label``
    columns suitable for metric calculation.
    """
    season, test_year = _parse_run_info(map_path)
    if not (season and test_year):
        return {}

    input_dir = exp_dir / f"input-data-test-{test_year}"
    if not input_dir.exists():
        return {}

    splits: dict[str, list[pd.DataFrame]] = {}
    for gpkg in input_dir.glob(f"{season}_*.gpkg"):
        name = gpkg.name.lower()
        if "train" in name:
            split = "train"
        elif "val" in name or "validation" in name:
            split = "val"
        elif "test" in name:
            split = "test"
        else:
            continue
        try:
            gdf = gpd.read_file(gpkg)
        except Exception:
            continue
        label = 0 if "background" in name else 1
        gdf["label"] = label
        gdf["x"] = gdf.geometry.x
        gdf["y"] = gdf.geometry.y
        splits.setdefault(split, []).append(gdf[["x", "y", "label"]])

    return {k: pd.concat(v, ignore_index=True) for k, v in splits.items() if v}


def _recalculate_metrics(
    map_path: Path, point_sets: dict[str, pd.DataFrame], out_dir: Path
) -> None:
    """Sample ``map_path`` for each split in ``point_sets`` and persist metrics."""
    out_dir.mkdir(parents=True, exist_ok=True)
    with rasterio.open(map_path) as ds:
        for split, pts in point_sets.items():
            if pts.empty:
                continue
            coords = list(zip(pts['x'], pts['y']))
            samples = np.array([val[0] for val in ds.sample(coords)])
            pts = pts.copy()
            pts['prediction'] = samples
            pts.to_csv(out_dir / f"{split}_evaluation_points.csv", index=False)

            metrics = _compute_metrics(
                pts['label'].to_numpy(), samples, threshold='auto'
            )
            metrics = {
                k: float(v) if np.isscalar(v) else np.asarray(v).tolist()
                for k, v in metrics.items()
                if not k.endswith('_curve')
            }
            with open(out_dir / f"metrics_{split}.json", 'w', encoding='utf-8') as fh:
                json.dump(metrics, fh, indent=2)


# ---------------------------------------------------------------------------
def main() -> None:
    exps_root = Path("exps")
    out_root = exps_root / "corrected"

    for name in EXPERIMENTS:
        exp_dir = exps_root / name
        if not exp_dir.exists():
            print(f"Experiment {name} not found, skipping")
            continue

        out_dir = out_root / name
        cloglog_files = list(exp_dir.rglob("*_cloglog.tif"))
        if not cloglog_files:
            print(f"No cloglog map found in {exp_dir}, skipping")
            continue
        for src_map in cloglog_files:
            dst_map = out_dir / src_map.relative_to(exp_dir)
            _shift_raster(src_map, dst_map, SHIFT_DISTANCE)

            pts_all = _gather_points(exp_dir, src_map)
            splits: dict[str, pd.DataFrame] = {}
            if src_map.name.startswith("train_"):
                for key in ("train", "val"):
                    if key in pts_all:
                        splits[key] = pts_all[key]
            elif src_map.name.startswith("test_"):
                if "test" in pts_all:
                    splits["test"] = pts_all["test"]
            else:
                splits = pts_all
            if splits:
                _recalculate_metrics(dst_map, splits, dst_map.parent)

            print(f"Corrected map {src_map.relative_to(exp_dir)} in experiment {name}")


if __name__ == "__main__":
    main()
