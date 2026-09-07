"""Compute and compare MESS (multivariate environmental similarity surfaces).

This utility reads artefacts from mono- and multitemporal experiment runs and
derives multivariate environmental similarity diagnostics for every available
test year.  The implementation follows the recommendations by Elith et al.
(2010) and Mesgaran et al. (2014) by reporting both MESS scores and the
limiting predictor responsible for the minimum similarity.  When supplied with
matching mono- and multitemporal runs stemming from the same meta experiment it
generates a comparison dashboard that highlights novelty and correlation
changes for each test scenario.

Examples
--------

Compute MESS diagnostics for a single run::

    python scripts/mess_evaluation.py --run-dir path/to/output/monotemporal/winter

Generate a full mono-versus-multi comparison::

    python scripts/mess_evaluation.py \
        --mono-run path/to/output/monotemporal/winter \
        --multi-run path/to/output/multitemporal/winter
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import platform
import subprocess
import sys
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field
from datetime import datetime
import colorsys
from pathlib import Path
from typing import Iterable, Sequence
from xml.sax.saxutils import escape

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window
from tqdm.auto import tqdm

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors

log = logging.getLogger(__name__)


# Histogram configuration used for summary statistics.  Values below the lower
# bound are clipped which is acceptable as such extreme scores indicate strong
# extrapolation and already dominate the minimum.
HIST_MIN = -500.0
HIST_MAX = 100.0
HIST_STEP = 0.5
HIST_EDGES = np.arange(HIST_MIN, HIST_MAX + HIST_STEP, HIST_STEP, dtype="float64")
NT2_QUANTILE = 0.99


class _StreamTee:
    """Duplicate writes to both the original stream and a log file."""

    def __init__(self, original, log_file):
        self._original = original
        self._log_file = log_file

    def write(self, data: str) -> int:
        written = self._original.write(data)
        self._log_file.write(data)
        return written

    def flush(self) -> None:
        self._original.flush()
        self._log_file.flush()


@contextmanager
def capture_console(output_root: Path, parameters: dict[str, object] | None = None):
    """Capture stdout/stderr into a log file while preserving console output."""

    output_root.mkdir(parents=True, exist_ok=True)
    log_path = output_root / "console.log"
    params_path = output_root / "parameters.json"
    if parameters is not None:
        params_path.write_text(json.dumps(parameters, indent=2, default=str), encoding="utf-8")
    with open(log_path, "w", encoding="utf-8") as log_file:
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        sys.stdout = _StreamTee(original_stdout, log_file)
        sys.stderr = _StreamTee(original_stderr, log_file)
        try:
            yield log_path
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            sys.stdout = original_stdout
            sys.stderr = original_stderr


def _current_git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        )
    except Exception:
        return None
    return result.stdout.strip() or None


@dataclass
class ReferenceStats:
    """Container storing reference distribution information for a variable."""

    name: str
    values: np.ndarray

    def __post_init__(self) -> None:
        valid = self.values[np.isfinite(self.values)]
        if valid.size == 0:
            raise ValueError(f"Reference data for '{self.name}' contains no valid values")
        self.sorted = np.sort(valid.astype("float64"))
        self.min = float(self.sorted[0])
        self.max = float(self.sorted[-1])
        self.range = float(self.max - self.min)


@dataclass
class TrainingContext:
    """Statistics derived from the training data for a predictor set."""

    predictors: list[str]
    reference: dict[str, ReferenceStats]
    correlation: np.ndarray
    mean: np.ndarray
    cov: np.ndarray
    inv_cov: np.ndarray
    md2_threshold: float
    nt2_quantile: float


class CorrelationAccumulator:
    """Streaming estimator for Pearson correlation matrices."""

    def __init__(self, size: int) -> None:
        self.size = size
        self.count = 0
        self.sum = np.zeros(size, dtype="float64")
        self.sum_sq = np.zeros(size, dtype="float64")
        self.sum_prod = np.zeros((size, size), dtype="float64")

    def update(self, values: np.ndarray) -> None:
        if values.shape[0] != self.size:
            raise ValueError("Mismatched predictor count for correlation accumulator")
        mask = np.all(np.isfinite(values), axis=0)
        if not np.any(mask):
            return
        valid = values[:, mask].astype("float64", copy=False)
        self.count += valid.shape[1]
        self.sum += np.sum(valid, axis=1)
        self.sum_sq += np.sum(valid**2, axis=1)
        self.sum_prod += valid @ valid.T

    def correlation_matrix(self) -> np.ndarray:
        if self.count < 2:
            return np.full((self.size, self.size), np.nan, dtype="float64")
        cov = self.sum_prod - np.outer(self.sum, self.sum) / self.count
        cov /= max(self.count - 1, 1)
        var = np.diag(cov)
        var = np.maximum(var, 0.0)
        std = np.sqrt(var, dtype=np.float64)
        denom = np.outer(std, std)
        corr = np.divide(cov, denom, out=np.full_like(cov, np.nan), where=denom > 0)
        np.fill_diagonal(corr, 1.0)
        return corr


@dataclass
class VariableAccumulator:
    """Track summary statistics for per-variable similarity scores."""

    name: str
    hist_edges: np.ndarray
    count: int = 0
    sum: float = 0.0
    sum_sq: float = 0.0
    min: float = field(default=float("inf"))
    max: float = field(default=float("-inf"))
    negative: int = 0
    below10: int = 0
    outside_low: int = 0
    outside_high: int = 0
    hist: np.ndarray = field(default_factory=lambda: np.zeros(len(HIST_EDGES) - 1, dtype="float64"))
    limiting_total: int = 0
    limiting_negative: int = 0

    def update(self, scores: np.ndarray, mask_valid: np.ndarray, mask_low: np.ndarray, mask_high: np.ndarray) -> None:
        if not np.any(mask_valid):
            return
        values = scores[mask_valid]
        self.count += values.size
        self.sum += float(np.sum(values, dtype="float64"))
        self.sum_sq += float(np.sum(values.astype("float64") ** 2))
        self.min = min(self.min, float(np.min(values)))
        self.max = max(self.max, float(np.max(values)))
        self.negative += int(np.count_nonzero(values < 0))
        self.below10 += int(np.count_nonzero(values < 10))
        self.outside_low += int(np.count_nonzero(mask_low[mask_valid]))
        self.outside_high += int(np.count_nonzero(mask_high[mask_valid]))
        clipped = np.clip(values, HIST_MIN, HIST_MAX)
        hist, _ = np.histogram(clipped, bins=self.hist_edges)
        self.hist += hist

    def mean(self) -> float:
        return float(self.sum / self.count) if self.count else float("nan")

    def std(self) -> float:
        if not self.count:
            return float("nan")
        mean = self.sum / self.count
        var = self.sum_sq / self.count - mean**2
        var = max(var, 0.0)
        return float(math.sqrt(var))


@dataclass
class OverallAccumulator:
    """Track summary statistics for the final MESS surface."""

    hist_edges: np.ndarray
    pixel_area_km2: float
    count: int = 0
    sum: float = 0.0
    sum_sq: float = 0.0
    min: float = field(default=float("inf"))
    max: float = field(default=float("-inf"))
    negative: int = 0
    below10: int = 0
    below20: int = 0
    above50: int = 0
    hist: np.ndarray = field(default_factory=lambda: np.zeros(len(HIST_EDGES) - 1, dtype="float64"))

    def update(self, values: np.ndarray) -> None:
        if values.size == 0:
            return
        values64 = values.astype("float64", copy=False)
        self.count += values64.size
        self.sum += float(np.sum(values64))
        self.sum_sq += float(np.sum(values64**2))
        self.min = min(self.min, float(np.min(values64)))
        self.max = max(self.max, float(np.max(values64)))
        self.negative += int(np.count_nonzero(values64 < 0))
        self.below10 += int(np.count_nonzero((values64 >= 0) & (values64 < 10)))
        self.below20 += int(np.count_nonzero((values64 >= 10) & (values64 < 20)))
        self.above50 += int(np.count_nonzero(values64 >= 50))
        clipped = np.clip(values64, HIST_MIN, HIST_MAX)
        hist, _ = np.histogram(clipped, bins=self.hist_edges)
        self.hist += hist

    @property
    def area_km2(self) -> float:
        return self.count * self.pixel_area_km2

    @property
    def negative_area_km2(self) -> float:
        return self.negative * self.pixel_area_km2

    @property
    def below10_area_km2(self) -> float:
        return self.below10 * self.pixel_area_km2

    @property
    def below20_area_km2(self) -> float:
        return self.below20 * self.pixel_area_km2

    @property
    def above50_area_km2(self) -> float:
        return self.above50 * self.pixel_area_km2

    def mean(self) -> float:
        return float(self.sum / self.count) if self.count else float("nan")

    def std(self) -> float:
        if not self.count:
            return float("nan")
        mean = self.sum / self.count
        var = self.sum_sq / self.count - mean**2
        var = max(var, 0.0)
        return float(math.sqrt(var))


def quantiles_from_hist(hist: np.ndarray, edges: np.ndarray, probs: Sequence[float]) -> list[float]:
    if hist.sum() == 0:
        return [float("nan") for _ in probs]
    cdf = np.cumsum(hist)
    total = cdf[-1]
    out: list[float] = []
    for p in probs:
        target = p * total
        idx = int(np.searchsorted(cdf, target, side="left"))
        idx = min(max(idx, 0), len(hist) - 1)
        left = edges[idx]
        right = edges[idx + 1]
        prev = cdf[idx - 1] if idx > 0 else 0.0
        count = hist[idx]
        if count == 0:
            out.append(float(left))
        else:
            fraction = (target - prev) / count
            fraction = max(0.0, min(1.0, fraction))
            out.append(float(left + fraction * (right - left)))
    return out


def find_experiment_root(start: Path, year: int) -> Path:
    marker = f"input-data-test-{year}"
    for candidate in [start, *start.parents]:
        if (candidate / marker).exists():
            return candidate
    raise FileNotFoundError(
        f"Could not locate experiment root from {start}; '{marker}' is missing"
    )


def read_predictors(stack_dir: Path) -> list[str]:
    predictors_file = stack_dir / "selected_predictors.txt"
    if predictors_file.exists():
        predictors = [line.strip() for line in predictors_file.read_text().splitlines() if line.strip()]
        if predictors:
            return predictors
    return sorted(
        f.stem
        for f in stack_dir.glob("*.tif")
        if f.is_file() and f.stem not in {"boundary_mask"}
    )


def load_training_context(
    features_dir: Path, season: str, predictors: Sequence[str], *, nt2_quantile: float
) -> TrainingContext:
    train_file = features_dir / f"{season}_train_features.csv"
    bg_file = features_dir / f"{season}_background_features.csv"
    if not train_file.exists():
        raise FileNotFoundError(f"Training features file missing: {train_file}")
    if not bg_file.exists():
        raise FileNotFoundError(f"Background features file missing: {bg_file}")
    train_df = pd.read_csv(train_file)
    bg_df = pd.read_csv(bg_file)
    stats: dict[str, ReferenceStats] = {}
    valid_predictors: list[str] = []
    for var in predictors:
        if var not in train_df or var not in bg_df:
            logging.error(
                "Predictor '%s' not found in training tables; skipping variable",
                var,
            )
            continue
        series = pd.concat([train_df[var], bg_df[var]], ignore_index=True)
        stats[var] = ReferenceStats(var, series.to_numpy(dtype="float64", copy=False))
        valid_predictors.append(var)

    if not valid_predictors:
        raise ValueError("No predictors available after filtering missing training variables")

    combo = pd.concat([train_df[valid_predictors], bg_df[valid_predictors]], ignore_index=True)
    combo = combo.replace([np.inf, -np.inf], np.nan).dropna(axis=0, how="any")
    if combo.empty:
        raise ValueError("Training data for correlation analysis is empty after cleaning")
    mat = combo.to_numpy(dtype="float64", copy=False)
    corr = np.corrcoef(mat, rowvar=False)
    corr = np.asarray(corr, dtype="float64")
    if corr.ndim == 0:
        corr = np.array([[float(corr)]], dtype="float64")
    elif corr.ndim == 1:
        corr = np.array([corr], dtype="float64")

    mu = np.nanmean(mat, axis=0)
    cov = np.cov(mat, rowvar=False)
    cov = np.asarray(cov, dtype="float64")
    if cov.ndim == 0:
        cov = np.array([[float(cov)]], dtype="float64")
    elif cov.ndim == 1:
        cov = np.array([cov], dtype="float64")
    inv_cov = np.linalg.pinv(cov, rcond=1e-8)
    diffs = mat - mu
    md2_train = np.einsum("ij,jk,ik->i", diffs, inv_cov, diffs, dtype="float64")
    finite_md2 = md2_train[np.isfinite(md2_train)]
    if finite_md2.size:
        tau = float(np.nanquantile(finite_md2, nt2_quantile))
    else:
        tau = float("nan")

    return TrainingContext(valid_predictors, stats, corr, mu, cov, inv_cov, tau, nt2_quantile)


def _mess_scores(values: np.ndarray, stats: ReferenceStats) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return MESS scores for ``values`` and auxiliary masks.

    Parameters
    ----------
    values:
        Flat array of predictor values for the evaluation grid.
    stats:
        :class:`ReferenceStats` describing the training distribution.

    Returns
    -------
    scores, valid_mask, low_mask, high_mask
        Similarity scores and boolean masks indicating valid cells as well as
        values below and above the training range.
    """

    scores = np.full(values.shape, np.nan, dtype="float32")
    valid = np.isfinite(values)
    if not np.any(valid):
        return scores, valid, valid.copy(), valid.copy()

    vals = values[valid]
    low = vals < stats.min
    high = vals > stats.max
    inside = ~(low | high)

    if stats.range > 0:
        scores_valid = np.empty(vals.shape, dtype="float32")
        scores_valid[low] = 100.0 * (vals[low] - stats.min) / stats.range
        scores_valid[high] = 100.0 * (stats.max - vals[high]) / stats.range
        if stats.sorted.size and np.any(inside):
            ranks = np.searchsorted(stats.sorted, vals[inside], side="right").astype("float32")
            cdf = ranks / stats.sorted.size
            res = np.empty_like(cdf, dtype="float32")
            lower = cdf <= 0.5
            if np.any(lower):
                res[lower] = (cdf[lower] / 0.5) * 100.0
            if np.any(~lower):
                res[~lower] = ((1.0 - cdf[~lower]) / 0.5) * 100.0
            scores_valid[inside] = res
        else:
            scores_valid[inside] = 100.0
    else:
        scores_valid = np.full(vals.shape, -100.0, dtype="float32")
        scores_valid[vals == stats.min] = 100.0

    scores[valid] = scores_valid
    mask_low = np.zeros(values.shape, dtype=bool)
    mask_high = np.zeros(values.shape, dtype=bool)
    mask_low[valid] = low
    mask_high[valid] = high
    return scores, valid, mask_low, mask_high


def iter_windows(height: int, width: int, block_h: int, block_w: int) -> Iterable[Window]:
    for row_off in range(0, height, block_h):
        h = min(block_h, height - row_off)
        for col_off in range(0, width, block_w):
            w = min(block_w, width - col_off)
            yield Window(col_off, row_off, w, h)


def _generate_palette(names: Sequence[str]) -> list[str]:
    colors: list[str] = []
    n = max(len(names), 1)
    for idx in range(n):
        hue = (idx / n) % 1.0
        r, g, b = colorsys.hsv_to_rgb(hue, 0.65, 0.85)
        colors.append("#{:02X}{:02X}{:02X}".format(int(r * 255), int(g * 255), int(b * 255)))
    return colors


def _write_limiting_qml(path: Path, mapping: dict[int, str]) -> None:
    keys = [idx for idx in sorted(mapping) if idx >= 0]
    names = [mapping[idx] for idx in keys]
    colors = _generate_palette(names)
    entries = []
    for color, idx, name in zip(colors, keys, names):
        label = escape(name, {"\"": "&quot;"})
        entries.append(
            f'      <paletteEntry value="{idx}" label="{label}" color="{color}" alpha="255"/>'
        )
    qml = """<?xml version="1.0" encoding="UTF-8"?>
<qgis version="3" styleCategories="AllStyleCategories">
  <pipe>
    <rasterrenderer band="1" type="paletted" opacity="1">
      <rasterTransparency/>
      <minMaxOrigin>
        <limits>None</limits>
        <extent>WholeRaster</extent>
        <statAccuracy>Estimated</statAccuracy>
      </minMaxOrigin>
      <colorPalette>
{entries}
      </colorPalette>
    </rasterrenderer>
  </pipe>
</qgis>
"""
    path.write_text(qml.format(entries="\n".join(entries)), encoding="utf-8")


def _save_continuous_raster_jpeg(
    raster_path: Path,
    jpeg_path: Path,
    *,
    title: str,
    cmap: str = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
) -> None:
    with rasterio.open(raster_path) as src:
        data = src.read(1, masked=True).filled(np.nan)
    mask = np.isfinite(data)
    fig, ax = plt.subplots(figsize=(8, 6))
    if not np.any(mask):
        ax.text(0.5, 0.5, "No valid data", ha="center", va="center", fontsize=14)
        ax.set_title(title)
        ax.axis("off")
        fig.savefig(jpeg_path, dpi=200, format="jpg")
        plt.close(fig)
        return
    im = ax.imshow(np.where(mask, data, np.nan), cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.axis("off")
    fig.colorbar(im, ax=ax, shrink=0.75)
    fig.savefig(jpeg_path, dpi=200, format="jpg")
    plt.close(fig)


def _save_discrete_raster_jpeg(
    raster_path: Path,
    jpeg_path: Path,
    *,
    title: str,
    labels: Sequence[str],
    values: Sequence[int],
    colors: Sequence[str],
    nodata: int,
) -> None:
    with rasterio.open(raster_path) as src:
        data = src.read(1)
    mask = data != nodata
    fig, ax = plt.subplots(figsize=(8, 6))
    if not np.any(mask):
        ax.text(0.5, 0.5, "No valid data", ha="center", va="center", fontsize=14)
        ax.set_title(title)
        ax.axis("off")
        fig.savefig(jpeg_path, dpi=200, format="jpg")
        plt.close(fig)
        return
    display = np.where(mask, data, np.nan)
    cmap = mcolors.ListedColormap(colors)
    bounds = [v - 0.5 for v in values] + [values[-1] + 0.5]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)
    im = ax.imshow(display, cmap=cmap, norm=norm)
    ax.set_title(title)
    ax.axis("off")
    cbar = fig.colorbar(im, ax=ax, shrink=0.75, ticks=values)
    cbar.ax.set_yticklabels(labels)
    fig.savefig(jpeg_path, dpi=200, format="jpg")
    plt.close(fig)


def _save_histogram_jpeg(
    values: np.ndarray,
    jpeg_path: Path,
    *,
    title: str,
    xlabel: str,
    bins: int = 50,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    if values.size == 0:
        ax.text(0.5, 0.5, "No values", ha="center", va="center", fontsize=14)
        ax.set_axis_off()
    else:
        ax.hist(values, bins=bins, color="#1f77b4", edgecolor="black", alpha=0.85)
        ax.set_ylabel("Frequency")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    fig.tight_layout()
    fig.savefig(jpeg_path, dpi=200, format="jpg")
    plt.close(fig)


def _save_correlation_heatmap(diff: np.ndarray, jpeg_path: Path, predictors: Sequence[str], title: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    if diff.size == 0:
        ax.text(0.5, 0.5, "No correlation data", ha="center", va="center", fontsize=14)
        ax.set_axis_off()
    else:
        im = ax.imshow(diff, cmap="coolwarm", vmin=-1, vmax=1)
        ax.set_xticks(range(len(predictors)))
        ax.set_yticks(range(len(predictors)))
        ax.set_xticklabels(predictors, rotation=45, ha="right")
        ax.set_yticklabels(predictors)
        ax.set_title(title)
        fig.colorbar(im, ax=ax, shrink=0.75, label="Δ correlation")
    fig.tight_layout()
    fig.savefig(jpeg_path, dpi=200, format="jpg")
    plt.close(fig)


def _generate_visualizations(
    out_dir: Path,
    run_label: str,
    season: str,
    year: int,
    predictors: Sequence[str],
    mapping: dict[int, str],
    corr_diff: np.ndarray,
) -> dict[str, Path]:
    visualizations: dict[str, Path] = {}
    title_prefix = f"{run_label} {season} {year}"

    mess_raster = out_dir / "mess_scores.tif"
    mess_jpeg = out_dir / "mess_scores.jpg"
    _save_continuous_raster_jpeg(
        mess_raster,
        mess_jpeg,
        title=f"{title_prefix} – MESS",
        cmap="RdYlGn",
        vmin=HIST_MIN,
        vmax=HIST_MAX,
    )
    visualizations["mess_jpeg"] = mess_jpeg

    limiting_raster = out_dir / "mess_limiting_variable.tif"
    limiting_jpeg = out_dir / "mess_limiting_variable.jpg"
    keys = [idx for idx in sorted(mapping) if idx >= 0]
    if keys:
        labels = [mapping[idx] for idx in keys]
        colors = _generate_palette(labels)
        _save_discrete_raster_jpeg(
            limiting_raster,
            limiting_jpeg,
            title=f"{title_prefix} – Limiting Variable",
            labels=labels,
            values=keys,
            colors=colors,
            nodata=-1,
        )
        visualizations["limiting_jpeg"] = limiting_jpeg

    nt2_raster = out_dir / "nt2_mahalanobis.tif"
    nt2_jpeg = out_dir / "nt2_mahalanobis.jpg"
    _save_continuous_raster_jpeg(
        nt2_raster,
        nt2_jpeg,
        title=f"{title_prefix} – NT2",
        cmap="magma",
        vmin=0.0,
    )
    visualizations["nt2_jpeg"] = nt2_jpeg

    class_raster = out_dir / "exdet_class.tif"
    class_jpeg = out_dir / "exdet_class.jpg"
    class_values = [0, 1, 2, 3]
    class_labels = ["None", "NT1 only", "NT2 only", "Both"]
    class_colors = ["#D3D3D3", "#E66101", "#5E3C99", "#4DAF4A"]
    _save_discrete_raster_jpeg(
        class_raster,
        class_jpeg,
        title=f"{title_prefix} – Novelty Classes",
        labels=class_labels,
        values=class_values,
        colors=class_colors,
        nodata=255,
    )
    visualizations["exdet_class_jpeg"] = class_jpeg

    with rasterio.open(mess_raster) as src:
        mess_data = src.read(1, masked=True).compressed()
    mess_hist = out_dir / "mess_histogram.jpg"
    _save_histogram_jpeg(
        mess_data,
        mess_hist,
        title=f"{title_prefix} – MESS Distribution",
        xlabel="MESS score",
        bins=60,
    )
    visualizations["mess_histogram_jpeg"] = mess_hist

    with rasterio.open(nt2_raster) as src:
        nt2_data = src.read(1, masked=True).compressed()
    nt2_hist = out_dir / "nt2_histogram.jpg"
    _save_histogram_jpeg(
        nt2_data,
        nt2_hist,
        title=f"{title_prefix} – NT2 Distribution",
        xlabel="NT2 index",
        bins=60,
    )
    visualizations["nt2_histogram_jpeg"] = nt2_hist

    corr_heatmap = out_dir / "correlation_change.jpg"
    _save_correlation_heatmap(
        corr_diff,
        corr_heatmap,
        predictors,
        title=f"{title_prefix} – Correlation Change",
    )
    visualizations["correlation_heatmap_jpeg"] = corr_heatmap

    return visualizations


def process_season_year(
    season_dir: Path,
    test_dir: Path,
    year: int,
    predictors: Sequence[str],
    training: TrainingContext,
    out_dir: Path,
    *,
    run_label: str,
    compression: str = "LZW",
) -> dict[str, Path]:
    """Compute MESS surfaces for a single season/year combination."""

    test_stack = test_dir / "test_stack"
    if not test_stack.exists():
        raise FileNotFoundError(f"Test stack not found in {test_stack}")

    predictors = [p for p in predictors if p != "boundary_mask"]
    if not predictors:
        raise ValueError("No predictors available for MESS computation")

    out_dir.mkdir(parents=True, exist_ok=True)

    outputs: dict[str, Path] = {}

    stats = training.reference

    boundary_mask_path = test_stack / "boundary_mask.tif"
    has_boundary_mask = boundary_mask_path.exists()

    with ExitStack() as stack:
        missing = [var for var in predictors if not (test_stack / f"{var}.tif").exists()]
        if missing:
            raise FileNotFoundError(
                "Missing predictors in test stack for MESS projection: "
                + ", ".join(sorted(missing))
                + f" (searched in {test_stack})"
            )
        srcs = {var: stack.enter_context(rasterio.open(test_stack / f"{var}.tif")) for var in predictors}
        mask_src = None
        if has_boundary_mask:
            mask_src = stack.enter_context(rasterio.open(boundary_mask_path))
        template = next(iter(srcs.values()))
        height, width = template.height, template.width
        if template.block_shapes:
            block_h, block_w = template.block_shapes[0]
        else:
            block_h = block_w = 512
        block_h = max(1, block_h)
        block_w = max(1, block_w)
        transform = template.transform
        crs = template.crs
        pixel_area = abs(transform.a * transform.e)
        pixel_area_km2 = pixel_area / 1_000_000.0

        mess_path = out_dir / "mess_scores.tif"
        limiting_path = out_dir / "mess_limiting_variable.tif"
        nt2_path = out_dir / "nt2_mahalanobis.tif"
        class_path = out_dir / "exdet_class.tif"
        profile = template.profile.copy()
        profile.update(
            dtype="float32",
            count=1,
            nodata=np.nan,
            compress=compression,
        )
        limit_profile = template.profile.copy()
        limit_profile.update(
            dtype="int16",
            count=1,
            nodata=-1,
            compress=compression,
        )
        nt2_profile = profile.copy()
        class_profile = template.profile.copy()
        class_profile.update(dtype="uint8", count=1, nodata=255, compress=compression)

        corr_acc = CorrelationAccumulator(len(predictors))

        with rasterio.open(mess_path, "w", **profile) as dst_mess, \
            rasterio.open(limiting_path, "w", **limit_profile) as dst_lim, \
            rasterio.open(nt2_path, "w", **nt2_profile) as dst_nt2, \
            rasterio.open(class_path, "w", **class_profile) as dst_cls:
            overall = OverallAccumulator(HIST_EDGES, pixel_area_km2)
            per_var = [VariableAccumulator(var, HIST_EDGES) for var in predictors]
            limiting_total = np.zeros(len(predictors), dtype="int64")
            limiting_negative = np.zeros(len(predictors), dtype="int64")
            novelty_counts = np.zeros(4, dtype="int64")

            total_windows = math.ceil(height / block_h) * math.ceil(width / block_w)
            pbar = tqdm(
                iter_windows(height, width, block_h, block_w),
                total=total_windows,
                desc=f"MESS {season_dir.name} {year}",
                unit="window",
            )
            for window in pbar:
                rows = window.height * window.width
                scores_per_var: list[np.ndarray] = []
                masks_valid: list[np.ndarray] = []
                masks_low: list[np.ndarray] = []
                masks_high: list[np.ndarray] = []
                values_per_var: list[np.ndarray] = []
                if mask_src is not None:
                    mask_data = mask_src.read(1, window=window, masked=True).filled(0)
                    boundary_valid = mask_data.reshape(rows) > 0.5
                else:
                    boundary_valid = None
                for idx, var in enumerate(predictors):
                    data = (
                        srcs[var]
                        .read(1, window=window, masked=True)
                        .filled(np.nan)
                        .astype("float32", copy=False)
                    )
                    flat = data.reshape(rows)
                    if boundary_valid is not None:
                        flat = flat.copy()
                        flat[~boundary_valid] = np.nan
                    values_per_var.append(flat)
                    score, mask_valid, mask_low, mask_high = _mess_scores(flat, stats[var])
                    scores_per_var.append(score)
                    masks_valid.append(mask_valid)
                    masks_low.append(mask_low)
                    masks_high.append(mask_high)
                    per_var[idx].update(score, mask_valid, mask_low, mask_high)

                stacked = np.vstack(values_per_var)
                corr_acc.update(stacked)

                mess = np.full(rows, np.inf, dtype="float32")
                limiting = np.full(rows, -1, dtype="int16")
                any_valid = np.zeros(rows, dtype=bool)
                for idx, score in enumerate(scores_per_var):
                    valid = masks_valid[idx]
                    finite = valid & np.isfinite(score)
                    if not np.any(finite):
                        continue
                    better = finite & (score < mess)
                    mess[better] = score[better]
                    limiting[better] = idx
                    any_valid |= finite

                mess[~any_valid] = np.nan

                X = stacked.T
                nt2 = np.full(rows, np.nan, dtype="float32")
                valid_nt2 = np.all(np.isfinite(X), axis=1)
                if np.any(valid_nt2):
                    centered = X[valid_nt2].astype("float64", copy=False) - training.mean
                    md2 = np.einsum(
                        "ij,jk,ik->i", centered, training.inv_cov, centered, dtype="float64"
                    )
                    tau = training.md2_threshold if training.md2_threshold > 0 else float("nan")
                    if np.isfinite(tau) and tau > 0:
                        nt2_vals = md2 / tau
                        nt2[valid_nt2] = nt2_vals.astype("float32", copy=False)
                    else:
                        nt2[valid_nt2] = np.nan

                cls = np.full(rows, 255, dtype="uint8")
                valid_mess = np.isfinite(mess)
                valid_both = valid_mess & np.isfinite(nt2)
                if np.any(valid_both):
                    nt1_mask = mess[valid_both] < 0
                    nt2_mask = nt2[valid_both] > 1.0
                    cls_vals = np.zeros(nt1_mask.size, dtype="uint8")
                    cls_vals[nt1_mask & ~nt2_mask] = 1
                    cls_vals[~nt1_mask & nt2_mask] = 2
                    cls_vals[nt1_mask & nt2_mask] = 3
                    cls[valid_both] = cls_vals
                    counts = np.bincount(cls_vals.astype(int), minlength=4)
                    novelty_counts[:4] += counts[:4]

                if boundary_valid is not None:
                    nodata_mask = ~boundary_valid
                    mess[nodata_mask] = np.nan
                    limiting[nodata_mask] = -1
                    nt2[nodata_mask] = np.nan
                    cls[nodata_mask] = 255

                mess2d = mess.reshape(window.height, window.width)
                limiting2d = limiting.reshape(window.height, window.width)
                nt22d = nt2.reshape(window.height, window.width)
                cls2d = cls.reshape(window.height, window.width)
                dst_mess.write(mess2d, window=window, indexes=1)
                dst_lim.write(limiting2d, window=window, indexes=1)
                dst_nt2.write(nt22d, window=window, indexes=1)
                dst_cls.write(cls2d, window=window, indexes=1)

                valid_mask = np.isfinite(mess)
                if np.any(valid_mask):
                    values = mess[valid_mask]
                    overall.update(values)
                    lim_valid = limiting[valid_mask]
                    neg_mask = values < 0
                    for idx in range(len(predictors)):
                        count = int(np.count_nonzero(lim_valid == idx))
                        limiting_total[idx] += count
                        if count:
                            limiting_negative[idx] += int(
                                np.count_nonzero((lim_valid == idx) & neg_mask)
                            )

            pbar.close()

        outputs["mess_raster"] = mess_path
        outputs["limiting_raster"] = limiting_path
        outputs["nt2_raster"] = nt2_path
        outputs["exdet_class_raster"] = class_path

    eval_corr = corr_acc.correlation_matrix()
    corr_path = out_dir / "mess_test_correlation.csv"
    corr_df = pd.DataFrame(eval_corr, index=predictors, columns=predictors)
    corr_df.to_csv(corr_path, index=True)
    outputs["evaluation_correlation"] = corr_path

    pair_records = []
    abs_changes: list[float] = []
    for i, var_i in enumerate(predictors):
        for j in range(i + 1, len(predictors)):
            var_j = predictors[j]
            base = float(training.correlation[i, j]) if training.correlation.size else float("nan")
            test_val = float(eval_corr[i, j]) if eval_corr.size else float("nan")
            change = test_val - base
            abs_change = abs(change) if np.isfinite(change) else float("nan")
            pair_records.append(
                {
                    "season": season_dir.name,
                    "test_year": year,
                    "run_type": run_label,
                    "var_i": var_i,
                    "var_j": var_j,
                    "training_corr": base,
                    "test_corr": test_val,
                    "corr_change": change,
                    "abs_corr_change": abs_change,
                }
            )
            if np.isfinite(abs_change):
                abs_changes.append(abs_change)

    corr_change_df = pd.DataFrame(pair_records)
    corr_change_path = out_dir / "mess_correlation_change.csv"
    corr_change_df.to_csv(corr_change_path, index=False)
    outputs["correlation_change"] = corr_change_path

    mean_abs_change = float(np.nanmean(abs_changes)) if abs_changes else float("nan")
    max_abs_change = float(np.nanmax(abs_changes)) if abs_changes else float("nan")

    fro_norm = float("nan")
    corr_diff_plot = np.empty((0, 0), dtype="float64")
    if (
        training.correlation.size
        and eval_corr.size
        and training.correlation.shape == eval_corr.shape
    ):
        diff = eval_corr - training.correlation
        mask = np.isfinite(diff)
        if np.any(mask):
            corr_diff_plot = np.where(mask, diff, np.nan)
            fro_norm = float(np.linalg.norm(diff[mask].ravel()) / math.sqrt(int(np.count_nonzero(mask))))

    total_cls = int(novelty_counts.sum())
    nt1_only = int(novelty_counts[1])
    nt2_only = int(novelty_counts[2])
    nt_both = int(novelty_counts[3])
    nt2_any = nt2_only + nt_both
    pixel_area_km2 = overall.pixel_area_km2

    quantiles = quantiles_from_hist(overall.hist, HIST_EDGES, [0.05, 0.25, 0.5, 0.75, 0.95])
    overall_summary = pd.DataFrame(
        [
            {
                "season": season_dir.name,
                "test_year": year,
                "run_type": run_label,
                "valid_cells": overall.count,
                "valid_area_km2": overall.area_km2,
                "mean_mess": overall.mean(),
                "std_mess": overall.std(),
                "min_mess": overall.min,
                "max_mess": overall.max,
                "p05_mess": quantiles[0],
                "p25_mess": quantiles[1],
                "median_mess": quantiles[2],
                "p75_mess": quantiles[3],
                "p95_mess": quantiles[4],
                "share_negative": overall.negative / overall.count if overall.count else np.nan,
                "share_below10": overall.below10 / overall.count if overall.count else np.nan,
                "share_below20": overall.below20 / overall.count if overall.count else np.nan,
                "share_above50": overall.above50 / overall.count if overall.count else np.nan,
                "novel_area_km2": overall.negative_area_km2,
                "mess_0_10_area_km2": overall.below10_area_km2,
                "mess_10_20_area_km2": overall.below20_area_km2,
                "mess_ge50_area_km2": overall.above50_area_km2,
                "mean_abs_corr_change": mean_abs_change,
                "max_abs_corr_change": max_abs_change,
                "fro_corr_change": fro_norm,
                "classified_cells": total_cls,
                "nt1_only_share_of_classified": nt1_only / total_cls if total_cls else np.nan,
                "nt2_only_share_of_classified": nt2_only / total_cls if total_cls else np.nan,
                "nt1_nt2_both_share_of_classified": nt_both / total_cls if total_cls else np.nan,
                "nt2_any_share_of_classified": nt2_any / total_cls if total_cls else np.nan,
                "nt1_only_cells": nt1_only,
                "nt2_only_cells": nt2_only,
                "nt1_nt2_both_cells": nt_both,
                "nt1_only_area_km2": nt1_only * pixel_area_km2,
                "nt2_only_area_km2": nt2_only * pixel_area_km2,
                "nt1_nt2_both_area_km2": nt_both * pixel_area_km2,
            }
        ]
    )
    overall_path = out_dir / "mess_overall_summary.csv"
    overall_summary.to_csv(overall_path, index=False)
    outputs["overall_summary"] = overall_path

    var_records = []
    for idx, acc in enumerate(per_var):
        if not acc.count:
            continue
        q = quantiles_from_hist(acc.hist, HIST_EDGES, [0.05, 0.25, 0.5, 0.75, 0.95])
        var_records.append(
            {
                "season": season_dir.name,
                "test_year": year,
                "run_type": run_label,
                "variable": acc.name,
                "valid_cells": acc.count,
                "mean_mj": acc.mean(),
                "std_mj": acc.std(),
                "min_mj": acc.min,
                "max_mj": acc.max,
                "p05_mj": q[0],
                "p25_mj": q[1],
                "median_mj": q[2],
                "p75_mj": q[3],
                "p95_mj": q[4],
                "outside_low_pct": acc.outside_low / acc.count if acc.count else np.nan,
                "outside_high_pct": acc.outside_high / acc.count if acc.count else np.nan,
                "negative_pct": acc.negative / acc.count if acc.count else np.nan,
                "below10_pct": acc.below10 / acc.count if acc.count else np.nan,
                "limiting_share": (
                    limiting_total[idx] / overall.count if overall.count else np.nan
                ),
                "limiting_negative_share": (
                    limiting_negative[idx] / overall.negative if overall.negative else np.nan
                ),
            }
        )

    var_summary = pd.DataFrame(var_records)
    var_path = out_dir / "mess_variable_summary.csv"
    var_summary.to_csv(var_path, index=False)
    outputs["variable_summary"] = var_path

    lim_df = pd.DataFrame(
        {
            "season": season_dir.name,
            "test_year": year,
            "run_type": run_label,
            "variable": predictors,
            "limiting_cells": limiting_total,
            "limiting_area_km2": limiting_total * overall.pixel_area_km2,
            "limiting_negative_cells": limiting_negative,
            "limiting_negative_area_km2": limiting_negative * overall.pixel_area_km2,
        }
    )
    lim_path = out_dir / "mess_limiting_variables.csv"
    lim_df.to_csv(lim_path, index=False)
    outputs["limiting_summary"] = lim_path

    mapping = {idx: name for idx, name in enumerate(predictors)}
    qml_path = out_dir / "mess_limiting_variable.qml"
    _write_limiting_qml(qml_path, mapping)
    outputs["limiting_qml"] = qml_path

    viz_outputs = _generate_visualizations(
        out_dir,
        run_label,
        season_dir.name,
        year,
        predictors,
        mapping,
        corr_diff_plot,
    )
    outputs.update(viz_outputs)

    novelty_dict = {
        "class_0_none": int(novelty_counts[0]),
        "class_1_nt1_only": nt1_only,
        "class_2_nt2_only": nt2_only,
        "class_3_nt1_nt2_both": nt_both,
    }
    metadata = {
        "season": season_dir.name,
        "test_year": year,
        "predictors": mapping,
        "test_stack": str(test_stack),
        "boundary_mask": str(boundary_mask_path) if has_boundary_mask else None,
        "run_type": run_label,
        "output_directory": str(out_dir),
        "pixel_area_km2": overall.pixel_area_km2,
        "nt2_quantile": training.nt2_quantile,
        "nt2_threshold_md2": training.md2_threshold,
        "novelty_class_counts": novelty_dict,
        "git_commit": _current_git_commit(),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "rasterio_version": rasterio.__version__,
        "references": [
            "Elith, Kearney & Phillips (2010) - The art of modelling range-shifting species",
            "Mesgaran, Cousens & Webber (2014) - Here be dragons: Quantifying novelty due to covariate range and correlation change",
        ],
        "visualizations": {key: str(path) for key, path in viz_outputs.items()},
    }
    meta_path = out_dir / "mess_metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2))
    outputs["metadata"] = meta_path

    return outputs


def collect_season_directories(exp_root: Path, run_dir: Path) -> list[Path]:
    output_dir = exp_root / "output"
    if not output_dir.exists():
        raise FileNotFoundError(f"Output directory not found: {output_dir}")

    if run_dir == exp_root or run_dir == output_dir:
        seasons = [d for d in output_dir.iterdir() if d.is_dir()]
    else:
        current = run_dir
        seasons = []
        while True:
            if current.parent == output_dir:
                seasons = [current]
                break
            if current == exp_root:
                seasons = [d for d in output_dir.iterdir() if d.is_dir()]
                break
            current = current.parent
    return seasons


def collect_test_years(season_dir: Path) -> list[int]:
    years: list[int] = []
    for child in season_dir.iterdir():
        if child.is_dir() and child.name.startswith("test_"):
            try:
                years.append(int(child.name.split("_")[1]))
            except (IndexError, ValueError):
                continue
    return sorted(set(years))


def _infer_first_year(run_dir: Path) -> int:
    for candidate in [run_dir, *run_dir.parents]:
        candidates = sorted(
            (p for p in candidate.glob("input-data-test-*") if p.is_dir()),
            key=lambda p: p.name,
        )
        for path in candidates:
            suffix = path.name.split("input-data-test-")[-1]
            if suffix.isdigit():
                return int(suffix)
    raise FileNotFoundError(
        f"Unable to infer a test year from {run_dir}; no 'input-data-test-<year>' directories found"
    )


@dataclass
class RunOutputs:
    run_type: str
    run_dir: Path
    experiment_root: Path
    output_root: Path
    run_output_dir: Path
    results: dict[tuple[str, int], dict[str, Path]]


def run_mess(
    run_dir: Path,
    *,
    years: Sequence[int] | None = None,
    season: str | None = None,
    compression: str = "LZW",
    run_type: str,
    output_root: Path | None = None,
    nt2_quantile: float = NT2_QUANTILE,
) -> RunOutputs:
    run_dir = run_dir.resolve()
    if years:
        if len(years) == 0:
            raise ValueError("At least one year must be provided when specifying --year")
        first_year = years[0]
    else:
        first_year = _infer_first_year(run_dir)
    exp_root = find_experiment_root(run_dir, first_year)
    seasons = collect_season_directories(exp_root, run_dir)
    if not seasons:
        raise FileNotFoundError(f"No season directories found under {run_dir}")
    if season:
        seasons = [s for s in seasons if s.name.lower() == season.lower()]
        if not seasons:
            raise ValueError(f"Season '{season}' not found for run {run_dir}")

    available_years = {}
    for season_dir in seasons:
        available_years[season_dir.name] = collect_test_years(season_dir)

    if years is None:
        target_years = sorted({year for vals in available_years.values() for year in vals})
    else:
        target_years = sorted(set(years))

    if output_root is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_parent = exp_root / "mess_outputs"
        final_output_root = base_parent / f"mess_{timestamp}"
    else:
        final_output_root = output_root.resolve()
    final_output_root.mkdir(parents=True, exist_ok=True)
    run_output_dir = final_output_root / run_type
    run_output_dir.mkdir(parents=True, exist_ok=True)

    results: dict[tuple[str, int], dict[str, Path]] = {}
    for season_dir in seasons:
        season_years = [y for y in target_years if y in available_years.get(season_dir.name, [])]
        for year in season_years:
            features_dir = exp_root / f"input-data-test-{year}"
            if not features_dir.exists():
                raise FileNotFoundError(f"Features directory missing: {features_dir}")
            test_dir = season_dir / f"test_{year}"
            stack_dir = features_dir / f"{season_dir.name}_stack"
            if not stack_dir.exists():
                stack_dir = test_dir / "test_stack"
            predictors = read_predictors(stack_dir)
            training = load_training_context(
                features_dir,
                season_dir.name,
                predictors,
                nt2_quantile=nt2_quantile,
            )
            if len(training.predictors) != len(predictors):
                missing = sorted(set(predictors) - set(training.predictors))
                if missing:
                    logging.error(
                        "Skipping %s missing predictor(s) for %s %s: %s",
                        len(missing),
                        season_dir.name,
                        year,
                        ", ".join(missing),
                    )
            predictors = training.predictors
            out_dir = run_output_dir / season_dir.name / f"test_{year}"
            outputs = process_season_year(
                season_dir,
                test_dir,
                year,
                predictors,
                training,
                out_dir,
                run_label=run_type,
                compression=compression,
            )
            results[(season_dir.name, year)] = outputs
    if not results:
        raise FileNotFoundError(
            "No matching season/year combinations found for the provided configuration"
        )
    return RunOutputs(run_type, run_dir, exp_root, final_output_root, run_output_dir, results)


def _read_overall_summary(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    numeric_cols = df.select_dtypes(include=["number"]).columns
    df[numeric_cols] = df[numeric_cols].astype("float64")
    return df


def compare_runs(
    mono: RunOutputs, multi: RunOutputs, *, output_root: Path | None = None
) -> dict[tuple[str, int], Path]:
    if mono.experiment_root != multi.experiment_root:
        log.warning(
            "Mono run (%s) and multi run (%s) report different experiment roots; "
            "continuing with shared comparison outputs",
            mono.experiment_root,
            multi.experiment_root,
        )

    common = sorted(set(mono.results) & set(multi.results))
    if not common:
        raise ValueError("No overlapping season/year combinations found between runs")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if output_root is None:
        if mono.experiment_root == multi.experiment_root:
            comparison_base = mono.experiment_root / "mess_comparisons"
        else:
            try:
                shared_root = Path(
                    os.path.commonpath(
                        [str(mono.experiment_root), str(multi.experiment_root)]
                    )
                )
            except ValueError:
                shared_root = Path.cwd()
            anchor_path = Path(shared_root.anchor) if shared_root.anchor else None
            if anchor_path and shared_root == anchor_path:
                shared_root = Path.cwd()
            comparison_base = shared_root / "mess_comparisons"
        comparison_dir = comparison_base / f"mess_{timestamp}"
    else:
        comparison_dir = output_root / "comparisons"
    comparison_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[tuple[str, int], Path] = {}

    aggregate_rows = []
    for key in common:
        mono_summary = _read_overall_summary(mono.results[key]["overall_summary"]).assign(
            run_type=mono.run_type
        )
        multi_summary = _read_overall_summary(multi.results[key]["overall_summary"]).assign(
            run_type=multi.run_type
        )
        combined = pd.concat([mono_summary, multi_summary], ignore_index=True)
        numeric_cols = combined.select_dtypes(include=["number"]).columns
        diff_row = {col: np.nan for col in combined.columns}
        diff_row["run_type"] = f"{multi.run_type}_minus_{mono.run_type}"
        diff_row["season"] = combined["season"].iloc[0]
        diff_row["test_year"] = combined["test_year"].iloc[0]
        for col in numeric_cols:
            if col in {"season", "test_year"}:
                continue
            mono_val = float(mono_summary.iloc[0][col]) if col in mono_summary else np.nan
            multi_val = float(multi_summary.iloc[0][col]) if col in multi_summary else np.nan
            if np.isfinite(mono_val) and np.isfinite(multi_val):
                diff_row[col] = multi_val - mono_val
        combined = pd.concat([combined, pd.DataFrame([diff_row])], ignore_index=True)
        out_path = comparison_dir / f"{key[0]}_test_{key[1]}_comparison.csv"
        combined.to_csv(out_path, index=False)
        outputs[key] = out_path
        aggregate_rows.append(combined)

    all_df = pd.concat(aggregate_rows, ignore_index=True)
    all_df.to_csv(comparison_dir / "mess_comparison_overview.csv", index=False)
    return outputs


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute MESS diagnostics for test years")
    parser.add_argument("--run-dir", type=Path, help="Path to a single run folder for MESS computation")
    parser.add_argument("--mono-run", type=Path, help="Path to the monotemporal run folder")
    parser.add_argument("--multi-run", type=Path, help="Path to the multitemporal run folder")
    parser.add_argument(
        "--year",
        type=int,
        action="append",
        help="Restrict processing to the specified test year(s). Repeat for multiple years.",
    )
    parser.add_argument("--season", help="Optional season filter (case insensitive)")
    parser.add_argument(
        "--compression",
        default="LZW",
        choices=["LZW", "DEFLATE", "ZSTD", "NONE"],
        help="Compression used for the output rasters",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity",
    )
    parser.add_argument(
        "--nt2-quantile",
        type=float,
        default=NT2_QUANTILE,
        help="Quantile of training MD² used to normalise NT2 (default: 0.99)",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        help="Base directory where timestamped MESS outputs will be stored",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.run_dir and (args.mono_run or args.multi_run):
        raise ValueError("Specify either --run-dir or both --mono-run/--multi-run, not both")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_parent = (args.output_root or Path.cwd() / "mess_outputs").resolve()
    session_root = base_parent / f"mess_{timestamp}"
    params = vars(args).copy()
    params["timestamp"] = timestamp
    params["session_root"] = str(session_root)

    with capture_console(session_root, params):
        log_level = getattr(logging, args.log_level.upper(), logging.INFO)
        root_logger = logging.getLogger()
        root_logger.handlers.clear()
        root_logger.setLevel(log_level)
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        file_handler = logging.FileHandler(session_root / "mess.log", encoding="utf-8")
        file_handler.setFormatter(formatter)
        root_logger.addHandler(stream_handler)
        root_logger.addHandler(file_handler)

        log.info("MESS outputs will be stored in %s", session_root)
        if args.mono_run or args.multi_run:
            if not (args.mono_run and args.multi_run):
                raise ValueError("Both --mono-run and --multi-run must be provided for comparisons")
            mono_outputs = run_mess(
                args.mono_run,
                years=args.year,
                season=args.season,
                compression=args.compression,
                run_type="monotemporal",
                output_root=session_root,
                nt2_quantile=args.nt2_quantile,
            )
            multi_outputs = run_mess(
                args.multi_run,
                years=args.year,
                season=args.season,
                compression=args.compression,
                run_type="multitemporal",
                output_root=session_root,
                nt2_quantile=args.nt2_quantile,
            )
            compare_paths = compare_runs(mono_outputs, multi_outputs, output_root=session_root)
            for (season, year), path in sorted(compare_paths.items()):
                log.info("Comparison summary %s %s: %s", season, year, path)
        elif args.run_dir:
            run_dir = args.run_dir.resolve()
            log.info("Starting MESS evaluation for %s", run_dir)
            outputs = run_mess(
                run_dir,
                years=args.year,
                season=args.season,
                compression=args.compression,
                run_type="single",
                output_root=session_root,
                nt2_quantile=args.nt2_quantile,
            )
            for (season, year), paths in outputs.results.items():
                log.info("Finished %s %s", season, year)
                for label, path in paths.items():
                    log.info("  %s: %s", label, path)
            log.info("All outputs written to %s", outputs.output_root)
        else:
            raise ValueError("Either --run-dir or both --mono-run/--multi-run must be supplied")


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()

