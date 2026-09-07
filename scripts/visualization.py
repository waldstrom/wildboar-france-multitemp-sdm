# =============================================================================
# Project Title: Wild Boar Distribution Model for France
# Path: scripts/visualization.py
# Purpose: Provide simple plotting utilities for pipeline diagnostics.
# Process Step: Generates figures for occurrences, variables and model outputs.
# Created by Adrian Meyer, Institute Geomatics, FHNW (adrian.meyer@fhnw.ch)
# Created: July 2025
# Version: v.0.1.0
# =============================================================================
"""Simple plotting utilities for pipeline diagnostics."""
from __future__ import annotations
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import geopandas as gpd
import rasterio
import rasterio.plot
from shapely.affinity import translate
from scripts.utils import validate_crs
import numpy as np
import xarray as xr
import pandas as pd
from typing import Iterable


def plot_points(
    gdf: gpd.GeoDataFrame,
    out_file: Path,
    title: str = "",
    markersize: float = 1.0,
) -> None:
    """Plot point ``gdf`` to ``out_file``.

    ``markersize`` can be adjusted to handle large datasets such as
    pseudo-absence points.
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    gdf.plot(ax=ax, markersize=markersize, color="red")
    ax.set_axis_off()
    ax.set_title(title)
    fig.savefig(out_file, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_variables(stack_path: Path, vars_used: list[str], out_file: Path) -> None:
    ds = xr.open_zarr(stack_path, consolidated=False)[vars_used]
    n = len(vars_used)
    cols = min(5, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
    axes = np.atleast_2d(axes)
    for ax, var in zip(axes.flat, vars_used):
        data = ds[var].values
        valid = data[~np.isnan(data)]
        vmin = float(np.nanmin(valid)) if valid.size else 0.0
        vmax = float(np.nanmax(valid)) if valid.size else 1.0
        im = ax.imshow(data, cmap="magma", vmin=vmin, vmax=vmax)
        im.cmap.set_bad("darkgreen")
        ax.set_title(var)
        ax.set_axis_off()
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.set_ylabel(f"{vmin:.2f} - {vmax:.2f}")
    for ax in axes.flat[n:]:
        ax.remove()
    plt.tight_layout()
    fig.savefig(out_file, dpi=150)
    plt.close(fig)


def plot_raster(tif: Path, out_file: Path, title: str = "") -> None:
    with rasterio.open(tif) as src:
        validate_crs(src.crs, context=f"raster {tif}")
        data = src.read(1).astype("float32")
        nodata = src.nodata if src.nodata is not None else -9999
        data[data == nodata] = np.nan
    valid = data[~np.isnan(data)]
    vmin = float(np.nanmin(valid)) if valid.size else 0.0
    vmax = float(np.nanmax(valid)) if valid.size else 1.0
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(data, cmap="magma", vmin=vmin, vmax=vmax)
    im.cmap.set_bad("darkgreen")
    ax.set_axis_off()
    ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel(f"{vmin:.2f} - {vmax:.2f}")
    fig.savefig(out_file, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_outline(ax, geom, **kwargs) -> None:
    """Plot polygon or multipolygon outline on ``ax``."""

    if getattr(geom, "geom_type", "") == "Polygon":
        ax.plot(*geom.exterior.xy, **kwargs)
    else:
        for part in getattr(geom, "geoms", [geom]):
            ax.plot(*part.exterior.xy, **kwargs)


def plot_cloglog_points(
    tif: Path,
    gdf: gpd.GeoDataFrame,
    bounds,
    mapping: Iterable,
    out_file: Path,
    title: str = "",
    markersize: float = 2.0,
    edgecolor: str = "white",
    label_years: bool = True,
) -> None:
    """Overlay ``gdf`` on ``tif`` and annotate year labels."""

    with rasterio.open(tif) as src:
        validate_crs(src.crs, context=f"raster {tif}")
        data = src.read(1).astype("float32")
        nodata = src.nodata if src.nodata is not None else -9999
        data[data == nodata] = np.nan
        extent = rasterio.plot.plotting_extent(src)
    valid = data[~np.isnan(data)]
    vmin = float(np.nanmin(valid)) if valid.size else 0.0
    vmax = float(np.nanmax(valid)) if valid.size else 1.0
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(data, cmap="magma", vmin=vmin, vmax=vmax, extent=extent)
    im.cmap.set_bad("darkgreen")
    gdf.plot(
        ax=ax,
        markersize=markersize,
        marker="o",
        facecolor="none",
        edgecolor=edgecolor,
        linewidth=0.5,
    )
    for yo in mapping:
        b = translate(bounds, xoff=getattr(yo, "dx", 0.0), yoff=getattr(yo, "dy", 0.0))
        _plot_outline(ax, b, color="grey", linewidth=0.5)
        if label_years:
            minx, miny, maxx, maxy = b.bounds
            ax.text(
                minx,
                maxy,
                str(getattr(yo, "year", "")),
                ha="left",
                va="top",
                fontsize=8,
                color="white",
            )
    ax.set_axis_off()
    ax.set_aspect("equal")
    ax.set_title(title)
    fig.savefig(out_file, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_variable_histograms(
    pres_df,
    bg_df,
    vars_used: list[str],
    out_file: Path,
    colors: tuple[str, str] = ("#FFCC02", "#00458C"),
) -> None:
    """Plot histograms of presence vs background values."""

    n = len(vars_used)
    cols = min(5, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
    axes = np.atleast_2d(axes)
    for ax, var in zip(axes.flat, vars_used):
        pvar = pres_df[var]
        bvar = bg_df[var]
        ax.hist(
            [pvar.dropna(), bvar.dropna()],
            density=True,
            alpha=0.7,
            label=["presence", "background"],
            color=list(colors),
        )
        ax.set_title(var)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc=(0.6, 0.9))
    plt.tight_layout()
    fig.savefig(out_file, dpi=150)
    plt.close(fig)


def export_response_curves(model, df: pd.DataFrame, vars_used: list[str], out_dir: Path) -> None:
    """Save response curves for each variable as CSV and PNG."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    base = df.median().values
    for i, var in enumerate(vars_used):
        vals = np.linspace(df[var].min(), df[var].max(), 50)
        X = np.tile(base, (50, 1))
        X[:, i] = vals
        preds = model.predict(X)
        pd.DataFrame({"value": vals, "response": preds}).to_csv(out_dir / f"{var}_response.csv", index=False)
        plt.figure(figsize=(4, 3))
        plt.plot(vals, preds)
        plt.xlabel(var)
        plt.ylabel("Suitability")
        plt.tight_layout()
        plt.savefig(out_dir / f"{var}_response.png", dpi=150)
        plt.close()


def compile_response_curves_pdf(curve_dir: Path, out_file: Path, max_cols: int = 5) -> None:
    """Combine individual response curve PNGs into a single PDF with small multiples.

    Parameters
    ----------
    curve_dir : Path
        Directory containing ``*_response.png`` files.
    out_file : Path
        Path to the PDF file to create.
    max_cols : int, optional
        Maximum number of columns in the grid layout, by default 5.
    """

    curve_dir = Path(curve_dir)
    pngs = sorted(curve_dir.glob("*_response.png"))
    if not pngs:
        return

    n = len(pngs)
    cols = min(max_cols, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows))
    axes = np.atleast_2d(axes)
    for ax, png in zip(axes.flat, pngs):
        img = plt.imread(png)
        ax.imshow(img)
        ax.set_axis_off()
        ax.set_title(png.stem.replace("_response", ""))
    for ax in axes.flat[n:]:
        ax.remove()
    plt.tight_layout()
    fig.savefig(out_file, dpi=150)
    plt.close(fig)


def plot_dropped_points(gdf: gpd.GeoDataFrame, out_file: Path) -> None:
    """Visualise records removed due to missing predictor values."""

    if gdf.empty:
        return
    plot_points(gdf, out_file, "Dropped records")
