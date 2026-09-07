# =============================================================================
# Project Title: Wild Boar Distribution Model for France
# Path: scripts/correlation_analysis.py
# Purpose: Perform full pairwise feature correlation analysis and rank variables.
# Process Step: Generates correlation matrices across all years and per-year.
# Created by OpenAI's Codex.
# =============================================================================
"""Full pairwise correlation analysis of all predictor variables.

This script loads every ``.asc`` and ``.tif`` raster located under the
``data`` directory, clips the pixels to the study area defined by
``data/boundary_france.asc`` and performs a pairwise correlation analysis.
Both a global correlation matrix (all years combined) and individual matrices
per year are generated.  Results are written to CSV files and correlation
matrix plots.
"""
from __future__ import annotations

import argparse
import logging
import math
import re
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from scripts.utils import validate_crs

try:  # fail fast if optional dependencies are missing
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import rasterio
    import rasterio.warp
except ModuleNotFoundError as exc:  # pragma: no cover - env specific
    sys.stderr.write(
        f"Missing required module: {exc.name}. Please install dependencies.\n"
    )
    raise

# ---------------------------------------------------------------------------
# Configuration parameters (edit as needed)
# ---------------------------------------------------------------------------
DEFAULT_OUT_DIR = "correlation"
DEFAULT_SAMPLE_FRACTION = 1.0  # Use all boundary pixels
DEFAULT_RANDOM_SEED = 0
LOG_FILE = "correlation_analysis.log"
DEFAULT_CRS = rasterio.crs.CRS.from_epsg(2154)

# Paths to input data
BOUNDARY_FILE = Path("data/boundary_france.asc")

# Map feature directories to logical domains
DOMAIN_MAP: dict[Path, str] = {
    Path("data/bioclim"): "bioclim",
    Path("data/dem"): "auxiliary",
    Path("data/hunting-bag"): "auxiliary",
    Path("data/special"): "auxiliary",
    Path("data/theia-special"): "auxiliary",
    Path("data/lc-distances"): "land_cover",
    Path("data/land-cover"): "land_cover",
    Path("data/sat-indices"): "satellite",
    Path("data/era5"): "era5",
}

# Flatten list of feature directories for loading and derive domain list
FEATURE_DIRS = list(DOMAIN_MAP.keys())
ALL_DOMAINS = sorted(set(DOMAIN_MAP.values()) | {"auxiliary"})

# A module level logger is created but the log level/handlers are configured
# inside ``main`` so that CLI options can influence the verbosity.  This allows
# debug statements deep in the code to be enabled with ``--log-level DEBUG``.
log = logging.getLogger(__name__)


def _ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


# -----------------------------------------------------------------------------

def _load_mask() -> tuple[np.ndarray, rasterio.Affine, rasterio.crs.CRS]:
    """Load study area mask and return mask, transform and CRS."""
    if not BOUNDARY_FILE.exists():
        log.error("Boundary file %s does not exist", BOUNDARY_FILE)
        raise FileNotFoundError(BOUNDARY_FILE)

    with rasterio.open(BOUNDARY_FILE) as src:
        mask = src.read(1, masked=True) == 1
        transform = src.transform
        crs = src.crs or DEFAULT_CRS

    log.debug(
        "Loaded mask %s with shape=%s, valid pixels=%d",
        BOUNDARY_FILE,
        mask.shape,
        int(mask.sum()),
    )

    # reproject boundary to DEFAULT_CRS if needed
    if crs.to_epsg() != DEFAULT_CRS.to_epsg():
        log.info("Reprojecting boundary mask from %s to %s", crs, DEFAULT_CRS)
        dst = np.empty_like(mask, dtype="float32")
        rasterio.warp.reproject(
            mask.astype("float32"),
            dst,
            src_transform=transform,
            src_crs=crs,
            dst_transform=transform,
            dst_crs=DEFAULT_CRS,
            resampling=rasterio.warp.Resampling.nearest,
        )
        mask = dst.astype(bool)
        crs = DEFAULT_CRS

    return mask, transform, crs


def _load_features(
    mask: np.ndarray,
    transform: rasterio.Affine,
    crs: rasterio.crs.CRS,
) -> tuple[pd.DataFrame, dict[str, str]]:
    """Load all raster features in parallel and clip to ``mask``.

    Returns
    -------
    DataFrame
        Table where each column represents a feature raster clipped to the
        study area.
    dict
        Mapping of column names to their logical domain ("auxiliary",
        "satellite", "land_cover").
    """

    files: list[Path] = []
    for root in FEATURE_DIRS:
        files.extend(sorted(root.rglob("*.asc")))
        files.extend(sorted(root.rglob("*.tif")))

    log.info("Found %d feature rasters", len(files))
    if not files:
        log.warning("No feature rasters found under %s", FEATURE_DIRS)

    def _get_domain(f: Path) -> str:
        for dir_path, domain in DOMAIN_MAP.items():
            if dir_path in f.parents:
                return domain
        return "auxiliary"

    def _load_one(f: Path) -> tuple[str, np.ndarray, str] | None:
        try:
            with rasterio.open(f) as src:
                src_crs = validate_crs(src.crs, context=f"raster {f}")
                data = src.read(1).astype("float32")
                nodata = src.nodata
                if nodata is not None:
                    data[data == nodata] = np.nan

                if (
                    src_crs != crs
                    or src.transform != transform
                    or src.width != mask.shape[1]
                    or src.height != mask.shape[0]
                ):
                    log.info("Reprojecting %s to %s", f, crs)
                    dest = np.full(mask.shape, np.nan, dtype="float32")
                    rasterio.warp.reproject(
                        data,
                        dest,
                        src_transform=src.transform,
                        src_crs=src_crs,
                        dst_transform=transform,
                        dst_crs=crs,
                        resampling=rasterio.warp.Resampling.nearest,
                    )
                    data = dest

                masked_data = data[mask]
                log.debug(
                    "Loaded %s with %d valid pixels", f.name, np.count_nonzero(~np.isnan(masked_data))
                )
                return f.stem, masked_data, _get_domain(f)
        except Exception as exc:  # pragma: no cover - env specific
            log.warning("Failed loading %s: %s", f, exc)
            return None

    arrays: dict[str, np.ndarray] = {}
    domains: dict[str, str] = {}
    with ThreadPoolExecutor() as ex:
        for res in ex.map(_load_one, files):
            if res:
                name, arr, domain = res
                arrays[name] = arr
                domains[name] = domain

    if not arrays:
        log.warning("No features were successfully loaded")

    df = pd.DataFrame(arrays)
    # drop rows that have no valid values at all; pandas correlation will
    # automatically ignore missing values on a pairwise basis
    df = df.dropna(how="all")
    log.debug("Feature dataframe shape=%s", df.shape)
    if df.empty:
        for key, arr in arrays.items():
            log.debug(
                "Feature %s valid count=%d", key, np.count_nonzero(~np.isnan(arr))
            )
    return df, domains


# -----------------------------------------------------------------------------

def _compute_uniqueness(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return absolute correlation matrix and ranked uniqueness dataframe."""
    corr = df.corr().abs()
    max_corr = corr.where(~np.eye(len(corr), dtype=bool)).max()
    uniq = 1 - max_corr
    ranked = uniq.sort_values(ascending=False).reset_index()
    ranked.columns = ["variable", "uniqueness"]
    return corr, ranked


# -----------------------------------------------------------------------------

def _plot_matrix(
    corr: pd.DataFrame,
    out_file: Path,
    title: str,
    dpi: int = 300,
    font_size: int = 6,
) -> None:
    """Save correlation matrix plot."""
    plt.figure(figsize=(10, 8))
    plt.imshow(corr, cmap="viridis", vmin=0, vmax=1)
    plt.colorbar(label="|corr|")
    plt.title(title)
    ticks = range(len(corr))
    plt.xticks(ticks, corr.columns, rotation=90, fontsize=font_size)
    plt.yticks(ticks, corr.index, fontsize=font_size)
    plt.tight_layout()
    plt.savefig(out_file, dpi=dpi)
    plt.close()


def _plot_uniqueness(
    ranks: pd.DataFrame,
    out_file: Path,
    title: str,
    dpi: int = 300,
    font_size: int = 6,
) -> None:
    """Save bar plot of feature uniqueness."""
    plt.figure(figsize=(max(4, 0.25 * len(ranks)), 4))
    plt.bar(range(len(ranks)), ranks["uniqueness"])
    plt.xticks(range(len(ranks)), ranks["variable"], rotation=90, fontsize=font_size)
    plt.ylabel("uniqueness")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_file, dpi=dpi)
    plt.close()

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pairwise feature correlation analysis"
    )
    parser.add_argument(
        "--out-dir",
        default=DEFAULT_OUT_DIR,
        help="Directory to store correlation results",
    )
    parser.add_argument(
        "--sample-fraction",
        type=float,
        default=DEFAULT_SAMPLE_FRACTION,
        help="Fraction of boundary cells to sample",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=DEFAULT_RANDOM_SEED,
        help="Random seed for sampling",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging verbosity",
    )
    args = parser.parse_args()
    logging.getLogger().handlers.clear()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(LOG_FILE, mode="w"),
            logging.StreamHandler(sys.stdout),
        ],
    )

    out_dir = _ensure_dir(args.out_dir)
    log.info("Loading boundary mask from %s", BOUNDARY_FILE)
    mask, transform, crs = _load_mask()
    log.info("Loading raster features")
    df, domains = _load_features(mask, transform, crs)
    if args.sample_fraction < 1.0:
        df = df.sample(frac=args.sample_fraction, random_state=args.random_seed)
    log.debug("Dataframe after sampling shape=%s", df.shape)
    if df.empty:
        log.error(
            "Sampled dataframe is empty. Check boundary mask and feature rasters "
            "for alignment or missing data."
        )
        return

    log.info("Computing overall correlations on %d samples", len(df))
    corr_all, ranks_all = _compute_uniqueness(df)
    corr_all.to_csv(out_dir / "correlation_all.csv")
    ranks_all.to_csv(out_dir / "uniqueness_all.csv", index=False)
    _plot_matrix(corr_all, out_dir / "correlation_all.png", "All years")
    _plot_uniqueness(ranks_all, out_dir / "uniqueness_all.png", "All years")

    # extract year numbers that may appear anywhere in a column name
    # a year is identified as a four digit string starting with "20"
    year_pat = re.compile(r"(20\d{2})")
    years = sorted({m.group(1) for col in df.columns if (m := year_pat.search(col))})

    for year in years:
        log.info("Computing domain correlations for year %s", year)
        domain_cols: dict[str, list[str]] = {d: [] for d in ALL_DOMAINS}
        for col in df.columns:
            m = year_pat.search(col)
            col_year = m.group(1) if m else None
            if col_year == year or col_year is None:
                domain = domains.get(col, "auxiliary")
                if domain == "era5":
                    domain += "_anom" if "ANOM" in col.upper() else "_normal"
                domain_cols.setdefault(domain, []).append(col)

        valid_domains = {d: c for d, c in domain_cols.items() if len(c) >= 2}
        if valid_domains:
            n_domains = len(valid_domains)
            ncols = min(3, n_domains)
            nrows = math.ceil(n_domains / ncols)
            fig, axes = plt.subplots(
                nrows, ncols, figsize=(5 * ncols, 5 * nrows), constrained_layout=True
            )
            axes = np.array(axes).flatten()

            images = []
            for ax, (domain, cols) in zip(axes, sorted(valid_domains.items())):
                corr, ranks = _compute_uniqueness(df[cols])
                corr.to_csv(out_dir / f"correlation_{year}_{domain}.csv")
                ranks.to_csv(out_dir / f"uniqueness_{year}_{domain}.csv", index=False)
                _plot_uniqueness(
                    ranks,
                    out_dir / f"uniqueness_{year}_{domain}.png",
                    f"{year} {domain.replace('_', ' ')}",
                )
                im = ax.imshow(corr, cmap="viridis", vmin=0, vmax=1)
                images.append(im)
                ax.set_title(domain.replace("_", " ").title())
                ticks = range(len(corr))
                ax.set_xticks(ticks)
                ax.set_xticklabels(corr.columns, rotation=90, fontsize=6)
                ax.set_yticks(ticks)
                ax.set_yticklabels(corr.index, fontsize=6)

            # remove any unused subplots
            for ax in axes[len(valid_domains):]:
                fig.delaxes(ax)

            cbar = fig.colorbar(
                images[-1],
                ax=axes[: len(valid_domains)],
                orientation="horizontal",
                fraction=0.02,
                pad=0.05,
                location="top",
            )
            cbar.set_label("|corr|", fontsize=5)
            cbar.ax.tick_params(labelsize=4)
            fig.suptitle(f"Domain-specific correlogram {year}", y=1.02)
            fig.savefig(out_dir / f"correlation_{year}_domains.png", dpi=300)
            plt.close(fig)
        else:
            log.warning("No domains with sufficient features for year %s", year)

        # cross-domain comparison across all features for the year
        all_cols = [c for cols in domain_cols.values() for c in cols]
        if len(all_cols) >= 2:
            corr, ranks = _compute_uniqueness(df[all_cols])
            corr.to_csv(out_dir / f"correlation_{year}_crossdomain.csv")
            ranks.to_csv(out_dir / f"uniqueness_{year}_crossdomain.csv", index=False)
            _plot_matrix(
                corr,
                out_dir / f"correlation_{year}_crossdomain.png",
                f"{year} cross-domain",
                dpi=1000,
                font_size=3,
            )
            _plot_uniqueness(
                ranks,
                out_dir / f"uniqueness_{year}_crossdomain.png",
                f"{year} cross-domain",
                font_size=3,
            )
        else:
            log.warning("Not enough features for cross-domain comparison in year %s", year)

# ----------------------------------------------------------------------
if __name__ == "__main__":
    main()
