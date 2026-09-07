"""Utility script to fix raster alignment issues.

This script performs two data correction steps that are required to
harmonise the rasters stored in the repository:

* GeoTIFF rasters (satellite indices) are shifted 500 m west and 500 m
  north so that they align with the other layers.
* ASCII grid rasters get their pixels outside the boundary (where the
  boundary layer equals 0) set to the nodata value (-9999 by default).

Both operations are executed in-place by rewriting the affected files.
The script can be run in dry-run mode to review the planned changes.
"""

from __future__ import annotations

import argparse
import logging
import math
import os
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import rasterio
from rasterio.transform import Affine


LOGGER = logging.getLogger(__name__)


def parse_arguments() -> argparse.Namespace:
    """Create the command line parser and return parsed arguments."""

    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--tiff-root",
        type=Path,
        action="append",
        default=[],
        help=(
            "Directory that contains GeoTIFF rasters to shift. The option can be "
            "specified multiple times. All *.tif and *.tiff files found recursively "
            "will be processed."
        ),
    )
    parser.add_argument(
        "--tiff",
        type=Path,
        action="append",
        default=[],
        help="Explicit GeoTIFF files to shift in addition to the --tiff-root entries.",
    )
    parser.add_argument(
        "--asc-root",
        type=Path,
        action="append",
        default=[],
        help=(
            "Directory that contains ASCII grid rasters to update. The option can be "
            "specified multiple times. All *.asc files found recursively will be "
            "processed."
        ),
    )
    parser.add_argument(
        "--asc",
        type=Path,
        action="append",
        default=[],
        help="Explicit ASCII grid files to update in addition to the --asc-root entries.",
    )
    parser.add_argument(
        "--boundary",
        type=Path,
        help=(
            "Boundary raster in ASCII grid format. All processed ASCII layers will have "
            "cells that are zero in this file set to the nodata value."
        ),
    )
    parser.add_argument(
        "--nodata",
        type=float,
        default=-9999.0,
        help="Nodata value to assign to pixels outside the boundary (default: -9999).",
    )
    parser.add_argument(
        "--shift-west",
        type=float,
        default=500.0,
        help="Distance in the raster units by which GeoTIFFs are shifted to the west.",
    )
    parser.add_argument(
        "--shift-north",
        type=float,
        default=500.0,
        help="Distance in the raster units by which GeoTIFFs are shifted to the north.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only report the planned changes without modifying any files.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging output.",
    )

    args = parser.parse_args()

    if args.asc or args.asc_root:
        if args.boundary is None:
            parser.error("--boundary is required when ASCII grids are specified.")

    return args


def configure_logging(verbose: bool) -> None:
    """Configure the root logger."""

    level = logging.INFO if verbose else logging.WARNING
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")


def collect_files(paths: Iterable[Path], patterns: Sequence[str]) -> list[Path]:
    """Collect files that match the provided glob patterns."""

    collected: set[Path] = set()
    for path in paths:
        if path is None:
            continue
        path = path.expanduser().resolve()
        if path.is_file():
            collected.add(path)
            continue
        if path.is_dir():
            for pattern in patterns:
                collected.update(p.resolve() for p in path.rglob(pattern))
        else:
            LOGGER.warning("Path does not exist and will be skipped: %s", path)

    return sorted(collected)


def transforms_match(first: Affine, second: Affine, tolerance: float = 1e-6) -> bool:
    """Return whether both affine transforms are equal within a tolerance."""

    for a, b in zip(first[:6], second[:6]):
        if not math.isclose(a, b, rel_tol=0.0, abs_tol=tolerance):
            return False
    return True


def shift_geotiffs(paths: Sequence[Path], shift_west: float, shift_north: float, dry_run: bool) -> None:
    """Shift GeoTIFF rasters by updating their transform."""

    if not paths:
        LOGGER.info("No GeoTIFF rasters to update.")
        return

    LOGGER.info("Shifting %s GeoTIFF(s) by %.3f m west and %.3f m north.", len(paths), shift_west, shift_north)

    for path in paths:
        with rasterio.open(path) as src:
            transform = src.transform
            new_transform = Affine(
                transform.a,
                transform.b,
                transform.c - shift_west,
                transform.d,
                transform.e,
                transform.f + shift_north,
            )

            if transforms_match(transform, new_transform):
                LOGGER.info("%s already aligned, skipping.", path)
                continue

            LOGGER.info("%s: origin (%.3f, %.3f) -> (%.3f, %.3f)",
                        path,
                        transform.c,
                        transform.f,
                        new_transform.c,
                        new_transform.f)

            if dry_run:
                continue

            profile = src.profile.copy()
            profile.update(transform=new_transform)
            data = src.read()
            tags = src.tags()
            band_tags = [src.tags(i) for i in range(1, src.count + 1)]
            descriptions = list(src.descriptions)
            colormaps = []
            for band_index in range(1, src.count + 1):
                try:
                    colormaps.append(src.colormap(band_index))
                except ValueError:
                    colormaps.append(None)

        temp_path = path.with_suffix(path.suffix + ".tmp")
        if temp_path.exists():
            temp_path.unlink()
        with rasterio.open(temp_path, "w", **profile) as dst:
            dst.write(data)
            if tags:
                dst.update_tags(**tags)
            for band_index, band_tag in enumerate(band_tags, start=1):
                if descriptions[band_index - 1]:
                    dst.set_band_description(band_index, descriptions[band_index - 1])
                if band_tag:
                    dst.update_tags(band_index, **band_tag)
                if colormaps[band_index - 1]:
                    dst.write_colormap(band_index, colormaps[band_index - 1])

        os.replace(temp_path, path)


def determine_target_dtype(dtype_name: str, nodata_value: float) -> str:
    """Return a rasterio compatible dtype that can store the nodata value."""

    dtype = np.dtype(dtype_name)
    if np.issubdtype(dtype, np.floating):
        return dtype_name

    if np.issubdtype(dtype, np.unsignedinteger):
        return "int32"

    if np.issubdtype(dtype, np.integer):
        min_value = np.iinfo(dtype).min
        if nodata_value < min_value:
            return "int32"
        return dtype_name

    # Fallback to float32 for other dtypes (e.g. bool).
    return "float32"


def update_ascii_layers(
    asc_paths: Sequence[Path],
    boundary_path: Path,
    nodata_value: float,
    dry_run: bool,
) -> None:
    """Set nodata outside of the boundary for ASCII rasters."""

    if not asc_paths:
        LOGGER.info("No ASCII rasters to update.")
        return

    with rasterio.open(boundary_path) as boundary_src:
        boundary = boundary_src.read(1, masked=False)
        boundary_transform = boundary_src.transform
        boundary_shape = boundary.shape

    LOGGER.info(
        "Updating %s ASCII grid(s) using boundary %s (nodata %.1f).",
        len(asc_paths),
        boundary_path,
        nodata_value,
    )

    mask = boundary == 0

    for path in asc_paths:
        if path.resolve() == boundary_path.resolve():
            LOGGER.info("Skipping boundary raster itself: %s", path)
            continue

        with rasterio.open(path) as src:
            if (src.width, src.height) != (boundary_shape[1], boundary_shape[0]):
                LOGGER.warning(
                    "%s has shape %s which differs from the boundary %s; skipping.",
                    path,
                    (src.width, src.height),
                    (boundary_shape[1], boundary_shape[0]),
                )
                continue
            if not transforms_match(src.transform, boundary_transform):
                LOGGER.warning("%s has a different transform than the boundary; skipping.", path)
                continue

            data = src.read(masked=False)
            profile = src.profile.copy()
            tags = src.tags()

        dtype = determine_target_dtype(profile["dtype"], nodata_value)
        if dtype != profile["dtype"]:
            LOGGER.info("%s: casting from %s to %s to accommodate nodata value.", path, profile["dtype"], dtype)
            data = data.astype(dtype, copy=False)
            profile.update(dtype=dtype)

        profile.update(nodata=nodata_value)

        updates = 0
        for band_index in range(data.shape[0]):
            band = data[band_index]
            band_mask = mask & (band != nodata_value)
            updates += int(np.count_nonzero(band_mask))
            band[mask] = nodata_value
            data[band_index] = band

        if updates == 0:
            LOGGER.info("%s already contains the correct nodata values.", path)
            continue

        LOGGER.info("%s: updated %s pixel(s).", path, updates)

        if dry_run:
            continue

        temp_path = path.with_suffix(path.suffix + ".tmp")
        if temp_path.exists():
            temp_path.unlink()
        with rasterio.open(temp_path, "w", **profile) as dst:
            dst.write(data)
            if tags:
                dst.update_tags(**tags)

        os.replace(temp_path, path)


def main() -> None:
    args = parse_arguments()
    configure_logging(args.verbose)

    tiff_paths = collect_files(args.tiff_root + args.tiff, patterns=("*.tif", "*.tiff"))
    asc_paths = collect_files(args.asc_root + args.asc, patterns=("*.asc",))

    if tiff_paths:
        shift_geotiffs(tiff_paths, args.shift_west, args.shift_north, args.dry_run)
    else:
        LOGGER.info("No GeoTIFFs found that match the provided parameters.")

    if args.boundary and asc_paths:
        update_ascii_layers(asc_paths, args.boundary, args.nodata, args.dry_run)
    elif args.boundary and not asc_paths:
        LOGGER.info("No ASCII grids found that match the provided parameters.")


if __name__ == "__main__":
    main()
