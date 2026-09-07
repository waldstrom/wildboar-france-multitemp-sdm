# =============================================================================
# Project Title: Wild Boar Distribution Model for France
# Path: scripts/seasoncombinedtest.py
# Purpose: Build averaged predictor stacks for summer and winter and run the
#          modelling pipeline once per season.
# Process Step: Generates combined stacks across all years and executes the
#               pipeline. The resulting configuration remains fully optimisable
#               via Optuna when training.
# Created by OpenAI's Codex.
# =============================================================================
"""Run multiple pipeline variants with averaged predictor stacks.

Yearly predictor layers are merged into a single mean stack per season while
discarding variables marked ``drop`` in the feature configuration. Models are
trained for each data source (GBIF, RRN, SNCF and the combined set) for summer,
winter and all seasons. Additional variants disable variable selection or
presence thinning. The pipeline remains fully optimisable with Optuna.

"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
from datetime import datetime
from pathlib import Path

import xarray as xr
import yaml

# Local imports from the repository
from scripts.preprocessing import build_stack
from scripts.utils import load_config


# -----------------------------------------------------------------------------

def _update_paths(cfg: dict, exp_dir: Path) -> None:
    """Recursively update ``outputs`` paths to live under ``exp_dir``."""

    for k, v in cfg.items():
        if isinstance(v, dict):
            _update_paths(v, exp_dir)
        elif isinstance(v, str):
            if v.startswith("outputs/"):
                cfg[k] = str(exp_dir / v)
            elif v.startswith("sqlite:///outputs/"):
                cfg[k] = "sqlite:///" + str(exp_dir / v[len("sqlite:///"):])


# -----------------------------------------------------------------------------

def _determine_years(cfg: dict) -> list[int]:
    """Return list of years implied by the configuration."""

    seasons = cfg.get("seasons", {}).get("active", [])
    if seasons:
        years = [int(s.split("-")[0]) for s in seasons]
    else:
        start, end = cfg["presence_data"]["year_filter"]
        years = list(range(start, end + 1))
    return sorted(set(years))


# -----------------------------------------------------------------------------

def _average_stack(stack_path: Path, years: list[int], fc_path: Path) -> None:
    """Average yearly variables in ``stack_path`` and overwrite the dataset.

    All variables flagged ``drop`` in ``fc_path`` are removed. Yearly
    variables are averaged across the provided ``years`` and stored with the
    suffix of the latest available year. Non-temporal variables are kept as-is.
    """

    # Load the stack into memory before rewriting the same path.
    # ``to_zarr`` with ``mode="w"`` clears the target store which would make
    # lazily loaded variables unreadable while writing. Loading the dataset
    # ensures that data remain available even after the underlying Zarr store is
    # overwritten.
    ds = xr.open_zarr(stack_path, consolidated=False).load()

    feature_cfg = yaml.safe_load(fc_path.read_text()) if fc_path.exists() else {}
    drop_vars = {
        k for k, v in feature_cfg.get("variables", {}).items() if str(v).lower() == "drop"
    }

    # Determine the newest year actually present in the stack
    year_in_stack = set()
    for var in ds.data_vars:
        m = re.match(r"(\d{4})_", var)
        if m:
            year_in_stack.add(int(m.group(1)))
    last_year = max(year_in_stack) if year_in_stack else years[-1]

    groups: dict[str, list[xr.DataArray]] = {}
    for var in ds.data_vars:
        if var in drop_vars:
            continue

        m = re.match(r"(\d{4})_(.*)", var)
        if m:
            base = m.group(2)
            key = f"{last_year}_{base}"
        else:
            key = var
        groups.setdefault(key, []).append(ds[var])

    out = xr.Dataset()

    for name, arrs in groups.items():
        if len(arrs) == 1:
            out[name] = arrs[0]
        else:
            out[name] = xr.concat(arrs, dim="year").mean(dim="year", skipna=True)

    out.to_zarr(stack_path, mode="w", consolidated=False)


# -----------------------------------------------------------------------------

def _prepare_config(
    cfg_file: Path,
    exp_name: str,
    dataset: str | None = None,
    no_filter: bool = False,
    no_thin: bool = False,
) -> tuple[dict, Path]:
    """Load ``cfg_file`` and return modified config and path to save it."""

    cfg = load_config(cfg_file)
    cfg["experiment"]["name"] = exp_name
    exp_dir = Path("exps") / exp_name
    _update_paths(cfg, exp_dir)

    years = _determine_years(cfg)
    for spec in cfg.get("predictors", {}).get("layers", {}).values():
        if isinstance(spec, dict) and "years" in spec:
            spec["years"] = years

    cfg["predictors"]["variable_selection"]["reference_year"] = max(years)


    if dataset and dataset.lower() != "combined":
        cfg["presence_data"]["source_filter"] = [dataset]
    else:
        cfg["presence_data"]["source_filter"] = None

    if no_filter:
        cfg["steps"]["variable_selection"] = False
    if no_thin:
        cfg["steps"]["thinning"] = False

    return cfg, exp_dir


# -----------------------------------------------------------------------------

def run_model(
    cfg_src: Path,
    season_name: str,
    dataset: str | None,
    no_filter: bool = False,
    no_thin: bool = False,
    dry: bool = False,
) -> None:
    """Build averaged stack and execute the pipeline for one experiment."""

    suffix = []
    if dataset:
        suffix.append(dataset.lower())
    if no_filter:
        suffix.append("nofilter")
    if no_thin:
        suffix.append("nothin")
    joined = "_".join([s for s in suffix if s])
    exp_name = f"seasoncombined_{season_name}"
    if joined:
        exp_name += f"_{joined}"
    exp_name += f"_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    cfg, exp_dir = _prepare_config(
        cfg_src,
        exp_name,
        dataset=dataset,
        no_filter=no_filter,
        no_thin=no_thin,
    )

    # 1. Build full stack and average yearly layers
    stack_path = build_stack(cfg)

    fc_file = Path(cfg["predictors"]["variable_selection"].get("feature_config", "featureconfig.yaml"))
    _average_stack(stack_path, _determine_years(cfg), fc_file)

    # 2. Disable preprocessing for the actual run
    cfg["steps"]["preprocessing"] = False
    cfg_out = exp_dir / "config.yaml"
    cfg_out.parent.mkdir(parents=True, exist_ok=True)
    cfg_out.write_text(yaml.safe_dump(cfg))

    if dry:
        return

    env = os.environ.copy()
    env["PROCESSED_DATA_DIR"] = str(exp_dir / "processed")
    subprocess.run(["python", "scripts/run_pipeline.py", "--config", str(cfg_out)], check=False, env=env)


# -----------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true", help="Only create configs and stacks")
    args = ap.parse_args()

    seasons = {
        "summer": Path("configsummer.yaml"),
        "winter": Path("configwinter.yaml"),
        "all": Path("configall.yaml"),
    }
    datasets = ["GBIF", "RRN", "SNCF", "Combined"]
    variants = [
        (False, False),  # default
        (True, False),   # no feature filtering
        (False, True),   # no thinning
    ]

    for season, cfg in seasons.items():
        for ds in datasets:
            for no_filter, no_thin in variants:
                run_model(
                    cfg,
                    season,
                    None if ds == "Combined" else ds,
                    no_filter=no_filter,
                    no_thin=no_thin,
                    dry=args.dry,
                )


if __name__ == "__main__":
    main()
