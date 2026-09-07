# =============================================================================
# Project Title: Wild Boar Distribution Model for France
# Path: scripts/science.py
# Purpose: Research grade front-end combining the full modelling pipeline
#           with spatially blocked cross-validation and extended metrics.
# Process Step: Convenience wrapper that adjusts configuration to produce
#               stable statistics, variable contributions and maps suitable
#               for scientific publications.
# Created: July 2025
# =============================================================================
"""High level pipeline entry point for scientific analyses.

This module wraps :mod:`scripts.run_pipeline` while enforcing settings that
produce reproducible statistics and figures for publication.  In particular
it

* activates spatially blocked cross-validation (50 km blocks by default),
* makes CBI and omission rate the lead evaluation metrics,
* enables all variable importance backends,
* allows light-weight parameter overrides from the command line.

Example
-------

``python -m scripts.science --config config.yaml --season 2022-Winter``

The script writes all outputs to the experiment directory configured in the
YAML file (or a timestamped folder when omitted).
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path
import yaml


def _update_metric_order(metrics: list[str]) -> list[str]:
    """Ensure CBI and omission are listed first.

    Parameters
    ----------
    metrics:
        Existing metric list from the configuration.
    """

    preferred = ["cbi", "omission_rate", "AUC"]
    # keep existing metrics but move preferred ones to the front and drop duplicates
    new = [m for m in preferred if m in metrics]
    for m in metrics:
        if m not in new:
            new.append(m)
    return new


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Run full scientific pipeline")
    ap.add_argument("--config", default="config.yaml", help="Path to YAML configuration")
    ap.add_argument(
        "--season",
        action="append",
        help="Specify seasons like '2022-Winter'. Can be given multiple times.",
    )
    ap.add_argument(
        "--block-size",
        type=int,
        default=50,
        help="Spatial block size in kilometres for cross-validation.",
    )
    ap.add_argument(
        "--regularization",
        type=float,
        help="Override MaxEnt regularization multiplier (optional)",
    )
    args = ap.parse_args(argv)

    cfg = yaml.safe_load(Path(args.config).read_text())

    # Enforce spatially blocked CV and prominent metrics
    cv_cfg = cfg.setdefault("evaluation", {}).setdefault("cross_validation", {})
    cv_cfg["enable"] = True
    cv_cfg["method"] = "spatial_block"
    cv_cfg.setdefault("params", {})["block_size_km"] = args.block_size

    eval_metrics = cfg.setdefault("evaluation", {}).setdefault(
        "metrics", ["AUC", "cbi", "omission_rate"]
    )
    cfg["evaluation"]["metrics"] = _update_metric_order(eval_metrics)

    # Activate all variable importance methods
    cfg.setdefault("steps", {}).setdefault("variable_importance", True)
    cfg["steps"]["variable_importance"] = True
    vi = cfg.setdefault("variable_importance", {})
    vi.setdefault("permutation", {}).setdefault("enable", True)
    vi.setdefault("gain", {}).setdefault("enable", True)
    vi.setdefault("jackknife", {}).setdefault("enable", True)

    # Optional season filter to avoid mixing years
    if args.season:
        cfg.setdefault("seasons", {})["active"] = args.season

    # Optional regularization override
    if args.regularization is not None:
        cfg.setdefault("maxent", {})["regularization_multiplier"] = args.regularization

    # Write modified configuration to a temporary file and delegate to run_pipeline
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as tmp:
        yaml.safe_dump(cfg, tmp)
        tmp_path = tmp.name

    # ``run_pipeline`` reads ``sys.argv`` directly; spoof it here
    import scripts.run_pipeline as rp

    old_argv = sys.argv
    try:
        sys.argv = ["run_pipeline.py", "--config", tmp_path]
        rp.main()
    finally:
        sys.argv = old_argv


if __name__ == "__main__":  # pragma: no cover - manual entry point
    main()

