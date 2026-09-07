"""GBIF-only variant of the leave-one-year-out meta runner.

The runner accepts aggregated configuration files that define the seasonal
scenarios to execute.  Two companion YAMLs are now available:

* :mod:`config_gbm_loyo_gbif.yaml` &ndash; uses the tuned Mahalanobis thinning
  percentages per season.
* :mod:`config_gbm_loyo_gbif_grid.yaml` &ndash; disables the Mahalanobis stage and
  instead keeps a single GBIF record per 20 km fishnet cell.

Select the desired behaviour by pointing ``--config`` at the corresponding
file.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from scripts import meta_run_leave_one_year as base
from scripts.utils import ensure_dir


def _gbif_only_config(cfg: dict) -> dict:
    """Return a deep-copied configuration restricted to GBIF records."""

    cloned = deepcopy(cfg)
    presence_cfg = cloned.setdefault("presence_data", {})
    presence_cfg["source_filter"] = ["GBIF"]

    thinning = presence_cfg.get("thinning")
    if isinstance(thinning, dict):
        percentages = thinning.get("percentages")
        if isinstance(percentages, dict):
            filtered: dict[str, dict] = {}
            for season, dataset_map in percentages.items():
                if not isinstance(dataset_map, dict):
                    continue
                gbif_block = dataset_map.get("GBIF")
                if isinstance(gbif_block, dict):
                    filtered[season] = {"GBIF": gbif_block}
                elif gbif_block is not None:
                    filtered[season] = {"GBIF": gbif_block}
                else:
                    filtered[season] = {"GBIF": {}}
            thinning["percentages"] = filtered
    return cloned


def _load_scenarios(config_path: Path) -> list[tuple[str, str, dict]]:
    """Load scenarios and adapt them to GBIF-only operation."""

    scenarios: list[tuple[str, str, dict]] = []
    for label, script, cfg in base._load_scenarios(config_path):
        pretty_label = label if "GBIF" in label else f"{label} (GBIF only)"
        scenarios.append((pretty_label, script, _gbif_only_config(cfg)))
    return scenarios

def main() -> None:
    """Execute the GBIF-only leave-one-year-out meta workflow."""

    parser = argparse.ArgumentParser(
        description="Run all leave-one-year-out seasonal comparison scenarios with GBIF-only data"
    )
    parser.add_argument(
        "--config",
        default="config_gbm_loyo_gbif.yaml",
        help=(
            "Aggregate YAML containing GBIF-only scenario definitions "
            "(e.g. config_gbm_loyo_gbif.yaml for Mahalanobis thinning or "
            "config_gbm_loyo_gbif_grid.yaml for 20 km grid thinning)"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional output directory. Defaults to exps/meta_loyo_gbif_<timestamp>",
    )
    parser.add_argument(
        "--feature-importance-repeats",
        type=int,
        default=10,
        help="Minimum number of permutation repeats for best-run feature importance",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    scenarios = _load_scenarios(config_path)
    if not scenarios:
        raise RuntimeError(f"No scenarios found in {config_path}")

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        out_dir = Path("exps") / f"meta_loyo_gbif_{timestamp}"
    ensure_dir(out_dir)

    metrics_frames: list[pd.DataFrame] = []
    run_infos: list[dict] = []
    for label, script, cfg in scenarios:
        print(f"Running scenario {label} ({script})")
        metrics_df, exp_dir, cfg_used, infos = base._run_scenario(label, script, cfg)
        metrics_frames.append(metrics_df)
        for info in infos:
            info["cfg"] = cfg_used
        run_infos.extend(infos)

    metrics = pd.concat(metrics_frames, ignore_index=True)
    metrics.to_csv(out_dir / "temporal_cv_metrics_all.csv", index=False)

    summary_long, summary_wide = base._summarise_combinations(metrics)
    summary_long.to_csv(out_dir / "combination_summary_long.csv", index=False)

    summary_wide = summary_wide.fillna(np.nan)
    summary_wide["ranking_score"] = summary_wide.get(
        "cbi_spearman_mean_test", pd.Series(dtype=float)
    )
    summary_wide = summary_wide.sort_values(
        ["ranking_score", "cbi_spearman_mean_validation"], ascending=[False, False]
    )
    summary_wide.to_csv(out_dir / "combination_summary_wide.csv", index=False)

    best_runs = base._select_best_runs(metrics)
    pd.DataFrame(best_runs).to_csv(out_dir / "best_runs.csv", index=False)

    info_index = {
        (
            info["scenario"],
            info["temporal"],
            info["engine"],
            info["test_year"],
        ): info
        for info in run_infos
    }

    for record in best_runs:
        key = (record["scenario"], record["temporal"], record["engine"], record["test_year"])
        info = info_index.get(key)
        if not info:
            print(
                "Run directory for "
                f"{record['scenario']} {record['engine']} test {record['test_year']} not found"
            )
            continue
        if info["engine"] == "elapid":
            base._run_spatial_cv(info, out_dir)
        base._run_feature_importance(info, out_dir, repeats=args.feature_importance_repeats)

    base._plot_boxplots(metrics, out_dir)

    if not summary_wide.empty:
        best_combo = summary_wide.iloc[0]
        print(
            "Top-performing GBIF-only combination (by Spearman CBI test mean): "
            f"{best_combo['scenario']} / {best_combo['engine']} with "
            f"CBI_test={best_combo.get('cbi_spearman_mean_test', float('nan')):.3f}"
        )


if __name__ == "__main__":
    main()
