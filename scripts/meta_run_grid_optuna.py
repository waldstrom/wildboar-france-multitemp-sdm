"""Meta Optuna study evaluating grid-based thinning for seasonal models.

This script iterates over four scenarios (winter/summer x
monotemporal/multitemporal) and uses Optuna to identify the best grid
spacing for presence data thinning.  Mahalanobis thinning is disabled and
replaced by a simple regular grid filter with candidate spacings between
5 and 80 km.  To keep the parameter sweep efficient the expensive
predictor stack is built only once per scenario and reused for subsequent
trials.  Each trial retains the produced figures and metrics including
the Continuous Boyce Index (CBI).
"""
from __future__ import annotations

import copy
import shutil
from pathlib import Path

import optuna
import pandas as pd
import yaml

from scripts import monotemporal, multitemporal

GRID_LEVELS = [5, 10, 20, 40, 60, 80]
TEST_YEAR = 2021


def _run_model(cfg: dict, module, grid_km: int, rebuild: bool) -> tuple[float, float]:
    """Run modelling ``module`` with ``cfg`` and return test AUC and CBI.

    ``rebuild`` controls whether predictor stacks are rebuilt.  After the first
    trial stacks are reused to speed up experimentation.  The output figures of
    each trial are copied to a scenario-specific directory so they are not
    overwritten by subsequent runs."""

    season = cfg["multitemporal"]["seasons"][0]
    module.process_season(cfg, season, rebuild)

    exp_dir = module._experiment_dir(cfg)
    season_out = exp_dir / "output" / season
    dest = exp_dir / "optuna" / f"{grid_km}km"
    shutil.copytree(season_out, dest, dirs_exist_ok=True, copy_function=shutil.copy)
    shutil.rmtree(exp_dir / "output", ignore_errors=True)

    summary = pd.read_csv(dest / "temporal_cv_summary.csv")
    test_row = summary[summary["split"] == "test"].iloc[0]
    auc = float(test_row.get("auc_mean", 0.0))
    cbi_val = test_row.get("cbi_spearman_mean")
    if cbi_val is None or pd.isna(cbi_val):
        cbi_val = test_row.get("cbi_mean", 0.0)
    cbi = float(0.0 if cbi_val is None else cbi_val)
    return auc, cbi


def _objective(trial: optuna.Trial, base_cfg: dict, module) -> float:
    """Optuna objective returning test AUC for a grid spacing trial."""

    grid_km = trial.suggest_categorical("grid_size_km", GRID_LEVELS)
    cfg = copy.deepcopy(base_cfg)
    thin_cfg = cfg.setdefault("presence_data", {}).setdefault("thinning", {})
    thin_cfg["grid_size_km"] = grid_km
    thin_cfg["percentages"] = {}

    rebuild = trial.number == 0
    auc, cbi = _run_model(cfg, module, grid_km, rebuild)
    trial.set_user_attr("cbi", cbi)
    return auc


def main() -> None:
    scenarios = [
        ("configmultitemporal_winter.yaml", multitemporal),
        ("configwinter_monotemporal.yaml", monotemporal),
        ("configmultitemporal_summer.yaml", multitemporal),
        ("configsummer_monotemporal.yaml", monotemporal),
    ]
    results = []
    for cfg_name, module in scenarios:
        base_cfg = yaml.safe_load(Path(cfg_name).read_text())
        base_cfg["run_test_year"] = TEST_YEAR
        exp_cfg = base_cfg.setdefault("experiment", {})
        exp_cfg["name"] = f"{Path(cfg_name).stem}-gridopt"

        sampler = optuna.samplers.GridSampler({"grid_size_km": GRID_LEVELS})
        study = optuna.create_study(direction="maximize", sampler=sampler)
        study.optimize(lambda t: _objective(t, base_cfg, module), n_trials=len(GRID_LEVELS))

        best = study.best_trial
        results.append(
            {
                "config": cfg_name,
                "script": module.__name__.split(".")[-1],
                "best_grid_km": best.params["grid_size_km"],
                "best_auc": best.value,
                "best_cbi": best.user_attrs.get("cbi", float("nan")),
            }
        )

    out_dir = Path("exps")
    out_dir.mkdir(exist_ok=True)
    pd.DataFrame(results).to_csv(out_dir / "grid_thin_optuna_results.csv", index=False)


if __name__ == "__main__":
    main()
