# =============================================================================
# Project Title: Wild Boar Distribution Model for France
# Path: scripts/seasontester_parallel.py
# Purpose: Run the modelling pipeline for multiple seasons in parallel.
# Process Step: Generates customised configs and executes the pipeline with
#               up to 12 concurrent processes.
# Created by Adrian Meyer, Institute Geomatics, FHNW (adrian.meyer@fhnw.ch)
# Created: July 2025
# Version: v.0.1.0
# =============================================================================
"""Run pipeline for each season and data source combination in parallel.

This script mirrors :mod:`seasontester` but executes the pipeline concurrently.
It spawns up to 12 ``run_pipeline.py`` processes at a time to speed up the
seasonal experiments.
"""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import subprocess
import os
import re
import time


MAX_PROCESSES = 12


# -----------------------------------------------------------------------------

def modify_config(cfg: str, exp_name: str) -> str:
    """Return config text with updated ``experiment.name``."""
    return re.sub(
        r"^  name:.*$",
        f"  name: \"{exp_name}\"",
        cfg,
        flags=re.MULTILINE,
    )


def wait_for_slot(processes: list[subprocess.Popen]) -> None:
    """Block until a free process slot is available."""
    while len(processes) >= MAX_PROCESSES:
        for p in list(processes):
            if p.poll() is not None:
                processes.remove(p)
        if len(processes) >= MAX_PROCESSES:
            time.sleep(1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true", help="Only create configs")
    args = ap.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    parent_name = f"seasontest_{timestamp}"
    parent_dir = Path("exps") / parent_name
    config_dir = parent_dir / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)

    cfg_src_dir = Path("configs/legacy/scenarios")
    cfg_files = sorted(cfg_src_dir.glob("*.yaml"))

    # Run preprocessing once for all seasons
    master_proc = parent_dir / "processed_master"
    env_master = os.environ.copy()
    env_master["PROCESSED_DATA_DIR"] = str(master_proc)
    base_cfg = Path("configs/legacy/configall.yaml").read_text()
    pre_cfg = modify_config(base_cfg, f"{parent_name}/preprocess")
    for step in [
        "variable_selection",
        "thinning",
        "bias_grid",
        "background",
        "training",
        "evaluation",
        "variable_importance",
    ]:
        pre_cfg = re.sub(rf"^  {step}:.*$", "  " + step + ": false", pre_cfg, flags=re.MULTILINE)
    pre_cfg = re.sub(r"^  preprocessing:.*$", "  preprocessing: true", pre_cfg, flags=re.MULTILINE)
    pre_cfg_file = config_dir / "_preprocess.yaml"
    pre_cfg_file.write_text(pre_cfg)
    if not args.dry:
        subprocess.run(
            ["python", "scripts/run_pipeline.py", "--config", str(pre_cfg_file)],
            check=False,
            env=env_master,
        )

    processes: list[subprocess.Popen] = []
    for cfg_file in cfg_files:
        cfg_text = cfg_file.read_text()
        exp_name = f"{parent_name}/{cfg_file.stem}"
        mod_text = modify_config(cfg_text, exp_name)
        cfg_out = config_dir / cfg_file.name
        cfg_out.write_text(mod_text)

        if not args.dry:
            env = os.environ.copy()
            run_proc = parent_dir / f"processed_{cfg_file.stem}"
            env["PROCESSED_DATA_DIR"] = str(run_proc)
            if not run_proc.exists():
                run_proc.mkdir(parents=True, exist_ok=True)
            master_stack = master_proc / "stack" / "predictor_stack.zarr"
            dest_stack = run_proc / "stack" / "predictor_stack.zarr"
            if master_stack.exists() and not dest_stack.exists():
                import shutil
                shutil.copytree(master_stack, dest_stack)
            wait_for_slot(processes)
            proc = subprocess.Popen(
                ["python", "scripts/run_pipeline.py", "--config", str(cfg_out)],
                env=env,
            )
            processes.append(proc)

    for p in processes:
        p.wait()


if __name__ == "__main__":
    main()
