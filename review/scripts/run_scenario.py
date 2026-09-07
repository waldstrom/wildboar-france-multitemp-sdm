from __future__ import annotations

import argparse
import importlib
import inspect
import json
import logging
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from review.scripts.common import deep_get, load_yaml, now_iso, write_json


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def call_process_season(
    module: Any,
    cfg: dict[str, Any],
    season: str,
    rebuild: bool,
) -> None:
    """Call the repository's non-interactive season API across minor signature changes."""
    func = getattr(module, "process_season", None)
    if func is None:
        raise AttributeError(
            f"{module.__name__} has no process_season(); refusing to call interactive main()"
        )
    signature = inspect.signature(func)
    params = list(signature.parameters)
    if len(params) >= 3:
        func(cfg, season, rebuild)
    elif len(params) == 2:
        func(cfg, season)
    else:
        raise TypeError(
            f"Unsupported {module.__name__}.process_season signature: {signature}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run one resolved reviewer-correction scenario non-interactively."
    )
    parser.add_argument("--repo-root", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--runner", required=True, choices=["monotemporal", "multitemporal"])
    parser.add_argument("--scenario-dir", required=True, type=Path)
    parser.add_argument("--season", default="Winter")
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    configure_logging(args.log_level)
    logger = logging.getLogger("review_corrections.child")
    repo_root = args.repo_root.resolve()
    scenario_dir = args.scenario_dir.resolve()
    scenario_dir.mkdir(parents=True, exist_ok=True)
    os.chdir(repo_root)
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    cfg = load_yaml(args.config)
    include_hunting = bool(deep_get(cfg, "multitemporal.include_hunting", True))
    expected = bool(cfg.get("experiment", {}).get("review_expect_hunting", True))
    if include_hunting != expected:
        raise RuntimeError(
            f"Resolved config has include_hunting={include_hunting}; expected {expected}"
        )

    seed = int(deep_get(cfg, "experiment.random_seed", 42))
    random.seed(seed)
    np.random.seed(seed)
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    os.environ.setdefault("MPLBACKEND", "Agg")

    module_name = f"scripts.{args.runner}"
    logger.info("Importing %s", module_name)
    module = importlib.import_module(module_name)

    state_path = scenario_dir / "scenario_state.json"
    started = time.monotonic()
    state = {
        "status": "running",
        "started_utc": now_iso(),
        "runner": args.runner,
        "season": args.season,
        "config": str(args.config.resolve()),
        "rebuild": bool(args.rebuild),
        "pid": os.getpid(),
    }
    write_json(state, state_path)

    try:
        logger.info(
            "Starting %s / %s with hunting=%s and rebuild=%s",
            args.runner,
            args.season,
            include_hunting,
            args.rebuild,
        )
        call_process_season(module, cfg, args.season, args.rebuild)
    except Exception as exc:
        state.update(
            {
                "status": "failed",
                "finished_utc": now_iso(),
                "elapsed_seconds": time.monotonic() - started,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        )
        write_json(state, state_path)
        logger.exception("Scenario failed")
        return 1

    state.update(
        {
            "status": "complete",
            "finished_utc": now_iso(),
            "elapsed_seconds": time.monotonic() - started,
        }
    )
    write_json(state, state_path)
    (scenario_dir / "COMPLETE").write_text(now_iso() + "\n", encoding="utf-8")
    logger.info("Scenario completed successfully")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
