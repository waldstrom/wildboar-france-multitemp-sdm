from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

from review.scripts.common import now_iso, write_json


def main() -> int:
    parser = argparse.ArgumentParser(description="Run MESS/NT1/NT2 reviewer diagnostics.")
    parser.add_argument("--repo-root", required=True, type=Path)
    parser.add_argument("--mono-run", required=True, type=Path)
    parser.add_argument("--multi-run", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--season", default="Winter")
    parser.add_argument("--year", type=int)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    logger = logging.getLogger("review_corrections.novelty")
    repo_root = args.repo_root.resolve()
    os.chdir(repo_root)
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from scripts.mess_evaluation import compare_runs, run_mess

    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    state = {
        "status": "running",
        "dataset": args.dataset,
        "started_utc": now_iso(),
        "mono_run": str(args.mono_run.resolve()),
        "multi_run": str(args.multi_run.resolve()),
        "season": args.season,
        "year": args.year,
    }
    write_json(state, output_root / "novelty_state.json")
    try:
        years = [args.year] if args.year is not None else None
        mono_dir = output_root / "01_monotemporal"
        multi_dir = output_root / "02_multitemporal"
        compare_dir = output_root / "03_comparison"
        logger.info("Running monotemporal novelty diagnostics")
        mono_result = run_mess(
            args.mono_run.resolve(),
            season=args.season,
            run_type=f"{args.dataset}_winter_monotemporal",
            output_root=mono_dir,
            years=years,
        )
        logger.info("Running multitemporal novelty diagnostics")
        multi_result = run_mess(
            args.multi_run.resolve(),
            season=args.season,
            run_type=f"{args.dataset}_winter_multitemporal",
            output_root=multi_dir,
            years=years,
        )
        logger.info("Comparing novelty diagnostics")
        comparison = compare_runs(mono_result, multi_result, output_root=compare_dir)
    except Exception as exc:
        state.update(
            {
                "status": "failed",
                "finished_utc": now_iso(),
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        )
        write_json(state, output_root / "novelty_state.json")
        logger.exception("Novelty diagnostics failed")
        return 1

    state.update(
        {
            "status": "complete",
            "finished_utc": now_iso(),
            "mono_result": str(mono_result),
            "multi_result": str(multi_result),
            "comparison": str(comparison),
        }
    )
    write_json(state, output_root / "novelty_state.json")
    (output_root / "COMPLETE").write_text(now_iso() + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
