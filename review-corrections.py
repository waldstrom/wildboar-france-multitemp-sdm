#!/usr/bin/env python3
"""Re-run and export the winter-monotemporal hunting-predictor correction.

This launcher deliberately leaves the repository's original experiment YAML files
untouched. It resolves corrected, immutable run configurations below a timestamped
``review/output`` directory, executes the repository's non-interactive season APIs
in isolated child processes, audits hunting-predictor provenance, and exports
publication artifacts in manuscript-first and supplement-second order.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import traceback
import zipfile
from pathlib import Path
from typing import Any

# Keep imports valid when invoked from any working directory.
SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from review.scripts.artifacts import export_publication_artifacts, load_registry
from review.scripts.audit import aggregate_audits, audit_scenario
from review.scripts.common import (
    RunLock,
    atomic_write_text,
    environment_metadata,
    file_inventory,
    git_metadata,
    load_yaml,
    local_run_id,
    now_iso,
    relative_or_absolute,
    stream_command,
    write_csv,
    write_json,
)
from review.scripts.configuration import (
    discover_disabled_winter_monotemporal_configs,
    resolve_scenario,
)
from review.scripts.validate import validate_run

LOG = logging.getLogger("review_corrections")


def configure_logging(level: str, log_path: Path | None = None) -> None:
    LOG.setLevel(getattr(logging, level.upper(), logging.INFO))
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.setLevel(LOG.level)
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    root_logger.addHandler(console)
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)


def find_repo_root(start: Path) -> Path:
    """Require the launcher to be dropped at the repository root."""
    start = start.resolve()
    markers = [
        start / "scripts" / "monotemporal.py",
        start / "scripts" / "multitemporal.py",
        start / "configs/experiments/configwinter_monotemporal.yaml",
    ]
    missing = [str(path) for path in markers if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "review-corrections.py must be located at the repository root. "
            f"Missing required repository files: {missing}"
        )
    return start


def required_dependencies() -> dict[str, str]:
    modules = [
        "yaml",
        "pandas",
        "numpy",
        "matplotlib",
        "rasterio",
        "geopandas",
        "xarray",
        "zarr",
        "sklearn",
    ]
    result = {}
    for module in modules:
        try:
            imported = __import__(module)
            result[module] = getattr(imported, "__version__", "available")
        except Exception as exc:
            result[module] = f"MISSING: {exc}"
    return result


def write_plan(
    run_dir: Path,
    config: dict[str, Any],
    specs: list[dict[str, Any]],
    *,
    corrected_only: bool,
    include_controls: bool,
) -> None:
    lines = [
        "# Reviewer-correction execution plan",
        "",
        f"- Created: {now_iso()}",
        "- Scientific correction: activate the hunting-bag predictor in winter monotemporal models.",
        "- Default matched rerun: corrected monotemporal plus multitemporal reference for all-points and GBIF-only datasets.",
        "- Fresh-stack policy: each scenario uses an isolated timestamped experiment directory and `rebuild=True`.",
        "- Candidate-vs-selected audit: hunting source, master-stack presence and fold-level retention are reported separately.",
        "",
        "## Scenario order",
        "",
    ]
    for spec in sorted(specs, key=lambda s: int(s.get("order", 999))):
        role = str(spec.get("role", "corrected"))
        enabled = bool(spec.get("enabled", True))
        if corrected_only and role == "reference":
            enabled = False
        if role == "control" and not include_controls:
            enabled = False
        lines.append(
            f"{int(spec.get('order', 999)):02d}. `{spec['id']}` — "
            f"{role}; {'RUN' if enabled else 'SKIP'}"
        )
    lines += [
        "",
        "## Publication export order",
        "",
        "1. Main LOYO performance table",
        "2. Main LOYO performance figures",
        "3. Main 2019 winter prediction map comparison",
        "4. Main winter feature importance",
        "5. Main-text numerical replacement register",
        "6. Supplement S5 corrected configuration",
        "7. Supplement S6 fold-wise metrics and confidence intervals",
        "8. Supplement S7 feature provenance and hunting audit",
        "9. Supplementary response curves and feature importance",
        "10. Supplementary MESS/NT1/NT2 diagnostics",
        "11. Full corrected held-out-year maps",
        "",
    ]
    atomic_write_text(run_dir / "00_RUN" / "00_execution_plan.md", "\n".join(lines))


def save_environment(repo_root: Path, run_dir: Path) -> None:
    metadata = environment_metadata(repo_root)
    metadata["dependencies"] = required_dependencies()
    write_json(metadata, run_dir / "00_RUN" / "02_provenance.json")
    try:
        freeze = subprocess.run(
            [sys.executable, "-m", "pip", "freeze"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=False,
        )
        atomic_write_text(
            run_dir / "00_RUN" / "03_environment_pip_freeze.txt",
            freeze.stdout + ("\n# STDERR\n" + freeze.stderr if freeze.stderr else ""),
        )
    except Exception as exc:
        atomic_write_text(
            run_dir / "00_RUN" / "03_environment_pip_freeze.txt",
            f"pip freeze unavailable: {exc}\n",
        )


def selected_specs(
    config: dict[str, Any],
    *,
    corrected_only: bool,
    include_controls: bool,
    only: set[str] | None,
) -> list[dict[str, Any]]:
    out = []
    for raw in config.get("scenarios", []):
        spec = dict(raw)
        if not bool(spec.get("enabled", True)):
            continue
        role = str(spec.get("role", "corrected"))
        if corrected_only and role == "reference":
            continue
        if role == "control" and not include_controls:
            continue
        if only and str(spec["id"]) not in only:
            continue
        out.append(spec)
    return sorted(out, key=lambda s: int(s.get("order", 999)))


def prepare_run_directory(
    repo_root: Path,
    config: dict[str, Any],
    *,
    output_root_override: Path | None,
    run_id_override: str | None,
    resume: Path | None,
) -> tuple[Path, str]:
    if resume:
        run_dir = resume.resolve()
        if not run_dir.exists():
            raise FileNotFoundError(f"Resume directory not found: {run_dir}")
        return run_dir, run_dir.name

    review_cfg = config["review"]
    output_root = output_root_override or Path(review_cfg.get("output_root", "review/output"))
    if not output_root.is_absolute():
        output_root = repo_root / output_root
    run_id = run_id_override or local_run_id(
        str(review_cfg.get("timezone", "Europe/Athens"))
    )
    run_dir = output_root / run_id
    if run_dir.exists():
        raise FileExistsError(
            f"Run directory already exists: {run_dir}. Use --resume or a new --run-id."
        )
    for relative in [
        "00_RUN",
        "01_INPUT/01_configs",
        "01_INPUT/02_publication_index",
        "01_INPUT/03_repository_snapshot",
        "02_MODELS",
        "03_MANUSCRIPT",
        "04_SUPPLEMENT",
        "05_DIAGNOSTICS/scenario_audits",
        "05_DIAGNOSTICS/novelty",
        "05_DIAGNOSTICS/feature_importance",
        "06_MACHINE_READABLE",
        "07_ARCHIVE",
    ]:
        (run_dir / relative).mkdir(parents=True, exist_ok=True)
    return run_dir, run_id


def snapshot_inputs(
    repo_root: Path,
    config_path: Path,
    run_dir: Path,
    config: dict[str, Any],
) -> None:
    shutil.copy2(config_path, run_dir / "01_INPUT" / "01_configs" / config_path.name)
    artifact_manifest = repo_root / "review" / "config" / "artifact-manifest.yaml"
    if artifact_manifest.exists():
        shutil.copy2(
            artifact_manifest,
            run_dir / "01_INPUT" / "01_configs" / artifact_manifest.name,
        )
    publication_index = repo_root / "review" / "input" / "current_publication_index.csv"
    if publication_index.exists():
        shutil.copy2(
            publication_index,
            run_dir / "01_INPUT" / "02_publication_index" / publication_index.name,
        )

    source_paths = set()
    for spec in config.get("scenarios", []):
        source = spec.get("source", {})
        value = source.get("aggregate_config") or source.get("config")
        if value:
            source_paths.add(repo_root / str(value))
    rows = []
    for path in sorted(source_paths):
        if path.exists():
            target = run_dir / "01_INPUT" / "03_repository_snapshot" / path.name
            shutil.copy2(path, target)
            row = file_inventory(path)
            row["relative_path"] = relative_or_absolute(path, repo_root)
            rows.append(row)
    write_csv(rows, run_dir / "00_RUN" / "04_input_config_inventory.csv")


def archive_partial_scenario(record: dict[str, Any]) -> None:
    scenario_dir = Path(record["scenario_dir"])
    experiment_dir = Path(record["experiment_dir"])
    if (scenario_dir / "COMPLETE").exists() or not experiment_dir.exists():
        return
    archive_root = scenario_dir / "_partial_archive"
    archive_root.mkdir(parents=True, exist_ok=True)
    target = archive_root / f"{experiment_dir.name}_{local_run_id('UTC', 'partial')}"
    LOG.warning("Archiving partial scenario output to %s", target)
    shutil.move(str(experiment_dir), str(target))


def run_scenarios(
    repo_root: Path,
    run_dir: Path,
    records: list[dict[str, Any]],
    *,
    log_level: str,
    force: bool,
) -> None:
    for record in records:
        scenario_dir = Path(record["scenario_dir"])
        complete = scenario_dir / "COMPLETE"
        if complete.exists() and not force:
            LOG.info(
                "[%02d/%02d] %s already complete; resuming without rerun",
                records.index(record) + 1,
                len(records),
                record["id"],
            )
            continue
        if force and complete.exists():
            complete.unlink()
            state = scenario_dir / "scenario_state.json"
            if state.exists():
                state.rename(scenario_dir / "scenario_state.previous.json")
        archive_partial_scenario(record)
        LOG.info(
            "[%02d/%02d] Running %s (%s, %s, %s)",
            records.index(record) + 1,
            len(records),
            record["id"],
            record["dataset"],
            record["temporal"],
            record["role"],
        )
        command = [
            sys.executable,
            "-m",
            "review.scripts.run_scenario",
            "--repo-root",
            str(repo_root),
            "--config",
            record["resolved_config"],
            "--runner",
            record["runner"],
            "--scenario-dir",
            record["scenario_dir"],
            "--season",
            record["season"],
            "--rebuild",
            "--log-level",
            log_level,
        ]
        return_code = stream_command(
            command,
            cwd=repo_root,
            log_path=scenario_dir / "scenario_console.log",
            prefix=record["id"],
            env={
                "PYTHONPATH": str(repo_root),
                "MPLBACKEND": "Agg",
            },
        )
        if return_code != 0 or not complete.exists():
            raise RuntimeError(
                f"Scenario {record['id']} failed with exit code {return_code}. "
                f"Resume after correction with: python review-corrections.py run --resume {run_dir}"
            )


def run_audits(
    repo_root: Path,
    run_dir: Path,
    records: list[dict[str, Any]],
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    expected_splits = list(
        config.get("review", {}).get("validation", {}).get(
            "expected_splits", ["train", "validation", "test"]
        )
    )
    audit_dirs = []
    summaries = []
    for record in records:
        audit_dir = (
            run_dir
            / "05_DIAGNOSTICS"
            / "scenario_audits"
            / f"{int(record['order']):02d}_{record['id']}"
        )
        LOG.info("Auditing hunting provenance and outputs for %s", record["id"])
        summary = audit_scenario(
            repo_root,
            record,
            output_dir=audit_dir,
            expected_splits=expected_splits,
        )
        summaries.append(summary)
        audit_dirs.append(audit_dir)
        if summary["status"] != "pass":
            LOG.error("Audit failed for %s: %s", record["id"], summary["issues"])
    aggregate_audits(audit_dirs, run_dir / "00_RUN" / "scenario_audit_summary")
    return summaries


def run_feature_importance(
    repo_root: Path,
    run_dir: Path,
    records: list[dict[str, Any]],
    config: dict[str, Any],
    *,
    log_level: str,
    force: bool,
) -> None:
    feature_cfg = config.get("review", {}).get("feature_importance", {})
    n_repeats = int(feature_cfg.get("n_repeats", 10))
    seed = int(config.get("review", {}).get("seed", 42))
    for record in records:
        output = run_dir / "05_DIAGNOSTICS" / "feature_importance" / record["id"]
        complete = output / "COMPLETE"
        if complete.exists() and not force:
            LOG.info("Feature importance already complete for %s", record["id"])
            continue
        LOG.info("Computing best-fold feature importance for %s", record["id"])
        command = [
            sys.executable,
            "-m",
            "review.scripts.run_feature_importance",
            "--repo-root",
            str(repo_root),
            "--experiment-dir",
            record["experiment_dir"],
            "--scenario",
            record["id"],
            "--season",
            record["season"],
            "--output-root",
            str(output),
            "--n-repeats",
            str(n_repeats),
            "--random-state",
            str(seed),
            "--log-level",
            log_level,
        ]
        rc = stream_command(
            command,
            cwd=repo_root,
            log_path=output / "feature_importance_console.log",
            prefix=f"importance:{record['id']}",
            env={"PYTHONPATH": str(repo_root), "MPLBACKEND": "Agg"},
        )
        if rc != 0:
            raise RuntimeError(f"Feature importance failed for {record['id']}")


def run_novelty(
    repo_root: Path,
    run_dir: Path,
    records: list[dict[str, Any]],
    *,
    log_level: str,
    force: bool,
) -> None:
    datasets = sorted({record["dataset"] for record in records if record.get("role") != "control"})
    for dataset in datasets:
        mono = next(
            (
                r
                for r in records
                if r["dataset"] == dataset
                and r["temporal"] == "monotemporal"
                and r.get("role") != "control"
            ),
            None,
        )
        multi = next(
            (
                r
                for r in records
                if r["dataset"] == dataset and r["temporal"] == "multitemporal"
            ),
            None,
        )
        if not mono or not multi:
            LOG.warning(
                "Skipping novelty comparison for %s: matched mono/multi pair unavailable",
                dataset,
            )
            continue
        output = run_dir / "05_DIAGNOSTICS" / "novelty" / dataset
        complete = output / "COMPLETE"
        if complete.exists() and not force:
            LOG.info("Novelty diagnostics already complete for %s", dataset)
            continue
        LOG.info("Running MESS/NT1/NT2 diagnostics for %s", dataset)
        command = [
            sys.executable,
            "-m",
            "review.scripts.run_novelty",
            "--repo-root",
            str(repo_root),
            "--mono-run",
            mono["experiment_dir"],
            "--multi-run",
            multi["experiment_dir"],
            "--output-root",
            str(output),
            "--season",
            mono["season"],
            "--dataset",
            dataset,
            "--log-level",
            log_level,
        ]
        rc = stream_command(
            command,
            cwd=repo_root,
            log_path=output / "novelty_console.log",
            prefix=f"novelty:{dataset}",
            env={"PYTHONPATH": str(repo_root), "MPLBACKEND": "Agg"},
        )
        if rc != 0:
            raise RuntimeError(f"Novelty diagnostics failed for {dataset}")


def make_archive(
    run_dir: Path,
    *,
    max_member_mb: int,
) -> Path:
    archive_path = run_dir / "07_ARCHIVE" / f"{run_dir.name}_publication_bundle.zip"
    include_roots = [
        run_dir / "00_RUN",
        run_dir / "01_INPUT",
        run_dir / "03_MANUSCRIPT",
        run_dir / "04_SUPPLEMENT",
        run_dir / "06_MACHINE_READABLE",
    ]
    max_bytes = max_member_mb * 1024 * 1024
    skipped = []
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for include_root in include_roots:
            for path in sorted(include_root.rglob("*")):
                if not path.is_file():
                    continue
                if path == archive_path:
                    continue
                if path.stat().st_size > max_bytes:
                    skipped.append(
                        {
                            "path": relative_or_absolute(path, run_dir),
                            "size_bytes": path.stat().st_size,
                            "reason": f"exceeds archive.max_member_mb={max_member_mb}",
                        }
                    )
                    continue
                zf.write(path, arcname=str(path.relative_to(run_dir)))
    write_csv(skipped, run_dir / "07_ARCHIVE" / "large_files_excluded_from_zip.csv")
    return archive_path


def resolve_records(
    repo_root: Path,
    run_dir: Path,
    config: dict[str, Any],
    specs: list[dict[str, Any]],
    run_id: str,
) -> list[dict[str, Any]]:
    review_cfg = config["review"]
    default_engines = [str(v) for v in review_cfg.get("engines", ["elapid", "gbm"])]
    seed = int(review_cfg.get("seed", 42))
    records = []
    for spec in specs:
        scenario_dir = (
            run_dir
            / "02_MODELS"
            / f"{int(spec.get('order', 999)):02d}_{spec['id']}"
        )
        resolved = resolve_scenario(
            repo_root,
            spec,
            scenario_dir,
            run_id=run_id,
            default_engines=default_engines,
            default_seed=seed,
        )
        records.append(resolved.to_dict())
    write_json(records, run_dir / "00_RUN" / "scenario_registry.json")
    return records


def execute_run(args: argparse.Namespace) -> int:
    repo_root = find_repo_root(SCRIPT_PATH.parent)
    config_path = (
        args.config.resolve()
        if args.config.is_absolute()
        else (repo_root / args.config).resolve()
    )
    config = load_yaml(config_path)
    run_dir, run_id = prepare_run_directory(
        repo_root,
        config,
        output_root_override=args.output_root,
        run_id_override=args.run_id,
        resume=args.resume,
    )
    configure_logging(args.log_level, run_dir / "00_RUN" / "01_review_corrections.log")
    lock = RunLock(run_dir / "00_RUN" / "RUNNING.lock")
    lock.acquire(force=args.force_lock)
    try:
        LOG.info("Repository: %s", repo_root)
        LOG.info("Review run: %s", run_dir)
        specs = selected_specs(
            config,
            corrected_only=args.corrected_only,
            include_controls=args.include_controls,
            only=set(args.only) if args.only else None,
        )
        if not specs:
            raise ValueError("No scenarios selected")

        write_plan(
            run_dir,
            config,
            specs,
            corrected_only=args.corrected_only,
            include_controls=args.include_controls,
        )
        save_environment(repo_root, run_dir)
        snapshot_inputs(repo_root, config_path, run_dir, config)

        disabled = discover_disabled_winter_monotemporal_configs(
            repo_root,
            list(
                config.get("review", {})
                .get("config_discovery", {})
                .get("globs", ["config*.yaml", "configs/**/*.yaml"])
            ),
        )
        write_csv(disabled, run_dir / "00_RUN" / "disabled_hunting_config_inventory.csv")
        LOG.info(
            "Repository-wide audit found %d winter-monotemporal config occurrence(s) with hunting disabled",
            sum(1 for row in disabled if row.get("status") == "disabled"),
        )

        records = resolve_records(repo_root, run_dir, config, specs, run_id)
        if args.dry_run:
            LOG.info("Dry run complete: configs resolved and inputs inventoried; no models executed")
            return 0

        run_scenarios(
            repo_root,
            run_dir,
            records,
            log_level=args.log_level,
            force=args.force_scenarios,
        )
        audit_summaries = run_audits(repo_root, run_dir, records, config)
        failures = [row for row in audit_summaries if row.get("status") != "pass"]
        if failures and not args.continue_after_audit_failure:
            raise RuntimeError(
                "One or more scenario audits failed; publication export stopped to avoid "
                "silently publishing stale or incomplete outputs. See "
                f"{run_dir / '00_RUN' / 'scenario_audit_summary.csv'}"
            )

        if not args.skip_feature_importance:
            run_feature_importance(
                repo_root,
                run_dir,
                records,
                config,
                log_level=args.log_level,
                force=args.force_diagnostics,
            )
        else:
            LOG.warning("Feature-importance stage skipped by command line")

        if not args.skip_novelty:
            run_novelty(
                repo_root,
                run_dir,
                records,
                log_level=args.log_level,
                force=args.force_diagnostics,
            )
        else:
            LOG.warning("Novelty stage skipped by command line")

        LOG.info("Exporting publication artifacts in manuscript/supplement order")
        export_result = export_publication_artifacts(run_dir, records, config)
        write_json(export_result, run_dir / "00_RUN" / "export_result.json")

        validation = validate_run(run_dir, config)
        if validation["status"] != "pass" and not args.allow_validation_failure:
            raise RuntimeError(
                f"Validation failed; inspect {run_dir / '00_RUN' / 'validation_report.json'}"
            )

        if not args.no_archive:
            max_member_mb = int(
                config.get("review", {}).get("archive", {}).get("max_member_mb", 100)
            )
            archive = make_archive(run_dir, max_member_mb=max_member_mb)
            LOG.info("Publication bundle: %s", archive)

        latest_file = run_dir.parent / "LATEST.txt"
        atomic_write_text(latest_file, str(run_dir.resolve()) + "\n")
        LOG.info("Correction run complete: %s", run_dir)
        return 0
    except Exception:
        LOG.error("Review correction failed:\n%s", traceback.format_exc())
        write_json(
            {
                "status": "failed",
                "failed_utc": now_iso(),
                "traceback": traceback.format_exc(),
            },
            run_dir / "00_RUN" / "FAILURE.json",
        )
        return 1
    finally:
        lock.release()


def export_existing(args: argparse.Namespace) -> int:
    repo_root = find_repo_root(SCRIPT_PATH.parent)
    config_path = args.config if args.config.is_absolute() else repo_root / args.config
    config = load_yaml(config_path)
    run_dir = args.run_dir.resolve()
    configure_logging(args.log_level, run_dir / "00_RUN" / "01_review_corrections.log")
    registry = load_registry(run_dir)
    result = export_publication_artifacts(run_dir, registry, config)
    write_json(result, run_dir / "00_RUN" / "export_result.json")
    LOG.info("Export complete")
    return 0


def validate_existing(args: argparse.Namespace) -> int:
    repo_root = find_repo_root(SCRIPT_PATH.parent)
    config_path = args.config if args.config.is_absolute() else repo_root / args.config
    config = load_yaml(config_path)
    run_dir = args.run_dir.resolve()
    configure_logging(args.log_level)
    result = validate_run(run_dir, config)
    print(json.dumps(result, indent=2, default=str))
    return 0 if result["status"] == "pass" else 1


def show_plan(args: argparse.Namespace) -> int:
    repo_root = find_repo_root(SCRIPT_PATH.parent)
    config_path = args.config if args.config.is_absolute() else repo_root / args.config
    config = load_yaml(config_path)
    specs = selected_specs(
        config,
        corrected_only=args.corrected_only,
        include_controls=args.include_controls,
        only=set(args.only) if args.only else None,
    )
    print("Selected correction scenarios:")
    for spec in specs:
        print(
            f"  {int(spec.get('order', 999)):02d}  {spec['id']:<45} "
            f"{spec.get('role', 'corrected')}"
        )
    disabled = discover_disabled_winter_monotemporal_configs(
        repo_root,
        list(
            config.get("review", {})
            .get("config_discovery", {})
            .get("globs", ["config*.yaml", "configs/**/*.yaml"])
        ),
    )
    print("\nRepository configurations with hunting disabled in winter monotemporal context:")
    for row in disabled:
        if row.get("status") == "disabled":
            print(f"  - {row.get('file')}: {row.get('scenario') or row.get('detail')}")
    return 0


def self_test(_: argparse.Namespace) -> int:
    import unittest

    suite = unittest.defaultTestLoader.discover(
        str(REPO_ROOT / "review" / "tests"),
        pattern="test_*.py",
    )
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Rerun and export the winter-monotemporal hunting correction."
    )
    sub = parser.add_subparsers(dest="command")

    def common(p: argparse.ArgumentParser) -> None:
        p.add_argument(
            "--config",
            type=Path,
            default=Path("review/config/review-corrections.yaml"),
        )
        p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    plan = sub.add_parser("plan", help="Audit repository configs and show the planned scenarios")
    common(plan)
    plan.add_argument("--corrected-only", action="store_true")
    plan.add_argument("--include-controls", action="store_true")
    plan.add_argument("--only", nargs="*")
    plan.set_defaults(func=show_plan)

    run = sub.add_parser("run", help="Execute the correction workflow")
    common(run)
    run.add_argument("--output-root", type=Path)
    run.add_argument("--run-id")
    run.add_argument("--resume", type=Path)
    run.add_argument("--dry-run", action="store_true")
    run.add_argument("--corrected-only", action="store_true")
    run.add_argument("--include-controls", action="store_true")
    run.add_argument("--only", nargs="*")
    run.add_argument("--skip-feature-importance", action="store_true")
    run.add_argument("--skip-novelty", action="store_true")
    run.add_argument("--force-scenarios", action="store_true")
    run.add_argument("--force-diagnostics", action="store_true")
    run.add_argument("--force-lock", action="store_true")
    run.add_argument("--continue-after-audit-failure", action="store_true")
    run.add_argument("--allow-validation-failure", action="store_true")
    run.add_argument("--no-archive", action="store_true")
    run.set_defaults(func=execute_run)

    export = sub.add_parser("export", help="Re-export artifacts from a completed run")
    common(export)
    export.add_argument("--run-dir", required=True, type=Path)
    export.set_defaults(func=export_existing)

    validate = sub.add_parser("validate", help="Validate a completed run")
    common(validate)
    validate.add_argument("--run-dir", required=True, type=Path)
    validate.set_defaults(func=validate_existing)

    tests = sub.add_parser("self-test", help="Run package unit tests")
    tests.set_defaults(func=self_test)

    return parser


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    known = {"plan", "run", "export", "validate", "self-test", "-h", "--help"}
    if not argv:
        argv = ["run"]
    elif argv[0] not in known:
        # Preserve convenient historical usage: options without a subcommand mean run.
        argv.insert(0, "run")
    parser = build_parser()
    args = parser.parse_args(argv)
    if not hasattr(args, "func"):
        parser.print_help()
        return 2
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
