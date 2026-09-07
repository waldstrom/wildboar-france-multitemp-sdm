from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from review.scripts.common import load_yaml, write_csv, write_json


def validate_run(run_dir: Path, config: dict[str, Any]) -> dict[str, Any]:
    run_dir = Path(run_dir)
    issues: list[dict[str, Any]] = []
    checks: list[dict[str, Any]] = []

    registry_path = run_dir / "00_RUN" / "scenario_registry.json"
    if not registry_path.exists():
        issues.append({"severity": "error", "check": "registry", "detail": "scenario_registry.json missing"})
        registry = []
    else:
        registry = json.loads(registry_path.read_text(encoding="utf-8"))

    corrected = [r for r in registry if r.get("role") == "corrected"]
    for record in corrected:
        complete = Path(record["scenario_dir"]) / "COMPLETE"
        checks.append(
            {
                "check": "corrected_scenario_complete",
                "scenario": record["id"],
                "pass": complete.exists(),
                "detail": str(complete),
            }
        )
        if not complete.exists():
            issues.append(
                {
                    "severity": "error",
                    "check": "corrected_scenario_complete",
                    "detail": record["id"],
                }
            )
        cfg = load_yaml(Path(record["resolved_config"]))
        hunting = bool(cfg.get("multitemporal", {}).get("include_hunting", True))
        checks.append(
            {
                "check": "resolved_hunting_true",
                "scenario": record["id"],
                "pass": hunting,
                "detail": hunting,
            }
        )
        if not hunting:
            issues.append(
                {
                    "severity": "error",
                    "check": "resolved_hunting_true",
                    "detail": record["id"],
                }
            )

    aggregate_audit = run_dir / "00_RUN" / "scenario_audit_summary.json"
    if aggregate_audit.exists():
        audits = json.loads(aggregate_audit.read_text(encoding="utf-8"))
        for audit in audits:
            required = next(
                (r for r in registry if r["id"] == audit.get("scenario")),
                {},
            ).get("role") in {"corrected", "reference"}
            passed = audit.get("status") == "pass"
            checks.append(
                {
                    "check": "scenario_audit",
                    "scenario": audit.get("scenario"),
                    "pass": passed,
                    "detail": " | ".join(audit.get("issues", [])),
                }
            )
            if required and not passed:
                issues.append(
                    {
                        "severity": "error",
                        "check": "scenario_audit",
                        "detail": f"{audit.get('scenario')}: {audit.get('issues')}",
                    }
                )
    else:
        issues.append({"severity": "error", "check": "scenario_audit", "detail": "aggregate audit missing"})

    artifact_status = run_dir / "00_RUN" / "artifact_status.csv"
    if artifact_status.exists():
        frame = pd.read_csv(artifact_status)
        for _, row in frame.iterrows():
            status = str(row.get("status", ""))
            passed = status == "generated"
            checks.append(
                {
                    "check": "artifact_generated",
                    "scenario": row.get("publication_label"),
                    "pass": passed,
                    "detail": status,
                }
            )
            # Feature importance and novelty may be explicitly skipped at CLI level.
            optional = status == "skipped_or_missing"
            if not passed and not optional:
                issues.append(
                    {
                        "severity": "error",
                        "check": "artifact_generated",
                        "detail": f"{row.get('publication_label')}: {status}",
                    }
                )
    else:
        issues.append({"severity": "error", "check": "artifact_status", "detail": "artifact_status.csv missing"})

    for relative in [
        "03_MANUSCRIPT/01_Table_2_LOYO_model_performance",
        "04_SUPPLEMENT/01_Supplement_S5_corrected_configuration",
        "04_SUPPLEMENT/02_Supplement_S6_foldwise_performance_and_CI",
        "04_SUPPLEMENT/03_Supplement_S7_feature_provenance_and_hunting_audit",
    ]:
        path = run_dir / relative
        passed = path.exists() and any(p.is_file() and p.stat().st_size > 0 for p in path.rglob("*"))
        checks.append({"check": "required_output_tree", "scenario": relative, "pass": passed, "detail": str(path)})
        if not passed:
            issues.append({"severity": "error", "check": "required_output_tree", "detail": relative})

    result = {
        "status": "pass" if not any(i["severity"] == "error" for i in issues) else "fail",
        "run_dir": str(run_dir.resolve()),
        "checks": checks,
        "issues": issues,
    }
    write_json(result, run_dir / "00_RUN" / "validation_report.json")
    write_csv(checks, run_dir / "00_RUN" / "validation_checks.csv")
    write_csv(issues, run_dir / "00_RUN" / "validation_issues.csv")
    return result
