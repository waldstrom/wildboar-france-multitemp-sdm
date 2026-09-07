from __future__ import annotations

import copy
import difflib
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from review.scripts.common import (
    deep_get,
    deep_merge,
    dump_yaml,
    flatten_mapping,
    load_yaml,
    relative_or_absolute,
    stable_hash,
    write_json,
)

LOG = logging.getLogger("review_corrections.configuration")


@dataclass
class ResolvedScenario:
    id: str
    order: int
    role: str
    dataset: str
    temporal: str
    runner: str
    source_config: str
    source_scenario_label: str | None
    source_scenario_key: str | None
    scenario_dir: str
    resolved_config: str
    experiment_dir: str
    season: str
    years: list[int]
    engines: list[str]
    include_hunting_before: bool | None
    include_hunting_after: bool
    enabled: bool
    config_hash: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _iter_scenarios(raw: Any) -> Iterable[dict[str, Any]]:
    if isinstance(raw, list):
        for item in raw:
            if isinstance(item, dict):
                yield item
    elif isinstance(raw, dict):
        for key, item in raw.items():
            if isinstance(item, dict):
                item = copy.deepcopy(item)
                item.setdefault("key", key)
                yield item


def _normalise_season(value: Any) -> str:
    return str(value or "").strip().casefold()


def _normalise_runner(value: Any) -> str:
    raw = str(value or "").strip().replace("\\", "/").casefold()
    if "/" in raw or raw.endswith(".py"):
        return Path(raw).stem.casefold()
    if "." in raw:
        return raw.rsplit(".", 1)[-1]
    return raw


def _scenario_seasons(cfg: dict[str, Any]) -> list[str]:
    raw = deep_get(cfg, "multitemporal.seasons", [])
    if isinstance(raw, str):
        return [raw]
    return [str(v) for v in (raw or [])]


def select_aggregate_scenario(
    aggregate_path: Path,
    selector: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Select exactly one embedded scenario using semantic fields, not line numbers."""
    aggregate = load_yaml(aggregate_path)
    matches: list[dict[str, Any]] = []
    wanted_runner = str(selector.get("runner", "")).casefold()
    wanted_season = _normalise_season(selector.get("season"))
    wanted_key = str(selector.get("key", "")).casefold()
    wanted_label = str(selector.get("label_contains", "")).casefold()

    scenario_block = aggregate.get("scenarios")
    if scenario_block is None and isinstance(aggregate.get("meta_run"), dict):
        scenario_block = aggregate["meta_run"].get("scenarios")
    for item in _iter_scenarios(scenario_block):
        cfg = item.get("config") or item.get("cfg")
        if not isinstance(cfg, dict):
            continue
        runner = _normalise_runner(item.get("script", ""))
        seasons = {_normalise_season(v) for v in _scenario_seasons(cfg)}
        key = str(item.get("key", "")).casefold()
        label = str(item.get("label", "")).casefold()

        if wanted_runner and runner != _normalise_runner(wanted_runner):
            continue
        if wanted_season and wanted_season not in seasons:
            continue
        if wanted_key and key != wanted_key:
            continue
        if wanted_label and wanted_label not in label:
            continue
        matches.append(item)

    if len(matches) != 1:
        compact = [
            {
                "key": item.get("key"),
                "label": item.get("label"),
                "script": item.get("script"),
                "seasons": _scenario_seasons(item.get("config", {})),
            }
            for item in matches
        ]
        raise ValueError(
            f"Expected one scenario in {aggregate_path} for selector {selector}; "
            f"found {len(matches)}: {compact}"
        )
    item = matches[0]
    return copy.deepcopy(item["config"]), item


def _format_config_diff(before: dict[str, Any], after: dict[str, Any]) -> str:
    import yaml

    left = yaml.safe_dump(before, sort_keys=False, allow_unicode=True).splitlines()
    right = yaml.safe_dump(after, sort_keys=False, allow_unicode=True).splitlines()
    return "\n".join(
        difflib.unified_diff(
            left,
            right,
            fromfile="source_config.yaml",
            tofile="resolved_review_config.yaml",
            lineterm="",
        )
    ) + "\n"


def resolve_scenario(
    repo_root: Path,
    spec: dict[str, Any],
    scenario_dir: Path,
    *,
    run_id: str,
    default_engines: list[str],
    default_seed: int,
) -> ResolvedScenario:
    scenario_id = str(spec["id"])
    source = spec.get("source", {})
    source_file = repo_root / str(source.get("aggregate_config") or source.get("config"))
    if source.get("aggregate_config"):
        source_cfg, source_item = select_aggregate_scenario(
            source_file, source.get("selector", {})
        )
    else:
        source_cfg = load_yaml(source_file)
        source_item = {}

    before = copy.deepcopy(source_cfg)
    resolved = copy.deepcopy(source_cfg)
    deep_merge(resolved, copy.deepcopy(spec.get("patch", {})))

    runner = str(spec.get("runner") or source_item.get("script") or "").strip()
    if runner not in {"monotemporal", "multitemporal"}:
        raise ValueError(f"Unsupported runner for {scenario_id}: {runner!r}")

    temporal = str(spec.get("temporal") or runner)
    dataset = str(spec.get("dataset") or "allpoints")
    season = str(spec.get("season") or (_scenario_seasons(resolved) or ["Winter"])[0])
    years = [int(v) for v in deep_get(resolved, "multitemporal.years", [])]
    if not years:
        raise ValueError(f"{scenario_id} has no configured years")

    engines = [str(v).lower() for v in (spec.get("engines") or default_engines)]
    if not engines:
        raise ValueError(f"{scenario_id} has no model engines")

    include_before = deep_get(before, "multitemporal.include_hunting")
    include_after = bool(deep_get(resolved, "multitemporal.include_hunting", True))
    expected_hunting = bool(spec.get("expect_hunting", True))

    experiment_base = (scenario_dir / "experiments").resolve()
    prefix = "monotemp" if runner == "monotemporal" else "multitemp"
    experiment_name = scenario_id
    experiment_dir = experiment_base / f"{prefix}-{experiment_name}"

    dynamic_patch = {
        "experiment": {
            "name": experiment_name,
            "base_dir": str(experiment_base),
            "dir_prefix": prefix,
            "append_timestamp": False,
            "random_seed": int(
                deep_get(resolved, "experiment.random_seed", default_seed)
            ),
            "review_run_id": run_id,
            "review_scenario_id": scenario_id,
            "review_expect_hunting": expected_hunting,
        },
        "run_test_year": "all",
        "model": {"engines": engines},
        "variable_importance": {
            "permutation": {
                "enable": False,
            }
        },
        "multitemporal": {
            "stack_base_dir": str((experiment_dir / "data-stack").resolve()),
        },
    }
    deep_merge(resolved, dynamic_patch)

    # The central correction must never be lost through a malformed overlay.
    expected_hunting = bool(spec.get("expect_hunting", True))
    actual_hunting = bool(deep_get(resolved, "multitemporal.include_hunting", True))
    if actual_hunting != expected_hunting:
        raise ValueError(
            f"{scenario_id}: include_hunting={actual_hunting}, expected {expected_hunting}"
        )

    scenario_dir.mkdir(parents=True, exist_ok=True)
    source_snapshot = scenario_dir / "source_config.yaml"
    resolved_path = scenario_dir / "resolved_config.yaml"
    dump_yaml(before, source_snapshot)
    dump_yaml(resolved, resolved_path)
    (scenario_dir / "config.diff").write_text(
        _format_config_diff(before, resolved), encoding="utf-8"
    )

    before_flat = flatten_mapping(before)
    after_flat = flatten_mapping(resolved)
    diff_rows = []
    for key in sorted(set(before_flat) | set(after_flat)):
        if before_flat.get(key) != after_flat.get(key):
            diff_rows.append(
                {
                    "path": key,
                    "before": before_flat.get(key),
                    "after": after_flat.get(key),
                    "classification": (
                        "review_correction"
                        if key == "multitemporal.include_hunting"
                        else "run_isolation_or_declared_override"
                    ),
                }
            )
    write_json(diff_rows, scenario_dir / "config_diff.json")

    record = ResolvedScenario(
        id=scenario_id,
        order=int(spec.get("order", 999)),
        role=str(spec.get("role", "corrected")),
        dataset=dataset,
        temporal=temporal,
        runner=runner,
        source_config=relative_or_absolute(source_file, repo_root),
        source_scenario_label=source_item.get("label"),
        source_scenario_key=source_item.get("key"),
        scenario_dir=str(scenario_dir.resolve()),
        resolved_config=str(resolved_path.resolve()),
        experiment_dir=str(experiment_dir.resolve()),
        season=season,
        years=years,
        engines=engines,
        include_hunting_before=(
            bool(include_before) if include_before is not None else None
        ),
        include_hunting_after=actual_hunting,
        enabled=bool(spec.get("enabled", True)),
        config_hash=stable_hash(resolved),
    )
    write_json(record.to_dict(), scenario_dir / "scenario_registry.json")
    return record


def compare_candidate_configuration(
    mono_config: Path,
    multi_config: Path,
    allowed_prefixes: list[str],
) -> list[dict[str, Any]]:
    """Return all mono/multi differences and classify declared temporal differences."""
    mono = flatten_mapping(load_yaml(mono_config))
    multi = flatten_mapping(load_yaml(multi_config))
    rows: list[dict[str, Any]] = []
    for key in sorted(set(mono) | set(multi)):
        left, right = mono.get(key), multi.get(key)
        if left == right:
            continue
        allowed = any(key == prefix or key.startswith(prefix + ".") for prefix in allowed_prefixes)
        rows.append(
            {
                "path": key,
                "monotemporal": left,
                "multitemporal": right,
                "classification": "declared_temporal_difference" if allowed else "review_required",
            }
        )
    return rows


def discover_disabled_winter_monotemporal_configs(
    repo_root: Path,
    globs: list[str],
) -> list[dict[str, Any]]:
    """Inventory structurally detectable winter-monotemporal configs with hunting off."""
    findings: list[dict[str, Any]] = []
    seen: set[Path] = set()
    for pattern in globs:
        for path in repo_root.glob(pattern):
            if not path.is_file() or path in seen:
                continue
            seen.add(path)
            try:
                data = load_yaml(path)
            except Exception as exc:
                findings.append(
                    {
                        "file": str(path.relative_to(repo_root)),
                        "status": "parse_error",
                        "detail": str(exc),
                    }
                )
                continue

            top_is_mono = bool(data.get("monotemporal")) or "monotemporal" in path.name.lower()
            top_seasons = {_normalise_season(v) for v in _scenario_seasons(data)}
            if (
                top_is_mono
                and "winter" in top_seasons
                and deep_get(data, "multitemporal.include_hunting") is False
            ):
                findings.append(
                    {
                        "file": str(path.relative_to(repo_root)),
                        "scenario": data.get("experiment", {}).get("name"),
                        "yaml_path": "multitemporal.include_hunting",
                        "status": "disabled",
                        "detail": "standalone winter monotemporal configuration",
                    }
                )

            for item in _iter_scenarios(data.get("scenarios")):
                cfg = item.get("config", {})
                seasons = {_normalise_season(v) for v in _scenario_seasons(cfg)}
                if (
                    _normalise_runner(item.get("script", "")) == "monotemporal"
                    and "winter" in seasons
                    and deep_get(cfg, "multitemporal.include_hunting") is False
                ):
                    findings.append(
                        {
                            "file": str(path.relative_to(repo_root)),
                            "scenario": item.get("key") or item.get("label"),
                            "yaml_path": "scenarios[].config.multitemporal.include_hunting",
                            "status": "disabled",
                            "detail": item.get("label"),
                        }
                    )
    return findings
