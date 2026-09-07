"""Aggregate and compare permutation feature importances from ELAPID runs.

This helper script ingests configuration from a companion YAML file (see
``analyze_feature_importance_config.yaml`` in the same directory).  Each meta
run entry specifies the path to ``best_runs.csv`` *and* one or more directories
that store the corresponding permutation feature importance artefacts.  This is
necessary because ``best_runs.csv`` files are often generated in meta run output
folders while the feature importances live in adjacent mono- or multi-temporal
directories.

The script collects all permutation importance tables for ELAPID models, merges
and summarises them, optionally performs pairwise statistical comparisons
between selected meta runs (for example, "all points" versus "GBIF only"), and
can compute variance inflation factor (VIF) metrics in a dedicated post-
processing step.
"""
from __future__ import annotations

import argparse
import logging
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
import yaml
from tqdm.auto import tqdm

DEFAULT_CONFIG_PATH = Path(__file__).with_name("analyze_feature_importance_config.yaml")

log = logging.getLogger("feature_importance_analysis")


PREDICTOR_SAMPLE_FRACTION = 0.05
PREDICTOR_CONTAINER_NAMES = (
    "inputs",
    "input",
    "predictor_data",
    "predictor_tables",
    "predictors",
    "design",
    "data-stack",
    "data_stack",
    "stack",
    "stacks",
)

TEMPORAL_KEYWORDS: dict[str, tuple[str, ...]] = {
    "monotemporal": (
        "monotemporal",
        "mono_temporal",
        "mono-temporal",
        "monotemp",
        "mono-temp",
        "mono",
    ),
    "multitemporal": (
        "multitemporal",
        "multi_temporal",
        "multi-temporal",
        "multitemp",
        "multi-temp",
        "multi",
    ),
}


_YEAR_PATTERN = re.compile(r"(\d{4})")


@dataclass(frozen=True)
class FeatureImportanceRecord:
    """Container for a single feature importance observation."""

    meta_run: str
    meta_run_path: Path
    scenario: str | None
    season: str | None
    temporal: str | None
    engine: str | None
    test_year: int | None
    combo_name: str
    variable: str
    importance_mean: float
    importance_std: float | None
    relative_importance: float | None
    vif: float | None
    source_file: Path


@dataclass(frozen=True)
class MetaRunConfig:
    """Configuration for a single meta run input."""

    label: str
    best_runs_path: Path
    feature_importance_dirs: tuple[Path, ...]


@dataclass(frozen=True)
class AnalysisConfig:
    """User supplied configuration loaded from YAML."""

    meta_runs: dict[str, MetaRunConfig]
    comparisons: list[tuple[str, str]]
    output_directory: Path
    compute_vif: bool


@dataclass(frozen=True)
class VIFComputationRequest:
    """Container describing how to compute VIF values for a combo."""

    meta_run: str
    combo_name: str
    season: str | None
    temporal: str | None
    test_year: int | float | None
    variables: tuple[str, ...]
    feature_importance_dirs: tuple[Path, ...]
    predictor_roots: tuple[Path, ...]
    extra_roots: tuple[Path, ...]


def _sanitize_name(name: str) -> str:
    """Return ``name`` safe for use as a directory component."""

    return name.replace("/", "_")


def _load_best_runs(best_runs_path: Path) -> pd.DataFrame:
    """Load ``best_runs.csv`` from ``best_runs_path`` and normalise column names."""

    if best_runs_path.is_dir():
        best_runs_path = best_runs_path / "best_runs.csv"

    if not best_runs_path.exists():
        raise FileNotFoundError(f"best_runs.csv not found at {best_runs_path}")

    df = pd.read_csv(best_runs_path)
    # Standardise column cases to avoid surprises when building combo names.
    df.columns = [str(col).strip() for col in df.columns]
    required_cols = {"season", "temporal", "engine", "test_year"}
    missing = required_cols.difference(df.columns)
    if missing:
        raise ValueError(
            "best_runs.csv is missing required columns: " + ", ".join(sorted(missing))
        )
    return df


def _build_combo_name(season: str, temporal: str, engine: str, test_year: int | float) -> str:
    """Create the combination name used for feature importance folders."""

    season_lower = str(season).strip().lower().replace(" ", "_")
    temporal_name = str(temporal).strip()
    engine_name = str(engine).strip()
    year = int(test_year)
    return _sanitize_name(f"{season_lower}_{temporal_name}_{engine_name}_test_{year}")


def _season_candidates(season: str | None) -> tuple[str, ...]:
    """Return a set of potential directory names for ``season``."""

    if season is None:
        return tuple()

    base = str(season).strip()
    if not base:
        return tuple()

    candidates = {
        base,
        base.lower(),
        base.upper(),
        base.title(),
        base.replace(" ", "_"),
        base.lower().replace(" ", "_"),
    }
    return tuple(dict.fromkeys(candidate for candidate in candidates if candidate))


def _engine_candidates(engine: str | None) -> tuple[str, ...]:
    """Return a set of potential directory names for ``engine``."""

    if engine is None:
        return tuple()

    base = str(engine).strip()
    if not base:
        return tuple()

    candidates = {base, base.lower(), base.upper()}
    return tuple(dict.fromkeys(candidate for candidate in candidates if candidate))


def _prioritise_search_roots(
    search_roots: Sequence[Path], temporal: str | None
) -> list[Path]:
    """Return ``search_roots`` ordered to favour ``temporal`` specific directories."""

    if not temporal:
        return list(search_roots)

    temporal_key = str(temporal).strip().lower()
    if not temporal_key:
        return list(search_roots)

    keywords = TEMPORAL_KEYWORDS.get(temporal_key, (temporal_key,))
    opposing_keywords: tuple[str, ...] = tuple(
        kw
        for key, values in TEMPORAL_KEYWORDS.items()
        if key != temporal_key
        for kw in values
    )

    ranked: list[tuple[int, int, Path]] = []
    for index, root in enumerate(search_roots):
        root_str = str(root).lower()
        priority = 1
        if any(keyword and keyword in root_str for keyword in keywords):
            priority = 0
        elif opposing_keywords and any(
            keyword and keyword in root_str for keyword in opposing_keywords
        ):
            priority = 2
        ranked.append((priority, index, root))

    ranked.sort(key=lambda item: (item[0], item[1]))
    return [root for _, _, root in ranked]


def _locate_permutation_file(
    combo_name: str,
    search_roots: Sequence[Path],
    *,
    season: str | None = None,
    engine: str | None = None,
    test_year: int | float | None = None,
    temporal: str | None = None,
) -> Path | None:
    """Return the first matching permutation importance file across search roots."""

    for root in _prioritise_search_roots(search_roots, temporal):
        candidates = [
            root / combo_name / "permutation_importance.csv",
            root / "feature_importance" / combo_name / "permutation_importance.csv",
        ]

        if test_year is not None:
            year_component = f"test_{int(test_year)}"
        else:
            year_component = None

        if year_component:
            for season_dir in _season_candidates(season):
                for engine_dir in _engine_candidates(engine):
                    candidates.append(
                        root
                        / "output"
                        / season_dir
                        / year_component
                        / engine_dir
                        / "feature_importance"
                        / combo_name
                        / "permutation_importance.csv"
                    )
                    candidates.append(
                        root
                        / "output"
                        / season_dir
                        / year_component
                        / engine_dir
                        / "importance"
                        / "permutation_importance.csv"
                    )

        for candidate in candidates:
            if candidate.exists():
                return candidate

        # Fallback: expensive recursive glob used only when direct paths fail.
        try:
            match = next(
                candidate
                for candidate in root.glob(f"**/{combo_name}/permutation_importance.csv")
                if candidate.is_file()
            )
        except StopIteration:
            match = None

        if match:
            return match

    return None


def _discover_test_years(
    search_roots: Sequence[Path],
    season: str | None,
    engine: str | None,
    temporal: str | None,
) -> set[int]:
    """Return all test years with available importance artefacts for the combo."""

    years: set[int] = set()
    season_candidates = _season_candidates(season)
    if not season_candidates:
        return years

    engine_candidates = _engine_candidates(engine)

    for root in _prioritise_search_roots(search_roots, temporal):
        for season_dir in season_candidates:
            output_root = root / "output" / season_dir
            if not output_root.exists():
                continue

            for test_dir in output_root.iterdir():
                if not test_dir.is_dir():
                    continue

                match = _YEAR_PATTERN.search(test_dir.name)
                if not match:
                    continue

                year_val = int(match.group(1))
                if not engine_candidates:
                    years.add(year_val)
                    continue

                for engine_dir in engine_candidates:
                    engine_root = test_dir / engine_dir
                    if engine_root.exists():
                        years.add(year_val)
                        break

    return years


def _locate_vif_file(
    combo_name: str,
    search_roots: Sequence[Path],
    *,
    season: str | None = None,
    engine: str | None = None,
    test_year: int | float | None = None,
    temporal: str | None = None,
) -> Path | None:
    """Return the first matching VIF table for ``combo_name`` if present."""

    candidate_names = (
        "vif.csv",
        "vif_summary.csv",
        "variable_vif.csv",
    )

    for root in _prioritise_search_roots(search_roots, temporal):
        parents = [
            root / combo_name,
            root / "feature_importance" / combo_name,
        ]

        if test_year is not None:
            year_component = f"test_{int(test_year)}"
        else:
            year_component = None

        if year_component:
            for season_dir in _season_candidates(season):
                for engine_dir in _engine_candidates(engine):
                    parents.append(
                        root
                        / "output"
                        / season_dir
                        / year_component
                        / engine_dir
                        / "feature_importance"
                        / combo_name
                    )
                    parents.append(
                        root
                        / "output"
                        / season_dir
                        / year_component
                        / engine_dir
                        / "importance"
                    )

        for parent in parents:
            if not parent.exists():
                continue
            for name in candidate_names:
                candidate = parent / name
                if candidate.exists():
                    return candidate

        # Recursive fallback matching common patterns.
        for pattern in candidate_names:
            try:
                match = next(
                    candidate
                    for candidate in root.glob(
                        f"**/{combo_name}/**/{pattern}"
                    )
                    if candidate.is_file()
                )
            except StopIteration:
                match = None

            if match:
                return match

    return None


def _load_vif_mapping(vif_path: Path) -> dict[str, float]:
    """Return a mapping of variable name to VIF loaded from ``vif_path``."""

    try:
        df = pd.read_csv(vif_path)
    except Exception as exc:  # pragma: no cover - defensive logging
        log.warning("Failed to read VIF table %s: %s", vif_path, exc)
        return {}

    if df.empty:
        return {}

    columns = {str(col).strip().lower(): col for col in df.columns}
    variable_col = None
    for key in ("variable", "predictor", "name"):
        if key in columns:
            variable_col = columns[key]
            break

    vif_col = None
    for key in ("vif", "variance_inflation_factor"):
        if key in columns:
            vif_col = columns[key]
            break

    if variable_col is None or vif_col is None:
        log.warning(
            "VIF table %s is missing expected columns (found=%s)",
            vif_path,
            list(df.columns),
        )
        return {}

    mapping: dict[str, float] = {}
    for _, row in df.iterrows():
        var_name = str(row[variable_col]).strip()
        try:
            vif_value = float(row[vif_col])
        except (TypeError, ValueError):
            continue
        if not var_name:
            continue
        mapping[var_name] = vif_value

    return mapping


def _candidate_predictor_tables(parent: Path) -> Iterable[Path]:
    """Yield possible predictor tables used to compute VIFs."""

    if not parent.exists():
        return []

    directories = [parent]
    for name in ("predictors", "predictor_data", "predictor_tables", "design"):
        subdir = parent / name
        if subdir.exists():
            directories.append(subdir)

    patterns = (
        "*predictor*.parquet",
        "*predictor*.feather",
        "*predictor*.csv",
        "*design_matrix*.parquet",
        "*design_matrix*.feather",
        "*design_matrix*.csv",
        "*features*.parquet",
        "*features*.feather",
        "*features*.csv",
    )

    for directory in directories:
        for pattern in patterns:
            for path in sorted(directory.glob(pattern)):
                if path.is_file():
                    yield path


def _candidate_predictor_stacks(parent: Path) -> Iterable[Path]:
    """Yield predictor stack artefacts that may contain the raw features."""

    if not parent.exists():
        return []

    stack_patterns = (
        "*.zarr",
        "*.nc",
        "*.nc4",
        "*.cdf",
        "*stack*.zarr",
        "*predictor*.zarr",
    )

    for pattern in stack_patterns:
        for path in sorted(parent.glob(pattern)):
            if path.exists():
                yield path


def _iter_predictor_parents(
    root: Path,
    combo_name: str,
    *,
    season: str | None = None,
    engine: str | None = None,
    test_year: int | float | None = None,
) -> Iterable[Path]:
    """Yield directories that may contain predictor tables for ``combo_name``."""

    if not root.exists():
        return []

    candidates: list[Path] = []

    def _add(path: Path) -> None:
        if path.exists():
            candidates.append(path)

    _add(root / combo_name)
    _add(root / "feature_importance" / combo_name)

    for name in PREDICTOR_CONTAINER_NAMES:
        _add(root / name)
        _add(root / name / combo_name)

    year_component = f"test_{int(test_year)}" if test_year is not None else None
    if year_component:
        for season_dir in _season_candidates(season):
            season_root = root / "output" / season_dir / year_component
            _add(season_root)
            _add(season_root / combo_name)
            _add(season_root / "feature_importance" / combo_name)

            for name in PREDICTOR_CONTAINER_NAMES:
                _add(season_root / name)
                _add(season_root / name / combo_name)

            engine_candidates = _engine_candidates(engine) or (None,)
            for engine_dir in engine_candidates:
                engine_root = season_root / engine_dir if engine_dir else season_root
                _add(engine_root)
                _add(engine_root / combo_name)
                _add(engine_root / "feature_importance" / combo_name)

                for name in PREDICTOR_CONTAINER_NAMES:
                    _add(engine_root / name)
                    _add(engine_root / name / combo_name)

    seen: set[Path] = set()
    for candidate in candidates:
        if candidate not in seen:
            seen.add(candidate)
            yield candidate


def _load_predictor_stack(path: Path):
    """Return an opened xarray dataset for ``path`` if possible."""

    try:
        import xarray as xr
    except ImportError:  # pragma: no cover - optional dependency
        log.warning("xarray is required to load predictor stack %s", path)
        return None

    try:
        if path.is_dir() and (path.suffix == ".zarr" or (path / ".zarray").exists()):
            return xr.open_zarr(path, consolidated=False)
        if path.is_file() and path.suffix in {".nc", ".nc4", ".cdf"}:
            return xr.open_dataset(path)
    except Exception as exc:  # pragma: no cover - defensive logging
        log.warning("Failed to open predictor stack %s: %s", path, exc)
        return None

    return None


def _load_predictor_table(path: Path) -> pd.DataFrame | None:
    """Return a numeric predictor table loaded from ``path`` if possible."""

    try:
        if path.suffix == ".parquet":
            df = pd.read_parquet(path)
        elif path.suffix in {".feather", ".ft"}:
            df = pd.read_feather(path)
        elif path.suffix == ".csv":
            df = pd.read_csv(path)
        else:
            return None
    except Exception as exc:  # pragma: no cover - defensive logging
        log.warning("Failed to read predictor table %s: %s", path, exc)
        return None

    if df.empty:
        return None

    numeric = df.select_dtypes(include=[np.number])
    if numeric.empty:
        return None

    return numeric


def _vif_from_dataset(
    dataset,
    relevant_vars: Sequence[str],
    *,
    sample_fraction: float,
    random_seed: int = 42,
) -> tuple[np.ndarray, list[str]] | None:
    """Return a sampled predictor matrix and variable names from ``dataset``.

    ``dataset`` must be an xarray Dataset. Only variables listed in
    ``relevant_vars`` are retained. Returns ``None`` if a usable matrix cannot
    be created (for example because fewer than two variables are available).
    """

    try:
        import xarray as xr
    except ImportError:  # pragma: no cover - optional dependency
        return None

    if len(relevant_vars) < 2:
        return None

    subset = dataset[relevant_vars]
    if not isinstance(subset, xr.Dataset) or not subset.data_vars:
        return None

    array = subset.to_array(dim="variable")
    dims_to_stack = [dim for dim in array.dims if dim != "variable"]
    if not dims_to_stack:
        return None

    stacked = array.stack(sample=dims_to_stack).transpose("sample", "variable")
    total_rows = stacked.sizes.get("sample", 0)
    if total_rows < 2:
        return None

    sample_size = min(
        total_rows,
        max(2, int(round(total_rows * max(sample_fraction, 0)))),
    )

    if sample_size < total_rows:
        rng = np.random.default_rng(random_seed)
        indices = np.sort(rng.choice(total_rows, size=sample_size, replace=False))
        stacked = stacked.isel(sample=indices)
        log.debug(
            "Sampling %d of %d stack cells (%.2f%%) for VIF calculation",
            sample_size,
            total_rows,
            sample_size / total_rows * 100 if total_rows else 0,
        )
    matrix = stacked.to_numpy()
    if matrix.size == 0:
        return None

    matrix = matrix.astype(float, copy=False)
    matrix[~np.isfinite(matrix)] = np.nan
    finite_mask = np.all(np.isfinite(matrix), axis=1)
    matrix = matrix[finite_mask]
    if sample_size > matrix.shape[0]:
        log.debug(
            "Dropped %d rows with missing values when preparing stack for VIF",
            sample_size - matrix.shape[0],
        )
    if matrix.shape[0] < 2:
        return None

    std = np.nanstd(matrix, axis=0, ddof=0)
    valid_cols = [i for i, val in enumerate(std) if val > 0]
    if len(valid_cols) < 2:
        return None

    matrix = matrix[:, valid_cols]
    columns = [str(stacked.variable.values[i]) for i in valid_cols]

    return matrix, columns


def _compute_vif_from_predictors(
    combo_name: str,
    search_roots: Sequence[Path],
    variables: Iterable[str],
    *,
    season: str | None = None,
    engine: str | None = None,
    test_year: int | float | None = None,
    extra_roots: Sequence[Path] = (),
    temporal: str | None = None,
) -> dict[str, float]:
    """Compute VIF values directly from predictor matrices when tables are missing."""

    variable_set = {str(var).strip() for var in variables if str(var).strip()}
    if not variable_set:
        return {}

    root_candidates = list(dict.fromkeys([*search_roots, *extra_roots]))
    root_candidates = _prioritise_search_roots(root_candidates, temporal)

    tried_stacks: set[Path] = set()

    for root in root_candidates:
        for parent in _iter_predictor_parents(
            root,
            combo_name,
            season=season,
            engine=engine,
            test_year=test_year,
        ):
            for table_path in _candidate_predictor_tables(parent):
                predictors = _load_predictor_table(table_path)
                if predictors is None:
                    continue

                relevant_cols = [col for col in predictors.columns if col in variable_set]
                if len(relevant_cols) < 2:
                    continue

                matrix = (
                    predictors[relevant_cols]
                    .replace([np.inf, -np.inf], np.nan)
                    .dropna()
                )
                if matrix.empty:
                    continue

                non_constant_cols = [
                    col for col in matrix.columns if matrix[col].std(ddof=0) > 0
                ]
                if len(non_constant_cols) < 2:
                    continue

                matrix = matrix[non_constant_cols]

                row_count = matrix.shape[0]
                if row_count < 2:
                    continue

                sample_size = min(
                    row_count,
                    max(2, int(round(row_count * PREDICTOR_SAMPLE_FRACTION))),
                )
                if sample_size < row_count:
                    matrix = matrix.sample(n=sample_size, random_state=42)
                    log.debug(
                        "Sampling %d of %d rows (%.2f%%) from %s for VIF computation",
                        sample_size,
                        row_count,
                        sample_size / row_count * 100,
                        table_path,
                    )
                else:
                    sample_size = row_count

                design = add_constant(matrix.to_numpy(copy=True), has_constant="add")

                vif_mapping: dict[str, float] = {}
                for idx, col in enumerate(matrix.columns, start=1):
                    try:
                        vif_val = float(variance_inflation_factor(design, idx))
                    except Exception as exc:  # pragma: no cover - defensive logging
                        log.warning(
                            "Failed to compute VIF for %s in %s: %s", col, table_path, exc
                        )
                        continue
                    if np.isnan(vif_val) or np.isinf(vif_val):
                        continue
                    vif_mapping[col] = vif_val

                if vif_mapping:
                    log.info(
                        "Computed VIF values for combo '%s' using predictor table %s (n=%d)",
                        combo_name,
                        table_path,
                        sample_size,
                    )
                    return vif_mapping

            for stack_path in _candidate_predictor_stacks(parent):
                if stack_path in tried_stacks:
                    continue
                tried_stacks.add(stack_path)

                dataset = _load_predictor_stack(stack_path)
                if dataset is None:
                    continue

                try:
                    relevant_vars = [
                        str(var)
                        for var in dataset.data_vars
                        if str(var) in variable_set and str(var)
                    ]
                    result = _vif_from_dataset(
                        dataset,
                        relevant_vars,
                        sample_fraction=PREDICTOR_SAMPLE_FRACTION,
                    )
                finally:
                    close = getattr(dataset, "close", None)
                    if callable(close):
                        close()

                if not result:
                    continue

                matrix_np, columns = result
                design = add_constant(matrix_np, has_constant="add")
                vif_mapping: dict[str, float] = {}
                for idx, col in enumerate(columns, start=1):
                    try:
                        vif_val = float(variance_inflation_factor(design, idx))
                    except Exception as exc:  # pragma: no cover - defensive logging
                        log.warning(
                            "Failed to compute VIF for %s in %s: %s", col, stack_path, exc
                        )
                        continue
                    if np.isnan(vif_val) or np.isinf(vif_val):
                        continue
                    vif_mapping[col] = vif_val

                if vif_mapping:
                    log.info(
                        "Computed VIF values for combo '%s' using predictor stack %s (n=%d)",
                        combo_name,
                        stack_path,
                        matrix_np.shape[0],
                    )
                    return vif_mapping

    log.warning(
        "Unable to compute VIF values for combo '%s' - no suitable predictor data found",
        combo_name,
    )
    return {}


def collect_feature_importances(
    meta_config: MetaRunConfig,
) -> tuple[list[FeatureImportanceRecord], list[VIFComputationRequest]]:
    """Load all permutation importance tables for ELAPID best runs.

    Returns a tuple of feature importance records and VIF computation requests.
    """

    log.info("Collecting feature importances for meta run '%s'", meta_config.label)
    best_df = _load_best_runs(meta_config.best_runs_path)
    records: list[FeatureImportanceRecord] = []

    feature_dirs = list(meta_config.feature_importance_dirs)
    best_runs_root = (
        meta_config.best_runs_path
        if meta_config.best_runs_path.is_dir()
        else meta_config.best_runs_path.parent
    )
    extra_roots: list[Path] = []
    for candidate in (best_runs_root, best_runs_root.parent):
        if not candidate.exists():
            continue
        if candidate not in feature_dirs:
            feature_dirs.append(candidate)
        if candidate not in extra_roots:
            extra_roots.append(candidate)
    predictor_roots = tuple(dict.fromkeys(feature_dirs))

    engine_series = best_df.get("engine")
    if engine_series is None:
        log.warning("best_runs.csv for '%s' has no engine column", meta_config.label)
        return []

    engine_mask = engine_series.astype(str).str.strip().str.lower() == "elapid"
    elapid_df = best_df.loc[engine_mask].copy()

    partial_records: list[dict[str, object]] = []
    combo_details: dict[str, dict[str, object]] = {}
    processed_combos: set[str] = set()

    if elapid_df.empty:
        log.warning("No ELAPID feature importance records collected for '%s'", meta_config.label)
        return [], []

    for row in tqdm(
        elapid_df.to_dict(orient="records"),
        total=len(elapid_df),
        desc="Loading permutation importances",
    ):
        season = row.get("season")
        temporal = row.get("temporal")
        scenario = row.get("scenario") if "scenario" in row else None
        test_year = row.get("test_year")

        if pd.isna(season) or pd.isna(temporal):
            log.warning(
                "Skipping row with incomplete identifiers in %s: %s",
                meta_config.best_runs_path,
                row,
            )
            continue

        temporal_value = str(temporal) if temporal is not None else None
        discovered_years = _discover_test_years(
            meta_config.feature_importance_dirs,
            season,
            "elapid",
            temporal_value,
        )

        base_year: int | None = None
        if test_year is not None and not pd.isna(test_year):
            try:
                base_year = int(test_year)
                discovered_years.add(base_year)
            except (TypeError, ValueError):  # pragma: no cover - defensive
                log.warning(
                    "Unable to interpret test year '%s' for row in %s",
                    test_year,
                    meta_config.best_runs_path,
                )

        if not discovered_years:
            log.warning(
                "No test year directories found for %s (season=%s, temporal=%s)",
                meta_config.label,
                season,
                temporal,
            )
            continue

        for year in sorted(discovered_years):
            combo_name = _build_combo_name(str(season), str(temporal), "elapid", year)
            if combo_name in processed_combos:
                continue

            csv_path = _locate_permutation_file(
                combo_name,
                meta_config.feature_importance_dirs,
                season=season,
                engine="elapid",
                test_year=year,
                temporal=temporal_value,
            )
            if csv_path is None:
                log.warning(
                    "Missing permutation importance file for combo '%s' (test year %s) in %s",
                    combo_name,
                    year,
                    ", ".join(str(path) for path in meta_config.feature_importance_dirs),
                )
                continue

            if base_year is None or year != base_year:
                log.debug(
                    "Discovered permutation importance for additional test year %s at %s",
                    year,
                    csv_path,
                )

            fi_df = pd.read_csv(csv_path)
            if fi_df.empty:
                log.warning("Permutation importance table is empty: %s", csv_path)
                continue

            if "variable" not in fi_df.columns or "importance_mean" not in fi_df.columns:
                log.warning("Unexpected format for %s", csv_path)
                continue

            total_importance = fi_df["importance_mean"].sum()
            if total_importance:
                relative = fi_df["importance_mean"] / total_importance
            else:
                relative = pd.Series(np.nan, index=fi_df.index)

            combo_entry = combo_details.setdefault(
                combo_name,
                {
                    "season": season,
                    "temporal": temporal,
                    "test_year": year,
                    "variables": set(),
                },
            )

            for idx, fi_row in fi_df.iterrows():
                variable = str(fi_row["variable"]).strip()
                if not variable:
                    continue
                combo_entry["variables"].add(variable)
                mean_val = float(fi_row["importance_mean"])
                std_val = (
                    float(fi_row["importance_std"]) if "importance_std" in fi_df.columns else np.nan
                )
                rel_val = float(relative.iloc[idx]) if not pd.isna(relative.iloc[idx]) else np.nan
                partial_records.append(
                    {
                        "meta_run": meta_config.label,
                        "meta_run_path": meta_config.best_runs_path.parent,
                        "scenario": str(scenario) if scenario is not None else None,
                        "season": str(season) if season is not None else None,
                        "temporal": str(temporal) if temporal is not None else None,
                        "engine": "elapid",
                        "test_year": int(year),
                        "combo_name": combo_name,
                        "variable": variable,
                        "importance_mean": mean_val,
                        "importance_std": std_val,
                        "relative_importance": rel_val,
                        "source_file": csv_path,
                    }
                )

            processed_combos.add(combo_name)

    if not partial_records:
        log.warning("No ELAPID feature importance records collected for '%s'", meta_config.label)
        return [], []

    vif_requests: list[VIFComputationRequest] = []

    for record_data in partial_records:
        combo_name = str(record_data["combo_name"])
        variable = str(record_data["variable"])
        records.append(
            FeatureImportanceRecord(
                meta_run=str(record_data["meta_run"]),
                meta_run_path=record_data["meta_run_path"],
                scenario=record_data["scenario"],
                season=record_data["season"],
                temporal=record_data["temporal"],
                engine=str(record_data["engine"]),
                test_year=int(record_data["test_year"]),
                combo_name=combo_name,
                variable=variable,
                importance_mean=float(record_data["importance_mean"]),
                importance_std=float(record_data["importance_std"])
                if not pd.isna(record_data["importance_std"])
                else np.nan,
                relative_importance=float(record_data["relative_importance"])
                if not pd.isna(record_data["relative_importance"])
                else np.nan,
                vif=np.nan,
                source_file=record_data["source_file"],
            )
        )

    for combo_name, details in combo_details.items():
        variables = tuple(sorted(str(var) for var in details.get("variables", set())))
        vif_requests.append(
            VIFComputationRequest(
                meta_run=meta_config.label,
                combo_name=combo_name,
                season=str(details.get("season")) if details.get("season") is not None else None,
                temporal=str(details.get("temporal")) if details.get("temporal") is not None else None,
                test_year=details.get("test_year"),
                variables=variables,
                feature_importance_dirs=meta_config.feature_importance_dirs,
                predictor_roots=predictor_roots,
                extra_roots=tuple(extra_roots),
            )
        )

    log.info(
        "Collected %d feature importance rows for '%s'",
        len(records),
        meta_config.label,
    )
    return records, vif_requests


def records_to_dataframe(records: Sequence[FeatureImportanceRecord]) -> pd.DataFrame:
    """Convert feature importance records to a tidy :class:`pandas.DataFrame`."""

    if not records:
        return pd.DataFrame(
            columns=
            [
                "meta_run",
                "meta_run_path",
                "scenario",
                "season",
                "temporal",
                "engine",
                "test_year",
                "combo_name",
                "variable",
                "importance_mean",
                "importance_std",
                "relative_importance",
                "vif",
                "source_file",
            ]
        )

    data = [
        {
            "meta_run": rec.meta_run,
            "meta_run_path": str(rec.meta_run_path),
            "scenario": rec.scenario,
            "season": rec.season,
            "temporal": rec.temporal,
            "engine": rec.engine,
            "test_year": rec.test_year,
            "combo_name": rec.combo_name,
            "variable": rec.variable,
            "importance_mean": rec.importance_mean,
            "importance_std": rec.importance_std,
            "relative_importance": rec.relative_importance,
            "vif": rec.vif,
            "source_file": str(rec.source_file),
        }
        for rec in records
    ]
    return pd.DataFrame(data)


def _summarise_by_variable(df: pd.DataFrame, value_column: str) -> pd.DataFrame:
    """Return summary statistics grouped by meta run and variable."""

    grouped = df.dropna(subset=[value_column]).groupby(["meta_run", "variable"], dropna=False)
    summary = grouped[value_column].agg(["mean", "std", "median", "count"])
    summary = summary.rename(
        columns={
            "mean": f"{value_column}_mean",
            "std": f"{value_column}_std",
            "median": f"{value_column}_median",
            "count": "n",
        }
    ).reset_index()

    def _cv(row: pd.Series) -> float:
        mean_val = row[f"{value_column}_mean"]
        std_val = row[f"{value_column}_std"]
        if np.isnan(mean_val) or mean_val == 0:
            return np.nan
        return std_val / mean_val

    summary[f"{value_column}_cv"] = summary.apply(_cv, axis=1)
    return summary


def _welch_statistics(values_a: Sequence[float], values_b: Sequence[float]) -> dict[str, float]:
    """Compute Welch's t-test statistics and confidence interval."""

    arr_a = np.asarray(list(values_a), dtype=float)
    arr_b = np.asarray(list(values_b), dtype=float)
    n_a = arr_a.size
    n_b = arr_b.size

    if n_a == 0 or n_b == 0:
        return {
            "t_stat": np.nan,
            "p_value": np.nan,
            "df": np.nan,
            "se": np.nan,
            "ci_low": np.nan,
            "ci_high": np.nan,
        }

    mean_a = float(arr_a.mean())
    mean_b = float(arr_b.mean())
    var_a = float(arr_a.var(ddof=1)) if n_a > 1 else 0.0
    var_b = float(arr_b.var(ddof=1)) if n_b > 1 else 0.0
    diff = mean_a - mean_b
    se = math.sqrt(var_a / n_a + var_b / n_b)

    if se == 0:
        t_stat = 0.0
        p_value = 1.0
        df = np.nan
        ci_low = diff
        ci_high = diff
    else:
        t_stat, p_value = stats.ttest_ind(arr_a, arr_b, equal_var=False)
        numerator = (var_a / n_a + var_b / n_b) ** 2
        denominator = 0.0
        if n_a > 1:
            denominator += ((var_a / n_a) ** 2) / (n_a - 1)
        if n_b > 1:
            denominator += ((var_b / n_b) ** 2) / (n_b - 1)
        df = numerator / denominator if denominator > 0 else np.nan
        if np.isnan(df):
            ci_low = np.nan
            ci_high = np.nan
        else:
            t_crit = stats.t.ppf(0.975, df)
            ci_low = diff - t_crit * se
            ci_high = diff + t_crit * se

    return {
        "t_stat": t_stat,
        "p_value": p_value,
        "df": df,
        "se": se,
        "ci_low": ci_low,
        "ci_high": ci_high,
    }


def _effect_size(values_a: Sequence[float], values_b: Sequence[float]) -> float:
    """Compute Cohen's d for two independent samples."""

    arr_a = np.asarray(list(values_a), dtype=float)
    arr_b = np.asarray(list(values_b), dtype=float)
    n_a = arr_a.size
    n_b = arr_b.size

    if n_a < 2 or n_b < 2:
        return np.nan

    var_a = arr_a.var(ddof=1)
    var_b = arr_b.var(ddof=1)
    pooled_var = ((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2)
    if pooled_var <= 0:
        return np.nan
    return (arr_a.mean() - arr_b.mean()) / math.sqrt(pooled_var)


def _effect_size_label(effect: float) -> str:
    """Return a qualitative interpretation of Cohen's d."""

    if np.isnan(effect):
        return "undetermined"
    magnitude = abs(effect)
    if magnitude < 0.2:
        return "negligible"
    if magnitude < 0.5:
        return "small"
    if magnitude < 0.8:
        return "medium"
    return "large"


def compare_meta_runs(
    df: pd.DataFrame,
    run_a: str,
    run_b: str,
    value_column: str,
) -> pd.DataFrame:
    """Compare variable importances between two meta runs."""

    subset = df.dropna(subset=[value_column])
    data_a = subset[subset["meta_run"] == run_a]
    data_b = subset[subset["meta_run"] == run_b]

    variables = sorted(set(data_a["variable"]).intersection(data_b["variable"]))
    results: list[dict[str, float | str]] = []

    for variable in variables:
        vals_a = data_a[data_a["variable"] == variable][value_column]
        vals_b = data_b[data_b["variable"] == variable][value_column]
        if vals_a.empty or vals_b.empty:
            continue

        stats_info = _welch_statistics(vals_a, vals_b)
        effect = _effect_size(vals_a, vals_b)
        mean_a = float(vals_a.mean())
        mean_b = float(vals_b.mean())
        diff = mean_a - mean_b
        percent = np.nan if np.isnan(mean_b) or mean_b == 0 else diff / mean_b * 100
        vif_vals_a = data_a[data_a["variable"] == variable]["vif"].dropna()
        vif_vals_b = data_b[data_b["variable"] == variable]["vif"].dropna()
        vif_a = float(vif_vals_a.mean()) if not vif_vals_a.empty else np.nan
        vif_b = float(vif_vals_b.mean()) if not vif_vals_b.empty else np.nan

        results.append(
            {
                "variable": variable,
                f"{value_column}_{run_a}": mean_a,
                f"{value_column}_{run_b}": mean_b,
                "difference": diff,
                "percent_difference": percent,
                f"vif_{run_a}": vif_a,
                f"vif_{run_b}": vif_b,
                "t_stat": stats_info["t_stat"],
                "p_value": stats_info["p_value"],
                "degrees_of_freedom": stats_info["df"],
                "standard_error": stats_info["se"],
                "ci_low": stats_info["ci_low"],
                "ci_high": stats_info["ci_high"],
                "cohens_d": effect,
                "effect_size_label": _effect_size_label(effect),
                "n_obs_run_a": int(len(vals_a)),
                "n_obs_run_b": int(len(vals_b)),
            }
        )

    return pd.DataFrame(results)


def _ensure_output_dir(path: Path) -> Path:
    """Create ``path`` if required and return it."""

    path.mkdir(parents=True, exist_ok=True)
    return path


def _compare_temporal_modes(df: pd.DataFrame, value_column: str) -> pd.DataFrame:
    """Compare multitemporal and monotemporal importances within ``df``."""

    if "temporal" not in df.columns:
        return pd.DataFrame()

    subset = df.dropna(subset=[value_column]).copy()
    if subset.empty:
        return pd.DataFrame()

    subset = subset.dropna(subset=["temporal"]).copy()
    if subset.empty:
        return pd.DataFrame()

    subset["meta_run"] = subset["temporal"].astype(str).str.strip().str.lower()
    available = {str(val) for val in subset["meta_run"].unique()}
    if {"multitemporal", "monotemporal"}.issubset(available):
        return compare_meta_runs(subset, "multitemporal", "monotemporal", value_column)
    return pd.DataFrame()


def _write_analysis_outputs(
    df: pd.DataFrame,
    output_dir: Path,
    comparisons: Sequence[tuple[str, str]],
    *,
    include_temporal_comparison: bool = False,
) -> None:
    """Persist combined tables, summaries, and comparisons for ``df``."""

    if df.empty:
        log.warning("No feature importance data available for %s", output_dir)
        return

    out_dir = _ensure_output_dir(output_dir)
    combined_path = out_dir / "feature_importance_all_records.csv"
    df.to_csv(combined_path, index=False)
    log.info("Saved combined feature importance table to %s", combined_path)

    vif_summary = pd.DataFrame()
    if "vif" in df.columns and df["vif"].notna().any():
        vif_summary = _summarise_by_variable(df, "vif")
        if not vif_summary.empty:
            vif_summary = vif_summary.rename(columns={"n": "vif_n"})

    for value_column in ("importance_mean", "relative_importance"):
        summary = _summarise_by_variable(df, value_column)
        if not summary.empty and not vif_summary.empty:
            summary = summary.merge(
                vif_summary[
                    [
                        "meta_run",
                        "variable",
                        "vif_mean",
                        "vif_std",
                        "vif_median",
                        "vif_cv",
                        "vif_n",
                    ]
                ],
                on=["meta_run", "variable"],
                how="left",
            )
        summary_path = out_dir / f"summary_{value_column}.csv"
        summary.to_csv(summary_path, index=False)
        log.info("Saved %s summary to %s", value_column, summary_path)

    available_runs = {str(run) for run in df["meta_run"].dropna().unique()}
    if comparisons:
        for run_a, run_b in comparisons:
            if run_a not in available_runs or run_b not in available_runs:
                log.warning(
                    "Comparison (%s vs %s) skipped for %s because one of the meta runs has no data.",
                    run_a,
                    run_b,
                    output_dir,
                )
                continue

            for value_column in ("importance_mean", "relative_importance"):
                comparison = compare_meta_runs(df, run_a, run_b, value_column)
                if comparison.empty:
                    log.warning(
                        "No overlapping variables for comparison between %s and %s (%s) in %s.",
                        run_a,
                        run_b,
                        value_column,
                        output_dir,
                    )
                    continue
                comparison_path = (
                    out_dir
                    / f"comparison_{run_a}_vs_{run_b}_{value_column}.csv"
                )
                comparison.to_csv(comparison_path, index=False)
                log.info(
                    "Saved %s comparison between %s and %s to %s",
                    value_column,
                    run_a,
                    run_b,
                    comparison_path,
                )

                top_diff = (
                    comparison.reindex(
                        comparison["difference"].abs().sort_values(ascending=False).index
                    )
                    .head(5)
                    .copy()
                )
                if not top_diff.empty:
                    log.info(
                        "Top %s differences for %s vs %s:%s%s",
                        value_column,
                        run_a,
                        run_b,
                        "\n",
                        top_diff[
                            ["variable", "difference", "p_value", "effect_size_label"]
                        ].to_string(index=False),
                    )
    else:
        log.info("No meta run comparisons configured. Skipping statistical tests for %s.", output_dir)

    if include_temporal_comparison:
        for value_column in ("importance_mean", "relative_importance"):
            temporal_comparison = _compare_temporal_modes(df, value_column)
            if temporal_comparison.empty:
                log.info(
                    "Temporal comparison unavailable for %s (%s) due to missing data.",
                    output_dir,
                    value_column,
                )
                continue

            comparison_path = (
                out_dir
                / f"comparison_multitemporal_vs_monotemporal_{value_column}.csv"
            )
            temporal_comparison.to_csv(comparison_path, index=False)
            log.info(
                "Saved %s temporal comparison (multitemporal vs monotemporal) to %s",
                value_column,
                comparison_path,
            )

            top_diff = (
                temporal_comparison.reindex(
                    temporal_comparison["difference"].abs().sort_values(ascending=False).index
                )
                .head(5)
                .copy()
            )
            if not top_diff.empty:
                log.info(
                    "Top %s differences for multitemporal vs monotemporal:%s%s",
                    value_column,
                    "\n",
                    top_diff[
                        ["variable", "difference", "p_value", "effect_size_label"]
                    ].to_string(index=False),
                )


def analyse_feature_importances(
    records: Iterable[FeatureImportanceRecord],
    comparisons: Sequence[tuple[str, str]],
    output_directory: Path,
) -> None:
    """Entry point that orchestrates collection, summary, and comparisons."""

    df = records_to_dataframe(list(records))
    if df.empty:
        log.error("No feature importance data collected. Check configuration.")
        return

    available_runs = {str(run) for run in df["meta_run"].dropna().unique()}
    expanded_comparisons = list(comparisons)
    mandatory_pair = ("all_points", "gbif_only")
    if (
        mandatory_pair not in expanded_comparisons
        and set(mandatory_pair).issubset(available_runs)
    ):
        expanded_comparisons.append(mandatory_pair)

    out_dir = _ensure_output_dir(output_directory)
    _write_analysis_outputs(df, out_dir, expanded_comparisons)

    seasons_seen: dict[str, str] = {}
    for season_value in df["season"].dropna():
        season_str = str(season_value).strip()
        if not season_str:
            continue
        lower = season_str.lower()
        seasons_seen.setdefault(lower, season_str)

    for season_key, season_label in sorted(seasons_seen.items()):
        season_mask = df["season"].astype(str).str.strip().str.lower() == season_key
        season_df = df[season_mask].copy()
        if season_df.empty:
            continue
        season_dir = out_dir / f"season_{_sanitize_name(season_key)}"
        log.info(
            "Generating season-specific analysis for %s in %s", season_label, season_dir
        )
        _write_analysis_outputs(
            season_df,
            season_dir,
            expanded_comparisons,
            include_temporal_comparison=True,
        )


def compute_vif_outputs(
    requests: Sequence[VIFComputationRequest],
    output_directory: Path,
) -> None:
    """Compute or collect VIF values for combos and persist them separately."""

    if not requests:
        log.info("No VIF computation requests detected. Skipping VIF generation.")
        return

    total = len(requests)
    combined_rows: list[dict[str, object]] = []
    per_run: dict[str, list[dict[str, object]]] = {}

    progress = tqdm(
        requests,
        total=total,
        desc="Computing VIF values",
        unit="combo",
    )

    for index, request in enumerate(progress, start=1):
        log.info(
            "Computing VIF values for combo '%s' (%d/%d) in meta run '%s'",
            request.combo_name,
            index,
            total,
            request.meta_run,
        )

        season = request.season
        temporal = request.temporal
        test_year = request.test_year

        vif_path = _locate_vif_file(
            request.combo_name,
            request.feature_importance_dirs,
            season=season,
            engine="elapid",
            test_year=test_year,
            temporal=temporal,
        )

        if vif_path:
            mapping = _load_vif_mapping(vif_path)
            source_path = str(vif_path)
            source_type = "existing_file"
            if mapping:
                log.info(
                    "Loaded existing VIF table for combo '%s' from %s",
                    request.combo_name,
                    vif_path,
                )
            else:
                log.warning(
                    "Existing VIF table %s for combo '%s' did not contain usable data",
                    vif_path,
                    request.combo_name,
                )
        else:
            mapping = _compute_vif_from_predictors(
                request.combo_name,
                request.predictor_roots,
                request.variables,
                season=season,
                engine="elapid",
                test_year=test_year,
                extra_roots=request.extra_roots,
                temporal=temporal,
            )
            source_path = "computed_from_predictors"
            source_type = "computed"

        variables = sorted({*request.variables, *mapping.keys()})
        if not variables:
            log.warning(
                "No variables associated with combo '%s'; skipping VIF output",
                request.combo_name,
            )
            continue

        test_year_value = int(test_year) if test_year is not None and not pd.isna(test_year) else None

        for variable in variables:
            vif_value = mapping.get(variable, np.nan)
            row = {
                "meta_run": request.meta_run,
                "combo_name": request.combo_name,
                "season": season,
                "temporal": temporal,
                "test_year": test_year_value,
                "variable": variable,
                "vif": float(vif_value) if not pd.isna(vif_value) else np.nan,
                "vif_source": source_path,
                "vif_source_type": source_type,
            }
            combined_rows.append(row)
            per_run.setdefault(request.meta_run, []).append(row)

    if not combined_rows:
        log.warning("No VIF values were computed or loaded; no VIF files will be written.")
        return

    out_root = _ensure_output_dir(output_directory) / "vif_outputs"
    out_root.mkdir(parents=True, exist_ok=True)

    combined_df = pd.DataFrame(combined_rows)
    combined_path = out_root / "vif_all_records.csv"
    combined_df.to_csv(combined_path, index=False)
    log.info("Saved combined VIF table to %s", combined_path)

    summary_df = _summarise_by_variable(combined_df, "vif")
    if not summary_df.empty:
        summary_path = out_root / "summary_vif.csv"
        summary_df.to_csv(summary_path, index=False)
        log.info("Saved aggregated VIF summary to %s", summary_path)

    for meta_run, rows in per_run.items():
        df = pd.DataFrame(rows)
        run_path = out_root / f"vif_{_sanitize_name(meta_run)}.csv"
        df.to_csv(run_path, index=False)
        log.info("Saved VIF table for %s to %s", meta_run, run_path)

def _resolve_path(path_value: str, base_dir: Path) -> Path:
    """Resolve ``path_value`` relative to ``base_dir`` if required."""

    path = Path(path_value).expanduser()
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return path


def load_analysis_config(config_path: Path) -> AnalysisConfig:
    """Load :class:`AnalysisConfig` from the provided YAML file."""

    if not config_path.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {config_path}."
            " Please create it using the template that ships with the script."
        )

    with config_path.open("r", encoding="utf-8") as handle:
        raw_config = yaml.safe_load(handle) or {}

    base_dir = config_path.parent

    raw_meta_runs = raw_config.get("meta_runs") or {}
    if not raw_meta_runs:
        raise ValueError("Configuration must define at least one entry under 'meta_runs'.")

    meta_runs: dict[str, MetaRunConfig] = {}
    for label, values in raw_meta_runs.items():
        if not isinstance(values, dict):
            raise ValueError(f"Invalid configuration for meta run '{label}'.")

        best_runs_value = values.get("best_runs")
        if not best_runs_value:
            raise ValueError(f"Meta run '{label}' is missing the 'best_runs' path.")

        feature_dirs_value = values.get("feature_importance_dirs") or []
        if not feature_dirs_value:
            raise ValueError(
                f"Meta run '{label}' must define at least one 'feature_importance_dirs' entry."
            )

        feature_dirs = tuple(
            _resolve_path(str(path_value), base_dir) for path_value in feature_dirs_value
        )

        best_runs_path = _resolve_path(str(best_runs_value), base_dir)
        if best_runs_path.is_dir():
            best_runs_path = best_runs_path / "best_runs.csv"

        meta_runs[label] = MetaRunConfig(
            label=label,
            best_runs_path=best_runs_path,
            feature_importance_dirs=feature_dirs,
        )

    comparisons_raw = raw_config.get("comparisons") or []
    comparisons: list[tuple[str, str]] = []
    for item in comparisons_raw:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise ValueError("Each comparison entry must be a two-item list or tuple.")
        comparisons.append((str(item[0]), str(item[1])))

    output_value = raw_config.get("output_directory", "analysis_outputs")
    output_dir = _resolve_path(str(output_value), base_dir)

    compute_vif = bool(raw_config.get("compute_vif", True))

    return AnalysisConfig(
        meta_runs=meta_runs,
        comparisons=comparisons,
        output_directory=output_dir,
        compute_vif=compute_vif,
    )


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for the script."""

    parser = argparse.ArgumentParser(
        description="Aggregate and compare ELAPID permutation feature importances."
    )
    parser.add_argument(
        "-c",
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to the YAML configuration file.",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    args = parse_args()
    config = load_analysis_config(args.config)

    all_records: list[FeatureImportanceRecord] = []
    all_vif_requests: list[VIFComputationRequest] = []
    for meta_config in config.meta_runs.values():
        records, vif_requests = collect_feature_importances(meta_config)
        all_records.extend(records)
        all_vif_requests.extend(vif_requests)

    analyse_feature_importances(
        all_records,
        comparisons=config.comparisons,
        output_directory=config.output_directory,
    )

    if config.compute_vif:
        compute_vif_outputs(all_vif_requests, config.output_directory)
    else:
        log.info("VIF computation disabled via configuration flag. Skipping VIF analysis.")


if __name__ == "__main__":
    main()
