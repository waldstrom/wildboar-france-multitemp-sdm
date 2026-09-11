# Quick start

[Documentation home](README.md) · [Configuration](CONFIGURATION.md) · [Data](../data/README.md)

## Environment

Use the repository root as the working directory, including for module invocations. This project is a collection of scripts, not an installable package with a console command. `pyproject.toml` currently configures pytest.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Windows PowerShell activation is `.\.venv\Scripts\Activate.ps1`. The existing [setup.sh](../setup.sh) and [setup.ps1](../setup.ps1) are installation conveniences, not environment archives.

The source uses Python 3.10+ syntax. Rasterio/GDAL, GeoPandas/PROJ and LightGBM depend on compatible native libraries; LightGBM can require OpenMP. If wheels are unavailable for your platform, create a compatible geospatial environment with your usual package manager, then install remaining requirements. Record resolved Python and native-library versions. The repository does not claim compatibility with every dependency's latest release or supply the exact historical modelling environment.

## Checks without research data

Repository hygiene needs only Python. Documentation/config validation also needs PyYAML.

```bash
python -m pip install PyYAML
python scripts/validate_repository.py
python scripts/validate_documentation.py
python -m compileall -q scripts review review-corrections.py review-finalize-existing-run.py
```

These checks validate repository structure, YAML, links and metadata. They do not evaluate model performance or certify input preparation.

For the synthetic/unit tests:

```bash
python -m pip install -r requirements-dev.txt
python -m pytest
```

## Prepare inputs

1. Follow [data/README.md](../data/README.md) and the predictor acquisition guides.
2. Harmonize occurrence labels before loading: Winter `y` is September `y` through February `y+1`; Summer `y` is March through August `y`.
3. Confirm the actual coordinate CRS. Historical column names `dd long` and `dd lat` are not evidence that the values are geographical degrees.
4. Match every raster's CRS, transform, extent, shape, pixel size, mask and nodata. Keep units consistent across years.
5. Prepare source/year/season-specific bias surfaces and check paths expanded from the chosen YAML.
6. Record versions and checksums using the [provenance template](DATA_PROVENANCE_TEMPLATE.md).

GBIF-only experiments still need environmental predictors and configured bias inputs. They are not bundled-data demos. Obtain RRN and SNCF collision records from the respective owners.

## Run the reviewed configuration families

```bash
python -m scripts.meta_run_leave_one_year --config configs/reviewed/loyo_all_sources.yaml
python -m scripts.meta_run_leave_one_year_gbif --config configs/reviewed/loyo_gbif.yaml
```

Both meta runners enforce MaxEnt and GBM. Across both files, the configured years yield 104 fits before auxiliary diagnostics: 2 occurrence scenarios × 2 temporal representations × 2 engines × (7 winters + 6 summers). Outputs use timestamped experiment directories. Costs depend on input size, caching and parallelism; full modelling is outside CI.

These runners execute supplied parameters. They do not perform the complete nested tuning procedure described in the manuscript. Read [reproduction status](REPRODUCIBILITY.md#tuning-and-outer-evaluation) before claiming numerical reproduction.

## Run one seasonal scenario first

An aggregate contains a `scenarios` list; a seasonal runner expects the inner `config` mapping. Extract a local copy:

```bash
python -c "from pathlib import Path; import yaml; c=yaml.safe_load(Path('configs/reviewed/loyo_all_sources.yaml').read_text()); s=next(s for s in c['scenarios'] if s['key']=='winter_multitemporal'); p=Path('exps/local-configs/winter_multitemporal.yaml'); p.parent.mkdir(parents=True, exist_ok=True); p.write_text(yaml.safe_dump(s['config'], sort_keys=False))"
```

Edit that copy's experiment name, test year and engines. For example, `run_test_year: 2019` requests one outer fold and `model.engines: [elapid]` limits a direct seasonal run to MaxEnt. Preserve the full `multitemporal.years` list because it supplies training years.

```bash
python -m scripts.multitemporal --config exps/local-configs/winter_multitemporal.yaml
```

Use `scripts.monotemporal` for an extracted monotemporal scenario. Direct runners can ask about stack reuse. Changed years, candidates, mask, grid or hunting inclusion require a fresh or explicitly rebuilt stack.

## Inspect review operations

```bash
python review-corrections.py plan
python review-corrections.py run --dry-run
python review-finalize-existing-run.py --help
```

The dry run writes a plan and inventories required data without fitting models. Finalization requires a complete saved review run; see [review utilities](../review/README.md).

## Troubleshooting

| Symptom | Check |
|---|---|
| `No module named scripts` | Use `python -m scripts.<module>` from the repository root |
| Missing old root YAML | Consult the [migration table](CONFIG_MIGRATION.md) |
| Empty occurrence subset | Check year/season labels, source filter, coordinates and mask |
| Raster/predictor mismatch | Check alignment, actual names and selected-predictor lists |
| Enabled hunting absent from a model | Distinguish candidate inclusion, stack inclusion and fold-specific pruning |
| A run uses old variables | Inspect caches and rebuild in an isolated experiment |
| Export differs from the paper | Compare resolved config, fold membership, repeats and run provenance |
